# scene_check.py - 공장 장면 점검 + 외란 보정 (Isaac Sim 5.1)
# ---------------------------------------------------------------------------
# 1) 장면 사진: 고정·손목 카메라 이미지를 PNG로 저장, 밝기·상자 픽셀 크기 확인
# 2) 외란 사진: 조명·깜빡임·카메라 단계별 고정 카메라 이미지 (포스터 그림 재료)
# 3) 스크립트 제어기 시험: 외란 없음 + 물리 외란(마찰·질량·무게중심) 단계별로
#    전문가 제어기가 성공하는지. 전문가도 실패하는 단계 = 물리적으로 불가능한 구간이라
#    정책의 붕괴 지점과 구분해서 보고해야 한다.
#
# 실행: /data/isaacsim/python.sh -u checks/scene_check.py
#   CHECK=images      사진만
#   CHECK=expert      제어기 시험만
#   (기본: 둘 다)
# 결과: /data/jinju/factory/scene_check/
# ---------------------------------------------------------------------------
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factory_config as C
import disturbances as D

OUT = os.environ.get("CHECK_OUT", "/data/jinju/factory/scene_check")
MODE = os.environ.get("CHECK", "all")
os.makedirs(OUT, exist_ok=True)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

from PIL import Image
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
import factory_scene as F

s = F.build_scene(simulation_app)
print(f"[check] 장면 준비 완료 | 배경 조명 {s.n_bg_lights_off}개 끔", flush=True)
report = {"bg_lights_disabled": s.n_bg_lights_off, "exposure": s.exposure,
          "conveyor_bbox": s.conveyor_bbox, "warehouse_bbox": s.warehouse_bbox}
print(f"[check] 꺼진 외부 조명 {s.foreign_lights_off}", flush=True)
print(f"[check] 노출 {s.exposure} | 컨베이어 bbox {s.conveyor_bbox} | 창고 bbox {s.warehouse_bbox}", flush=True)


def snap(tag, n_settle=8):
    for _ in range(n_settle):
        s.world.step(render=True)
    o = F.grab_rgb(s, s.over_cam)
    w = F.grab_rgb(s, s.wrist_cam)
    Image.fromarray(o).save(os.path.join(OUT, f"over_{tag}.png"))
    Image.fromarray(w).save(os.path.join(OUT, f"wrist_{tag}.png"))
    return o, w


def box_pixels(img):
    """골판지색에 가까운 픽셀 수 (상자가 화면에 얼마나 보이는지)."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return int(((r > g + 15) & (g > b + 10) & (r > 60)).sum())


if MODE in ("all", "images"):
    clean, _ = D.parse("")
    F.reset_episode(s, (0.44, 0.16))
    F.apply_episode_disturbance(s, clean, D.episode_draws(clean, 0, 0))
    F.move_to_start(s)
    o, w = snap("clean")
    report["clean"] = {"over_mean": round(float(o.mean()), 1), "over_std": round(float(o.std()), 1),
                       "wrist_mean": round(float(w.mean()), 1),
                       "box_px_over": box_pixels(o), "box_px_wrist": box_pixels(w)}
    print(f"[check] 기준 이미지 {report['clean']}", flush=True)

    # 천장등 세기 보정: 기준 이미지 평균 밝기가 110~140(8bit) 근처가 되는 값을 찾는다
    report["light_calib"] = {}
    for inten in (3000, 6000, 10000, 15000, 20000, 30000, 50000):
        for lt in s.ceiling:
            lt.GetIntensityAttr().Set(float(inten))
        o, _ = snap(f"calib{inten}")
        report["light_calib"][inten] = round(float(o.mean()), 1)
        print(f"[check] 천장등 {inten:6d} -> 고정카메라 평균 {o.mean():.1f}", flush=True)
    F.set_light_scale(s, 1.0)

    for name in ("light", "camera"):
        for lv in range(1, D.N_LEVELS + 1):
            p, _ = D.parse(f"{name}={lv}")
            F.apply_episode_disturbance(s, p, D.episode_draws(p, 0, 0))
            o, _ = snap(f"{name}{lv}")
            report[f"{name}{lv}"] = {"over_mean": round(float(o.mean()), 1), "box_px_over": box_pixels(o)}
            print(f"[check] {name}{lv}: {report[f'{name}{lv}']}", flush=True)
    p, _ = D.parse("")
    F.apply_episode_disturbance(s, p, D.episode_draws(p, 0, 0))
    F.set_light_scale(s, D.FLICKER_LOW)
    o, _ = snap("flicker_dark")
    report["flicker_dark"] = {"over_mean": round(float(o.mean()), 1)}

if MODE in ("all", "expert"):
    pts = C.eval_positions(n=int(os.environ.get("CHECK_N", "5")), seed=99)
    specs = [""] + [f"friction={i}" for i in range(1, 6)] + [f"mass={i}" for i in range(1, 6)] \
        + [f"com={i}" for i in range(1, 6)]
    if os.environ.get("CHECK_SPECS"):
        specs = [x if x != "clean" else "" for x in os.environ["CHECK_SPECS"].split(";")]
    report["expert"] = {}
    for spec in specs:
        params, _ = D.parse(spec)
        ok = 0
        detail = []
        for k, (bx, by) in enumerate(pts):
            F.reset_episode(s, (bx, by))
            F.apply_episode_disturbance(s, params, D.episode_draws(params, 0, k))
            applied = F.read_back(s)
            s.franka.gripper.open()
            for _ in range(60):
                s.world.step(render=False)
            F.move_to_start(s)
            ctrl = PickPlaceController(name="pp", gripper=s.franka.gripper, robot_articulation=s.franka,
                                       end_effector_initial_height=C.TABLE_Z + 0.3)
            pick, place = F.box_pos(s), F.tray_place_target()
            max_z = float(pick[2])
            for _ in range(2000):
                q = np.array(s.franka.get_joint_positions())
                s.franka.apply_action(ctrl.forward(picking_position=pick, placing_position=place,
                                                   current_joint_positions=q))
                s.world.step(render=False)
                max_z = max(max_z, float(F.box_pos(s)[2]))
                if ctrl.is_done():
                    break
            for _ in range(60):
                s.world.step(render=False)
            cond = F.success_conditions(s, max_z)
            succ = all(cond.values())
            ok += succ
            detail.append({"box": [bx, by], "success": succ, "cond": cond})
        report["expert"][D.label(spec)] = {"success": ok, "n": len(pts), "applied_last": applied,
                                           "episodes": detail}
        print(f"[check] 전문가 {D.label(spec):12s} {ok}/{len(pts)} 성공 | 적용값 {applied}", flush=True)

with open(os.path.join(OUT, "scene_check.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
print(f"[check] 완료 -> {OUT}", flush=True)
import omni.replicator.core as rep
rep.orchestrator.stop()
simulation_app.close()
