# eval_client.py - 외란을 주입하며 정책 성공률 측정 (Isaac Sim 5.1)
# ---------------------------------------------------------------------------
# 정책은 policy_server.py(conda lerobot)가 돌리고, 여기서는 공장 장면 + 외란 +
# 판정만 한다.
#
#   DIST="friction=3" OUT_JSON=/data/jinju/factory/eval/act_friction3.json \
#   POLICY_PORT=5555 /data/isaacsim/python.sh -u eval_client.py
#
# DIST 문법은 disturbances.py 참고. 평가 위치(EVAL_N, EVAL_SEED)와 에피소드별 외란
# 무작위 요소는 시드로 고정 -> 모든 모델·조건이 같은 상자 위치, 같은 외란 방향을 받는다.
# ---------------------------------------------------------------------------
import os
import json
import time
import socket
import struct
import pickle
import numpy as np

import factory_config as C
import disturbances as D

HOST = os.environ.get("POLICY_HOST", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "5555"))
DIST = os.environ.get("DIST", "")
PARAMS, LEVELS = D.parse(DIST)
OUT_JSON = os.environ.get("OUT_JSON", f"/data/jinju/factory/eval/{D.label(DIST)}.json")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "")
ACTION_REPEAT = max(1, int(os.environ.get("ACTION_REPEAT", str(C.TRAIN_STRIDE))))
POSITIONS = C.eval_positions()
_only = os.environ.get("EVAL_ONLY_EP", "").strip()
EPISODES = [(i, p) for i, p in enumerate(POSITIONS)
            if not _only or i in {int(v) for v in _only.split(",")}]
SAVE_FAIL_FRAMES = os.environ.get("SAVE_FAIL_FRAMES", "0") == "1"
# RECORD_GIF=<경로>: 첫 에피소드를 고정 카메라 GIF로 저장 (화면 앱·포스터 시연용)
RECORD_GIF = os.environ.get("RECORD_GIF", "").strip()
RENDER_LOOKAHEAD = int(os.environ.get("RENDER_LOOKAHEAD", "0"))
# SAVE_OBS_DIR=<폴더>: 에피소드마다 정책이 받은 처음 3개 관측(고정 카메라)을 저장 (렌더링 방식 비교용)
SAVE_OBS_DIR = os.environ.get("SAVE_OBS_DIR", "").strip()


# ---- 0% 이후 단계 생략 --------------------------------------------------------
# 같은 정책(같은 결과 폴더)에서 같은 외란(조합)의 더 낮은 단계가 이미 0/N 이면, 더 센 단계도
# 평가하지 않고 0%로 기록한다 (시간 절약). 어느 단계 때문에 건너뛰었는지는 JSON의
# "skipped_after"에만 남는다. 켜진 외란이 모두 같은 단계일 때만 적용 (단독, 조합, all, all_vib).
# 끄려면 SKIP_AFTER_ZERO=0.
def _zero_below():
    on = {k: v for k, v in LEVELS.items() if v > 0}
    if not on or len(set(on.values())) != 1 or _only:
        return None
    lv = next(iter(on.values()))
    for low in range(1, lv):
        lab = D.label(",".join(f"{k}={low}" for k in on))
        f = os.path.join(os.path.dirname(OUT_JSON), f"{lab}.json")
        if os.path.exists(f):
            r = json.load(open(f))
            if r.get("n_episodes") and r.get("n_success") == 0:
                return lab
    return None


if os.environ.get("SKIP_AFTER_ZERO", "1") == "1" and (_src := _zero_below()):
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"model": MODEL_LABEL, "dist": DIST, "label": D.label(DIST), "levels": LEVELS,
                   "params": PARAMS, "n_episodes": 0, "n_success": 0, "success_rate": 0.0,
                   "skipped_after": _src, "failure_modes": {}, "blank_camera_frames": 0,
                   "camera_frames_total": 0, "eval_seed": C.EVAL_SEED, "episodes": []},
                  f, indent=2, ensure_ascii=False)
    print(f"[eval] DONE {D.label(DIST)} success=0% ({_src} 0% 이후 단계) -> {OUT_JSON}", flush=True)
    raise SystemExit(0)


# ---- 소켓 ------------------------------------------------------------------
def send(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv(sock):
    raw = recvall(sock, 4)
    if raw is None:
        return None
    data = recvall(sock, struct.unpack(">I", raw)[0])
    return None if data is None else pickle.loads(data)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
for attempt in range(120):
    try:
        sock.connect((HOST, PORT))
        break
    except OSError:
        time.sleep(1.0)
else:
    raise SystemExit(f"[eval] 정책 서버 {HOST}:{PORT} 연결 실패")
print(f"[eval] 정책 서버 연결 {HOST}:{PORT} | DIST='{DIST}' -> {PARAMS}", flush=True)

from PIL import Image
import io


def png(img):
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="PNG")
    return b.getvalue()


def act(w, o, state, with_images):
    msg = {"cmd": "act", "state": [float(x) for x in state]}
    if with_images:
        msg.update(enc="png", wrist=png(w), over=png(o), shape=list(w.shape))
    send(sock, msg)
    r = recv(sock)
    if r is None or "action" not in r:
        raise RuntimeError(f"[eval] 서버 응답 오류: {r}")
    return np.asarray(r["action"], dtype=float), bool(r.get("need_obs", True))


# ---- 시뮬 ------------------------------------------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})
from isaacsim.core.utils.types import ArticulationAction
import factory_scene as F

s = F.build_scene(simulation_app)


def binarize(cmd):
    cmd = np.asarray(cmd, dtype=float).copy()
    closing = float(np.mean(cmd[7:9])) < C.GRIPPER_CLOSE_THRESH
    cmd[7:9] = C.GRIPPER_CLOSED if closing else C.GRIPPER_OPEN
    return cmd


def failure_mode(r):
    """실패를 유형으로 나눈다 (논문의 실패 유형 분석용)."""
    if r["success"]:
        return "success"
    if not r["grasp_attempted"]:
        return "no_grasp"                     # 상자를 잡으러 가지도 않음 / 엉뚱한 곳
    if not r["cond"]["lifted"]:
        return "grasp_no_lift"                # 잡았지만 못 들어올림 (미끄러짐, 무게)
    if r["dropped"]:
        return "dropped_in_transit"           # 들고 가다 떨어뜨림
    if not r["cond"]["in_tray"]:
        return "missed_tray"                  # 트레이 밖에 놓음
    if not r["cond"]["released"]:
        return "not_released"
    return "other"


results = []
gif_frames = []
t0 = time.time()
for ep, (bx, by) in EPISODES:
    draws = D.episode_draws(PARAMS, C.EVAL_SEED, ep)
    F.reset_episode(s, (bx, by))
    F.apply_episode_disturbance(s, PARAMS, draws)
    applied = F.read_back(s)
    F.settle(s, 10)
    F.move_to_start(s)
    send(sock, {"cmd": "reset"})
    recv(sock)

    max_z = float(F.box_pos(s)[2])
    cmd, need_obs, k_obs = None, True, 0
    hold, success, steps = 0, False, C.MAX_EVAL_STEPS
    grasp_attempted, was_lifted, dropped = False, False, False
    # 재시도 분석: 그리퍼를 연 상태에서 닫으라고 명령한 횟수, 상자를 들어 올린 횟수
    n_grasp_cmds, n_lifts, prev_closed, prev_lifted = 0, 0, False, False
    n_saved_obs = 0
    min_ee_box = 9.0
    frames = []
    for step in range(C.MAX_EVAL_STEPS):
        q = np.array(s.franka.get_joint_positions(), dtype=np.float32)[:9]
        if step % ACTION_REPEAT == 0 or cmd is None:
            F.set_lights_for_step(s, PARAMS, draws, k_obs)
            F.set_camera_shake_for_step(s, PARAMS, draws, k_obs)
            k_obs += 1
            send_img = need_obs or step == 0
            w = o = None
            if send_img:
                w = F.grab_rgb(s, s.wrist_cam)
                o = F.grab_rgb(s, s.over_cam)
                if SAVE_FAIL_FRAMES and len(frames) < 400:
                    frames.append(o)
                if SAVE_OBS_DIR and n_saved_obs < 3:
                    os.makedirs(SAVE_OBS_DIR, exist_ok=True)
                    Image.fromarray(o).save(os.path.join(SAVE_OBS_DIR, f"ep{ep:02d}_obs{n_saved_obs}_step{step}.png"))
                    n_saved_obs += 1
            raw, need_obs = act(w, o, q, send_img)
            cmd = binarize(raw)
            closed = float(cmd[7]) < C.GRIPPER_CLOSE_THRESH
            if closed and not prev_closed:
                n_grasp_cmds += 1
            prev_closed = closed
        s.franka.apply_action(ArticulationAction(joint_positions=cmd))
        # 렌더링은 정책이 다음에 화면을 볼 때만 필요하다. RENDER_LOOKAHEAD=n이면 관측 직전 n개
        # 물리 스텝만 렌더링한다(물리는 매 스텝 그대로). 0이면 예전처럼 매 스텝 렌더링.
        # ACT는 한 번 보고 100 정책스텝을 실행하므로 렌더링이 크게 줄어든다.
        gif_step = RECORD_GIF and ep == EPISODES[0][0] and (step + 1) % 6 == 0
        render_now = (RENDER_LOOKAHEAD <= 0 or step < 2 or gif_step
                      or (need_obs and (step % ACTION_REPEAT) >= ACTION_REPEAT - RENDER_LOOKAHEAD))
        s.world.step(render=render_now)
        if RECORD_GIF and ep == EPISODES[0][0] and step % 6 == 5:
            gif_frames.append(Image.fromarray(F.grab_rgb(s, s.over_cam)))

        b = F.box_pos(s)
        max_z = max(max_z, float(b[2]))
        ee = F.get_ee_position(s)
        min_ee_box = min(min_ee_box, float(np.linalg.norm(ee[:2] - b[:2])))
        if q[7] < C.GRIPPER_CLOSING_RAW_THRESH:
            grasp_attempted = True
        lifted_now = float(b[2]) > C.BOX_Z + C.SUCCESS_MIN_LIFT
        if lifted_now and not prev_lifted:
            n_lifts += 1
        prev_lifted = lifted_now
        if lifted_now:
            was_lifted = True
        elif was_lifted and not C.box_in_tray_xy(float(b[0]), float(b[1])) \
                and float(b[2]) < C.BOX_Z + 0.01:
            dropped = True                    # 들렸다가 트레이 밖 작업대로 떨어짐
        cond = F.success_conditions(s, max_z)
        if all(cond.values()):
            hold += 1
            if hold >= C.SUCCESS_HOLD_STEPS:
                success, steps = True, step
                break
        else:
            hold = 0

    cond = F.success_conditions(s, max_z)
    r = {"episode": ep, "box_xy": [bx, by], "success": success, "steps": steps,
         "cond": {k: bool(v) for k, v in cond.items()}, "grasp_attempted": grasp_attempted,
         "dropped": dropped, "max_box_z": round(max_z, 4),
         "n_grasp_attempts": n_grasp_cmds, "n_lifts": n_lifts,
         "recovered": bool(success and (dropped or n_lifts > 1)),
         "min_ee_box": round(min_ee_box, 4),
         "final_box": [round(float(v), 4) for v in F.box_pos(s)],
         "applied": applied,
         "cam_offset": [round(v, 4) for v in draws["cam_offset"]],
         "com_offset": [round(v, 4) for v in draws["com_offset"]]}
    r["failure_mode"] = failure_mode(r)
    results.append(r)
    if SAVE_FAIL_FRAMES and not success and frames:
        d = os.path.splitext(OUT_JSON)[0] + "_frames"
        os.makedirs(d, exist_ok=True)
        for i in (0, len(frames) // 2, len(frames) - 1):
            Image.fromarray(frames[i]).save(os.path.join(d, f"ep{ep:02d}_{i:03d}.png"))
    print(f"[eval] ep {ep:2d} box=({bx:.3f},{by:.3f}) {r['failure_mode']:18s} "
          f"steps={steps} maxz={max_z:.3f} grasps={n_grasp_cmds} lifts={n_lifts} | {time.time() - t0:.0f}s",
          flush=True)
    if RECORD_GIF and gif_frames and ep == EPISODES[0][0]:
        gif_frames[0].save(RECORD_GIF, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        print(f"[eval] GIF 저장 -> {RECORD_GIF}", flush=True)

n = len(results)
ns = sum(r["success"] for r in results)
modes = {}
for r in results:
    modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
summary = {
    "model": MODEL_LABEL, "dist": DIST, "label": D.label(DIST), "levels": LEVELS, "params": PARAMS,
    "n_episodes": n, "n_success": ns, "success_rate": ns / n if n else None,
    "failure_modes": modes, "action_repeat": ACTION_REPEAT, "eval_seed": C.EVAL_SEED,
    "mean_grasp_attempts": round(sum(r["n_grasp_attempts"] for r in results) / n, 3) if n else None,
    "n_retried": sum(r["n_grasp_attempts"] > 1 for r in results),
    "n_recovered": sum(r["recovered"] for r in results),
    "blank_camera_frames": s.frame_stats["blank"], "camera_frames_total": s.frame_stats["total"],
    "wall_time_s": round(time.time() - t0, 1), "episodes": results,
}
os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
if s.frame_stats["blank"]:
    print(f"[eval] [!] 검은 프레임 {s.frame_stats['blank']}개 - 이 성공률은 믿을 수 없다", flush=True)
print(f"[eval] DONE {D.label(DIST)} success={ns}/{n} modes={modes} -> {OUT_JSON}", flush=True)
try:
    send(sock, {"cmd": "bye"})
    recv(sock)
except Exception:
    pass
sock.close()
simulation_app.close()
