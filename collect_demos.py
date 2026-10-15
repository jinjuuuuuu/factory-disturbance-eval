# collect_demos.py - 공장 장면에서 전문가 시연 수집 (Isaac Sim 5.1)
# ---------------------------------------------------------------------------
# IK 경유점 제어기(ik_expert.py)가 상자를 집어 트레이에 넣는 시연을 모은다.
# (RMPflow PickPlaceController는 상자를 쥔 채 운반할 때 팔이 흔들려 외란 없이도
#  0~5/6만 성공했다. IK 제어기는 외란 없음 6/6, 1.2kg 6/6 - checks/ik_check.py)
#
# 외란은 넣지 않는다 - 학습은 깨끗한 환경에서, 외란은 평가에서만 (재학습 없는 평가).
# 그래도 apply_episode_disturbance(기준값)를 매 에피소드 호출해 평가와 같은 경로를 탄다.
#
# 실행:  /data/isaacsim/python.sh -u collect_demos.py
#   SAVE_PATH=/data/jinju/factory/demos_v1   저장 위치
#   NUM_EPISODES=100                         목표 개수 (이어받기 지원)
#   FRESH_START=1                            기존 파일 삭제 후 처음부터
#
# 예전 프로젝트의 검증된 규칙: 그리퍼 라벨 = 실제 명령(0.0 / 0.04), 실패 시연은 버림,
# 시연 끝에 시작 자세로 돌아가는 구간까지 기록.
# ---------------------------------------------------------------------------
import os
import sys
import glob
import random
import numpy as np

import factory_config as C
import disturbances as D

SAVE_PATH = os.environ.get("SAVE_PATH", "/data/jinju/factory/demos_v1")
FRESH_START = os.environ.get("FRESH_START", "0") == "1"
SEED = int(os.environ.get("COLLECT_SEED", "1234"))
HOME_STEPS = 100
os.makedirs(SAVE_PATH, exist_ok=True)


def resume_plan():
    files = sorted(glob.glob(os.path.join(SAVE_PATH, "episode_*.npz")))
    if FRESH_START:
        for f in files:
            os.remove(f)
        print(f"[collect] FRESH_START=1 -> 기존 {len(files)}개 삭제")
        return C.NUM_EPISODES, 0
    if not files:
        return C.NUM_EPISODES, 0
    last = max(int(os.path.basename(f)[8:12]) for f in files)
    todo = max(0, C.NUM_EPISODES - len(files))
    print(f"[collect] 기존 {len(files)}개 발견 -> {todo}개 추가 (인덱스 {last + 1}부터)")
    return todo, last + 1


N_TODO, IDX0 = resume_plan()
if N_TODO == 0:
    print("[collect] 이미 목표 개수를 채웠다.")
    sys.exit(0)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

import factory_scene as F
from ik_expert import IKExpert, apply

s = F.build_scene(simulation_app)
ex = IKExpert(s)
print(f"[collect] 장면 준비 완료 | 손가락 드라이브 {s.finger_drives} | 저장 {SAVE_PATH}", flush=True)
params, _ = D.parse("")                       # 외란 없음
rng = random.Random(SEED + IDX0)
home = np.array(C.START_POSE, dtype=float)

saved = dropped = 0
ep = 0
while saved < N_TODO:
    ep += 1
    bx, by = C.sample_box_xy(rng)
    F.reset_episode(s, (bx, by))
    F.apply_episode_disturbance(s, params, D.episode_draws(params, SEED, ep))
    s.franka.gripper.open()
    F.settle(s, 30)
    F.move_to_start(s)

    if not ex.reset(F.box_pos(s), F.tray_place_target()):
        dropped += 1
        print(f"[collect] try {ep:3d} {ex.failed}", flush=True)
        continue

    obs_l, act_l, wr_l, ov_l = [], [], [], []
    max_z = float(F.box_pos(s)[2])

    def record(cmd):
        obs_l.append(np.array(s.franka.get_joint_positions(), dtype=np.float32)[:9])
        act_l.append(np.asarray(cmd, dtype=np.float32)[:9])
        wr_l.append(F.grab_rgb(s, s.wrist_cam))
        ov_l.append(F.grab_rgb(s, s.over_cam))

    while not ex.done:
        cmd, _ = ex.step(np.array(s.franka.get_joint_positions()))
        record(cmd)
        apply(s, cmd)
        s.world.step(render=True)
        max_z = max(max_z, float(F.box_pos(s)[2]))

    # 시작 자세로 복귀 (정책이 '끝나면 돌아간다'를 배우게)
    q0 = np.array(s.franka.get_joint_positions(), dtype=float)[:9]
    for k in range(HOME_STEPS):
        a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / HOME_STEPS)
        cmd = (1 - a) * q0 + a * home
        cmd[7:9] = C.GRIPPER_OPEN
        record(cmd)
        apply(s, cmd)
        s.world.step(render=True)

    cond = F.success_conditions(s, max_z)
    if not all(cond.values()):
        dropped += 1
        print(f"[collect] try {ep:3d} box=({bx:.3f},{by:.3f}) 실패 {cond}", flush=True)
        continue

    path = os.path.join(SAVE_PATH, f"episode_{IDX0 + saved:04d}.npz")
    np.savez_compressed(path, obs=np.array(obs_l, np.float32), actions=np.array(act_l, np.float32),
                        images_wrist=np.array(wr_l, np.uint8), images_over=np.array(ov_l, np.uint8),
                        box_xy=np.array([bx, by], np.float32))
    saved += 1
    print(f"[collect] 저장 {saved}/{N_TODO} {os.path.basename(path)} | {len(obs_l)}프레임 | "
          f"box=({bx:.3f},{by:.3f}) | 검은 프레임 누적 {s.frame_stats['blank']}", flush=True)

print(f"[collect] 완료: 저장 {saved}, 실패 버림 {dropped}, "
      f"검은 프레임 {s.frame_stats['blank']}/{s.frame_stats['total']}", flush=True)
import omni.replicator.core as rep
rep.orchestrator.stop()
simulation_app.close()
