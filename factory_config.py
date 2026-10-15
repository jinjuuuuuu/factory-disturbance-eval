#!/usr/bin/env python3
# factory_config.py - 공장 장면의 모든 숫자를 담는 단일 출처
# ---------------------------------------------------------------------------
# ~/pickplace/scene_config.py 와 같은 원칙: 수집·평가·점검 스크립트가 전부 이 파일
# 하나를 읽는다. 예전 프로젝트에서 수집 조명 1500 / 평가 1000 이 어긋나 성공률이
# 에러 없이 0%가 된 적이 있다. 장면을 정의하는 숫자는 여기에만 둔다.
#
# 순수 상수 모듈이다 (isaacsim/pxr import 없음) -> conda 환경에서도 import 된다.
#
# [!] 카메라·해상도·조명·상자 외형을 바꾸면 데이터를 다시 수집해야 한다.
# ---------------------------------------------------------------------------

import os as _os

# === 작업대 =================================================================
# 로봇과 상자가 모두 작업대 위에 있다. 모든 높이는 작업대 상판(TABLE_Z) 기준이다.
# 0으로 두면 예전 프로젝트처럼 바닥 위에서 동작한다(문제 생길 때의 대피로).
TABLE_Z = float(_os.environ.get("TABLE_Z", "0.75"))
TABLE_SIZE = (1.30, 1.40)            # x, y [m]
TABLE_CENTER_XY = (0.40, 0.0)        # 로봇 베이스(원점)가 작업대 뒤쪽에 오도록
TABLE_RGB = (0.38, 0.40, 0.42)       # 도장된 강철 작업대

# === 공장 배경 ==============================================================
# 배경은 시각 요소다. 물리에 관여하지 않게 작업 영역 밖에 둔다.
#   FACTORY_BG=0 이면 배경을 싣지 않는다 (문제 분리용)
FACTORY_BG = _os.environ.get("FACTORY_BG", "1") != "0"
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
WAREHOUSE_USD = ASSET_ROOT + "/Environments/Simple_Warehouse/warehouse.usd"
WAREHOUSE_OFFSET = (0.0, 0.0, 0.0)
CONVEYOR_USD = ASSET_ROOT + "/Props/Conveyors/ConveyorBelt_A08.usd"
# 배경의 옆 라인 (실측 bbox: 길이 2.7m x 폭 1.15m x 높이 1.17m). 작업 영역 밖에 둔다.
CONVEYOR_POS = (0.45, 1.40, 0.0)
CONVEYOR_ROT_Z = 0.0

# === 작업 컨베이어 (멈춘 상태, 상자를 집는 곳) =============================
# 에셋 컨베이어(ConveyorBelt_A08)는 2.7m x 1.15m로 팔이 닿는 범위(~0.85m)보다 커서
# 배경으로만 쓰고, 작업용 벨트는 치수를 직접 정해 만든다.
# v2 배치: 로봇(원점, +x를 바라봄) 앞에 벨트가 '가로로'(y방향) 지나가고, 상자는 로봇
# 정면의 벨트 위, 트레이는 로봇 오른쪽 옆 작업대 위. (v1은 벨트가 x방향이라 로봇
# 받침 옆을 지나가며 겹쳐 보였다.)
BELT_RISE = 0.06
BELT_TOP_Z = TABLE_Z + BELT_RISE
BELT_X_CENTER = 0.50
BELT_WIDTH = 0.30                     # x 0.35 ~ 0.65
BELT_Y_RANGE = (-0.70, 0.70)          # 작업대 폭만큼 가로로 이어지는 라인
BELT_RGB = (0.12, 0.12, 0.13)         # 고무 벨트
FRAME_RGB = (0.55, 0.57, 0.60)        # 알루미늄 프레임
RAIL_H = 0.012                        # 벨트 위로 솟은 옆 레일 높이 (손가락과 안 닿게 낮게)

# === 상자 (내용물이 든 소형 포장 상자) ======================================
# Franka 그리퍼 최대 개구 8cm. 5cm면 예전 큐브와 같은 폭이라 파지 기하가 검증돼 있다.
BOX_SIZE = 0.05
BOX_RGB = (0.62, 0.45, 0.28)         # 골판지
BOX_MASS = 0.20                      # 내용물 포함 기준 질량 [kg]
BOX_Z = BELT_TOP_Z + BOX_SIZE / 2.0  # 벨트 위에 놓인 상자 중심 높이

# === 부품 트레이 (놓을 곳) ==================================================
# KLT 상자를 흉내 낸 얕은 트레이. 에셋 대신 직접 만드는 이유: 치수를 우리가 정해야
# 성공 판정(트레이 안쪽인가)을 정확히 할 수 있다.
TRAY_CENTER_XY = (0.20, -0.42)        # 로봇 오른쪽 옆 (벨트 x 0.35부터와 겹치지 않음)
TRAY_INNER = (0.16, 0.16)            # 안쪽 치수 x, y
TRAY_WALL_H = 0.03
TRAY_WALL_T = 0.008
TRAY_FLOOR_T = 0.004
TRAY_RGB = (0.10, 0.30, 0.70)        # 파란 KLT
TRAY_FLOOR_Z = TABLE_Z + TRAY_FLOOR_T          # 트레이 바닥 윗면
PLACE_Z = TRAY_FLOOR_Z + BOX_SIZE / 2.0        # 트레이 안에 놓인 상자 중심 높이
PLACE_Z_OFFSET = 0.02                          # 컨트롤러가 놓는 높이 여유

# === 상자 초기 위치 (멈춘 벨트 위) ==========================================
# 수집은 연속 랜덤, 평가는 같은 분포에서 고정 시드로 뽑는다. x는 양쪽 레일에서
# 상자 반폭 + 여유를 두고, y는 로봇 정면 ±15cm.
BOX_X_RANGE = (BELT_X_CENTER - 0.08, BELT_X_CENTER + 0.08)
BOX_Y_RANGE = (-0.15, 0.15)
MIN_DISTANCE = 0.15                  # 트레이 중심과의 최소 거리

NUM_EPISODES = int(_os.environ.get("NUM_EPISODES", "100"))
START_POSE = [0.0, -0.3, 0.0, -2.5, 0.0, 2.2, 0.8, 0.04, 0.04]

# === 이미지 / 카메라 ========================================================
IMG_W, IMG_H = 320, 240
WRIST_CAM_PRIM = "/World/Franka/panda_hand/WristCam/Camera"
OVER_CAM_XFORM = "/World/OverheadCam"
OVER_CAM_PRIM = OVER_CAM_XFORM + "/Camera"
CAM_H_APERTURE = 20.955
CAM_V_APERTURE = 15.716
CAM_CLIP_NEAR, CAM_CLIP_FAR = 0.05, 100.0

# 손목 카메라: 예전 프로젝트에서 검증된 값 그대로 (손 프레임 기준이라 작업대 높이와 무관)
WRIST_FOCAL = 16.0
WRIST_TRANSLATE = (0.15, 0.0, 0.0)
WRIST_ROTATE = (-45.0, 179.9, -89.9)

# 고정 카메라: 예전 정면 경사뷰를 작업대 높이만큼 올린다. 공장 설비에 고정된 카메라이며
# '카메라 외란'은 이 카메라의 거치대가 진동으로 틀어진 상황을 모사한다.
# v2: 트레이가 로봇 옆(y=-0.42)에 있어 화각을 넓히고(60->48mm, 폭 약 1.1m) 카메라를
# y=-0.2로 옮겨 벨트 위 상자와 트레이가 함께 들어오게 했다.
OVER_FOCAL = 48.0
OVER_TRANSLATE = (2.0, -0.2, TABLE_Z + 2.0)
OVER_ROTATE = (40.0, 0.0, 89.99)

# === 조명 (공장 천장등) =====================================================
# 창고 안은 지붕이 있어 돔 라이트가 거의 들어오지 않는다. 작업대 위 천장등 두 개를
# 직접 두고 이것만 외란 대상으로 삼는다. 약한 돔 라이트는 그림자 속이 새까맣게
# 되지 않게 하는 보조광이다.
CEILING_LIGHTS = [                   # (x, y, z) 작업대 위 2.2m
    (0.25, 0.30, TABLE_Z + 2.2),
    (0.25, -0.30, TABLE_Z + 2.2),
]
CEILING_LIGHT_SIZE = (1.2, 0.3)      # 형광등 판 크기
CEILING_INTENSITY = float(_os.environ.get("CEILING_INTENSITY", "30000.0"))
AMBIENT_DOME_INTENSITY = 300.0

# === 그리퍼 (예전 프로젝트의 교훈을 그대로 가져온다) ========================
# 라벨과 실제 명령을 같은 값으로 통일한다. 둘이 어긋나 성공률이 0%였던 적이 있다.
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE_THRESH = (GRIPPER_CLOSED + GRIPPER_OPEN) / 2.0
GRIPPER_CLOSING_RAW_THRESH = 0.035
GRIPPER_RELEASED_THRESH = 0.030
# 손가락 패드 마찰. 상자 마찰 외란이 실제로 먹히려면 접촉 마찰이 상자 쪽 값으로
# 결정돼야 한다 -> 양쪽 모두 combine mode 'min', 손가락은 높게 둔다.
FINGER_FRICTION = 1.0
# 손가락 드라이브. Isaac Sim 기본값(maxForce 7.2N, stiffness 400)은 0.3kg 상자도 운반 중
# 놓쳤다. 70N/4000은 한쪽 손가락만 구동되는 구조라 상자를 밀어 팔이 흔들렸다.
# 40N/1000 + IK 경유점 제어기에서 외란 없음 6/6, 1.2kg 6/6 (checks/ik_check.py).
# 실제 Franka Hand 연속 파지력(70N) 이내다.
FINGER_MAX_FORCE = float(_os.environ.get("FINGER_MAX_FORCE", "40.0"))
FINGER_STIFFNESS = float(_os.environ.get("FINGER_STIFFNESS", "1000.0"))
BOX_FRICTION = 1.0                   # 외란 없을 때의 상자 표면 마찰

# === 기록 / 솎기 ============================================================
RECORD_HZ = 60
TRAIN_STRIDE = 3
TASK = "pick up the box and place it in the blue tray"

# === 성공 판정 ==============================================================
# (1) 상판에서 SUCCESS_MIN_LIFT 이상 들어올린 적 있음
# (2) 상자 중심이 트레이 안쪽 영역 안
# (3) 트레이 바닥에 놓임 (높이)
# (4) 그리퍼가 놓음
# 네 조건이 SUCCESS_HOLD_STEPS 동안 유지될 때 성공.
SUCCESS_MIN_LIFT = 0.04
SUCCESS_IN_TRAY_MARGIN = 0.005       # 안쪽 벽에서 이만큼 들어와야 '안'
BOX_IN_TRAY_MAX_Z = PLACE_Z + 0.015
SUCCESS_HOLD_STEPS = 30
MAX_EVAL_STEPS = 1500


def box_in_tray_xy(x, y):
    hx = TRAY_INNER[0] / 2.0 - SUCCESS_IN_TRAY_MARGIN
    hy = TRAY_INNER[1] / 2.0 - SUCCESS_IN_TRAY_MARGIN
    return abs(x - TRAY_CENTER_XY[0]) <= hx and abs(y - TRAY_CENTER_XY[1]) <= hy


def too_close_to_tray(x, y):
    dx, dy = x - TRAY_CENTER_XY[0], y - TRAY_CENTER_XY[1]
    return (dx * dx + dy * dy) ** 0.5 < MIN_DISTANCE


def sample_box_xy(rng):
    """수집과 평가가 같은 분포에서 뽑도록 한 곳에 둔다."""
    while True:
        x, y = rng.uniform(*BOX_X_RANGE), rng.uniform(*BOX_Y_RANGE)
        if not too_close_to_tray(x, y):
            return x, y


# 평가 위치: 같은 분포에서 고정 시드로 N개. 모든 외란 조건이 같은 좌표를 쓴다.
EVAL_N = int(_os.environ.get("EVAL_N", "30"))
EVAL_SEED = int(_os.environ.get("EVAL_SEED", "0"))


def eval_positions(n=None, seed=None):
    import random as _random
    rng = _random.Random(EVAL_SEED if seed is None else seed)
    return [tuple(round(v, 4) for v in sample_box_xy(rng)) for _ in range(n or EVAL_N)]


if __name__ == "__main__":
    print(f"작업대 높이   {TABLE_Z}m, 크기 {TABLE_SIZE}")
    print(f"공장 배경     {'ON' if FACTORY_BG else 'OFF'}")
    print(f"상자          {BOX_SIZE*100:.0f}cm, {BOX_MASS}kg, 마찰 {BOX_FRICTION}")
    print(f"상자 영역     x{BOX_X_RANGE} y{BOX_Y_RANGE}")
    print(f"트레이        중심 {TRAY_CENTER_XY}, 안쪽 {TRAY_INNER}, 벽 {TRAY_WALL_H*100:.0f}cm")
    print(f"고정 카메라   pos={OVER_TRANSLATE} rot={OVER_ROTATE} focal={OVER_FOCAL}")
    print(f"천장등        {len(CEILING_LIGHTS)}개 x {CEILING_INTENSITY}")
    print(f"수집          {NUM_EPISODES} 에피소드, {RECORD_HZ}Hz -> stride {TRAIN_STRIDE}")
    print(f"평가 위치     {EVAL_N}개 (seed {EVAL_SEED})")
