#!/usr/bin/env python3
# disturbances.py - 외란 종류와 강도 단계를 정의하는 단일 출처 (순수 파이썬)
# ---------------------------------------------------------------------------
# 평가는 환경변수 DIST 하나로 외란을 고른다. 여러 개를 쉼표로 이으면 동시에 들어간다.
#
#   DIST=""                          외란 없음 (기준 조건)
#   DIST="friction=3"                마찰만 3단계
#   DIST="light=2,com=4"             조명 2단계 + 무게중심 4단계 동시
#   DIST="all=3"                     모든 외란을 3단계로 동시에
#   DIST="friction_mu=0.33"          단계 대신 물리값을 직접 지정 (세밀한 탐색용)
#
# 단계 0은 '외란 없음'과 같다. 단계 1~5로 갈수록 강해진다.
# 값의 근거와 범위는 각 항목 주석 참고. [잠정] 표시는 장면 점검(scene_check.py)
# 결과를 보고 붕괴 구간이 단계 안에 들어오도록 조정할 값이다.
#
# 에피소드마다 달라지는 무작위 요소(카메라가 틀어지는 방향, 내용물이 쏠리는 방향,
# 깜빡임 패턴)는 (EVAL_SEED, 에피소드 번호)로 시드를 고정한다. 그래서 같은 조건이면
# 모델이 달라도 완전히 같은 외란을 받는다 - 모델 간 비교가 공정해진다.
# ---------------------------------------------------------------------------

import math
import random

N_LEVELS = 5

# 각 외란: 단계 0..5의 물리값
LEVELS = {
    # 상자 표면 마찰계수. 손가락 패드는 1.0, 두 재질 모두 combine 'min' 이라 접촉
    # 마찰은 이 값으로 정해진다. 가이드: mu in [0.1, 1.0].
    "friction": {"key": "friction_mu", "values": [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
                 "unit": "mu", "desc": "접촉 마찰 변동 (그리퍼 패드 마모, 절삭유)"},

    # 천장등 밝기 배율. 가이드: 조도 +-50%. 어두워지는 쪽이 현장에서 문제가 되므로
    # 0.25까지 내린다.
    # 자동 노출을 끈 상태에서 실측: 1.0 -> 고정카메라 평균 밝기 약 166, 0.25 -> 약 107.
    "light": {"key": "light_scale", "values": [1.0, 0.75, 0.55, 0.4, 0.25, 0.12],
              "unit": "x", "desc": "조명 밝기 저하 (주야간, 조명 노후)"},

    # 깜빡임: 정책이 관측할 때마다 이 확률로 천장등이 20%로 떨어진다.
    # Isaac Lab-Arena의 조명 외란은 환경을 만들 때 한 번만 바뀌어 이 상황을 못 만든다.
    "flicker": {"key": "flicker_prob", "values": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                "unit": "p", "desc": "조명 깜빡임 (안정기 노후, 전압 변동)"},

    # 고정 카메라 거치대 틀어짐: 이동 크기[m]와 회전 크기[deg]가 함께 커진다.
    # 가이드: dx,y,z +-1~5cm, dtheta +-1~5 deg. 방향은 에피소드마다 무작위(시드 고정).
    "camera": {"key": "cam_shift", "values": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
               "rot_values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
               "unit": "m/deg", "desc": "카메라 거치대 틀어짐 (에피소드 동안 고정, 위치+각도)"},

    # 상자 질량 배율 (기준 0.2kg). 가이드는 0.5~2.0배. [잠정] 2배까지는 파지에
    # 영향이 없을 수 있어 6배(1.2kg)까지 둔다 - 점검 결과로 조정.
    "mass": {"key": "mass_scale", "values": [1.0, 1.5, 2.0, 3.0, 4.5, 6.0],
             "unit": "x", "desc": "내용물 무게 편차"},

    # 내용물 쏠림 = 상자 무게중심의 수평 이동 크기[m]. 상자 반폭이 2.5cm라 2cm가 상한.
    # 방향은 에피소드마다 무작위(시드 고정). [잠정] 무게중심 효과는 질량에 비례하므로
    # 무게중심 단독 시험은 질량을 3배로 고정한 상태에서 한다(COM_TEST_MASS_SCALE).
    "com": {"key": "com_shift", "values": [0.0, 0.004, 0.008, 0.012, 0.016, 0.020],
            "unit": "m", "desc": "내용물 쏠림 (무게중심 편심)"},

    # 화면 흔들림: 고정 카메라가 정책이 관측할 때마다 다른 위치·각도로 떨린다(설비 진동).
    # 'camera'는 에피소드 동안 고정된 틀어짐, 'shake'는 시시각각 바뀌는 떨림이다 -
    # 조명의 '저하(고정)'와 '깜빡임(시시각각)'과 같은 짝. 값은 떨림의 RMS 크기.
    "shake": {"key": "shake_shift", "values": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03],
              "rot_values": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
              "unit": "m/deg (RMS)", "desc": "화면 흔들림 (설비 진동)"},
}

# 무게중심 단독 시험 때의 질량 배율 (가벼운 상자는 쏠려도 파지에 영향이 없다)
COM_TEST_MASS_SCALE = 3.0

ORDER = ["friction", "light", "flicker", "camera", "mass", "com", "shake"]
# 'all=k'(복합 외란)에 들어가는 외란 = 7종 전부. 현장에서는 진동도 다른 외란과 함께 온다.
ALL_FAMS = list(ORDER)
VISUAL = ["light", "flicker", "camera", "shake"]
PHYSICAL = ["friction", "mass", "com"]

FLICKER_LOW = 0.2       # 깜빡일 때 밝기 배율


def default_params():
    p = {LEVELS[n]["key"]: LEVELS[n]["values"][0] for n in ORDER}
    p["cam_rot_deg"] = 0.0
    p["shake_rot_deg"] = 0.0
    return p


def parse(spec):
    """'friction=3,light=2' -> (물리값 dict, 단계 dict)."""
    params = default_params()
    levels = {n: 0 for n in ORDER}
    spec = (spec or "").strip()
    if not spec:
        return params, levels
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"DIST 항목은 이름=값 이어야 한다: '{item}'")
        name, val = (s.strip() for s in item.split("=", 1))
        if name == "all":
            lv = _level(val, name)
            for n in ALL_FAMS:
                _apply_level(params, levels, n, lv)
        elif name in LEVELS:
            _apply_level(params, levels, name, _level(val, name))
        elif name in params:                       # 물리값 직접 지정
            params[name] = float(val)
            levels[_owner(name)] = -1              # -1 = 단계가 아닌 직접 지정
        else:
            raise ValueError(f"알 수 없는 외란 '{name}'. 가능: {ORDER + ['all'] + list(params)}")
    # 무게중심만 켜고 질량은 안 건드렸으면, 질량을 시험용 배율로 올린다
    if levels["com"] > 0 and levels["mass"] == 0:
        params["mass_scale"] = COM_TEST_MASS_SCALE
    return params, levels


def _level(val, name):
    lv = int(val)
    if not 0 <= lv <= N_LEVELS:
        raise ValueError(f"{name} 단계는 0~{N_LEVELS}: {val}")
    return lv


def _apply_level(params, levels, name, lv):
    d = LEVELS[name]
    params[d["key"]] = d["values"][lv]
    if name == "camera":
        params["cam_rot_deg"] = d["rot_values"][lv]
    if name == "shake":
        params["shake_rot_deg"] = d["rot_values"][lv]
    levels[name] = lv


def _owner(key):
    if key == "cam_rot_deg":
        return "camera"
    if key == "shake_rot_deg":
        return "shake"
    for n in ORDER:
        if LEVELS[n]["key"] == key:
            return n
    return key


def label(spec):
    """결과 파일 이름용 짧은 라벨. 외란 없음은 'clean'."""
    params, levels = parse(spec)
    parts = []
    for n in ORDER:
        if levels[n] > 0:
            parts.append(f"{n}{levels[n]}")
        elif levels[n] == -1:
            parts.append(f"{n}{params[LEVELS[n]['key']]:g}")
    return "_".join(parts) if parts else "clean"


# --- 에피소드별 무작위 요소 (시드 고정) ---------------------------------------

def episode_rng(seed, ep):
    return random.Random(seed * 100003 + ep * 7919 + 17)


def unit_vec3(rng):
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-6:
            return [x / n for x in v]


def episode_draws(params, seed, ep, n_policy_steps=600):
    """에피소드 하나에 쓸 무작위 값을 미리 전부 뽑는다.

    어떤 외란을 켜든 같은 순서로 같은 개수를 뽑는다 - 그래야 'light=2'와
    'light=2,com=3'에서 카메라 방향 같은 다른 요소가 바뀌지 않는다.
    """
    rng = episode_rng(seed, ep)
    cam_dir = unit_vec3(rng)
    cam_rot_axis = unit_vec3(rng)
    com_ang = rng.uniform(0.0, 2.0 * math.pi)
    flicker_u = [rng.random() for _ in range(n_policy_steps)]
    # 흔들림은 나중에 추가된 외란이라 기존 값들 '뒤에서' 뽑는다 -> 기존 외란의 무작위값은
    # 흔들림 추가 전과 똑같다. 축마다 표준편차 = 크기/sqrt(3) 이라 떨림의 RMS가 지정 크기가 된다.
    k = 1.0 / math.sqrt(3.0)
    shake = [([rng.gauss(0, k) for _ in range(3)], [rng.gauss(0, k) for _ in range(3)])
             for _ in range(n_policy_steps)]
    return {
        "cam_offset": [params["cam_shift"] * c for c in cam_dir],
        "cam_rot_axis": cam_rot_axis,
        "cam_rot_deg": params["cam_rot_deg"],
        "com_offset": [params["com_shift"] * math.cos(com_ang),
                       params["com_shift"] * math.sin(com_ang), 0.0],
        "flicker_on": [u < params["flicker_prob"] for u in flicker_u],
        # 관측 k번째의 떨림: (이동[m] 3축, 회전[deg] 3축)
        "shake": [([params["shake_shift"] * x for x in t], [params["shake_rot_deg"] * x for x in r])
                  for t, r in shake],
    }


if __name__ == "__main__":
    print("외란 단계표 (0 = 없음)")
    for n in ORDER:
        d = LEVELS[n]
        vals = " ".join(f"{v:g}" for v in d["values"])
        extra = (" / 회전 " + " ".join(f"{v:g}" for v in d["rot_values"]) + "deg") if "rot_values" in d else ""
        print(f"  {n:9s} [{d['unit']}] {vals}{extra}   - {d['desc']}")
    for s in ["", "friction=3", "light=2,com=4", "all=5", "friction_mu=0.33"]:
        print(f"  DIST={s!r:22s} -> {label(s):35s} {parse(s)[0]}")
