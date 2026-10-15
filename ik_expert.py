# ik_expert.py - 경유점 + 역기구학(IK) 기반 정답 제어기 (Isaac Sim 5.1)
# ---------------------------------------------------------------------------
# PickPlaceController(RMPflow)는 상자를 세게 쥐면 운반 중 팔이 흔들려 트레이 밖에
# 떨어뜨렸다. 여기서는 경유점마다 IK로 관절 목표를 구하고 관절 공간에서 부드럽게
# 보간한다. 궤적이 매번 같은 모양이라 모방학습용 시연으로도 낫다.
#
# 사용:
#   ex = IKExpert(s)                  # factory_scene.Scene
#   ex.reset(pick_xyz, place_xyz)
#   while not ex.done: s.franka.apply_action(ex.step(q)); s.world.step()
# ---------------------------------------------------------------------------
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver, LulaKinematicsSolver, interface_config_loader)

import factory_config as C

DOWN = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))   # 손가락이 아래를 향함
APPROACH_H = 0.12        # 상자 위 접근 높이
GRASP_DZ = 0.0           # 손가락 중심 = 상자 중심 높이에서 쥔다
TCP_OFFSET = 0.1034      # panda_hand -> 손가락 중심 (Franka 표준)

# (단계 이름, 프레임 수, 그리퍼) - 60Hz 기준
PHASES = [
    ("approach", 120, "open"),
    ("descend", 70, "open"),
    ("close", 40, "close"),
    ("lift", 70, "close"),
    ("transfer", 150, "close"),
    ("lower", 70, "close"),
    ("release", 30, "open"),
    ("retreat", 60, "open"),
]


class IKExpert:
    def __init__(self, s):
        self.s = s
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.solver = LulaKinematicsSolver(**cfg)
        base_p, base_q = s.franka.get_world_pose()
        self.solver.set_robot_base_pose(np.array(base_p), np.array(base_q))
        self.aks = ArticulationKinematicsSolver(s.franka, self.solver, "right_gripper")
        self.done = True

    def _ik(self, xyz, seed=None):
        act, ok = self.aks.compute_inverse_kinematics(np.array(xyz, dtype=float), DOWN)
        if not ok:
            return None
        return np.array(act.joint_positions, dtype=float)[:7]

    def reset(self, pick, place):
        pick = np.array(pick, dtype=float)
        place = np.array(place, dtype=float)
        up = np.array([0, 0, APPROACH_H])
        g = pick + np.array([0, 0, GRASP_DZ])
        targets = {
            "approach": pick + up, "descend": g, "close": g, "lift": pick + up,
            "transfer": place + up, "lower": place + np.array([0, 0, 0.005]),
            "release": place + np.array([0, 0, 0.005]), "retreat": place + up,
        }
        self.q_targets = {}
        for name, xyz in targets.items():
            q = self._ik(xyz)
            if q is None:
                self.done = True
                self.failed = f"IK 실패: {name} {np.round(xyz, 3)}"
                return False
            self.q_targets[name] = q
        self.failed = None
        self.phase = 0
        self.t = 0
        self.q_start = None
        self.done = False
        return True

    def step(self, q_now):
        """현재 관절 -> 이번 스텝 관절 명령(9). 단계 끝에서 다음 단계로 넘어간다."""
        name, n, grip = PHASES[self.phase]
        if self.q_start is None:
            self.q_start = np.array(q_now[:7], dtype=float)
        a = min(1.0, (self.t + 1) / n)
        a = 0.5 - 0.5 * np.cos(np.pi * a)                  # 부드러운 가감속
        q7 = (1 - a) * self.q_start + a * self.q_targets[name]
        g = C.GRIPPER_CLOSED if grip == "close" else C.GRIPPER_OPEN
        self.t += 1
        if self.t >= n:
            self.phase += 1
            self.t = 0
            self.q_start = None
            if self.phase >= len(PHASES):
                self.done = True
        return np.concatenate([q7, [g, g]]), name


def apply(s, cmd9):
    s.franka.apply_action(ArticulationAction(joint_positions=np.asarray(cmd9, dtype=float)))
