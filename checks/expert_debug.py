# expert_debug.py - 정답 제어기가 상자를 놓치는 원인 추적 (Isaac Sim 5.1)
#   DIST="mass=2" /data/isaacsim/python.sh -u checks/expert_debug.py
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factory_config as C
import disturbances as D

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})
from pxr import UsdPhysics, Usd
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
import factory_scene as F

s = F.build_scene(simulation_app)

# 손가락 관절 드라이브 설정
for j in ("panda_finger_joint1", "panda_finger_joint2"):
    for p in Usd.PrimRange(s.stage.GetPrimAtPath("/World/Franka")):
        if p.GetName() == j:
            d = UsdPhysics.DriveAPI.Get(p, "linear")
            print(f"[dbg] {j}: stiffness={d.GetStiffnessAttr().Get()} damping={d.GetDampingAttr().Get()} "
                  f"maxForce={d.GetMaxForceAttr().Get()}", flush=True)
try:
    print(f"[dbg] gripper closed pos={s.franka.gripper.joint_closed_positions} "
          f"opened={s.franka.gripper.joint_opened_positions} deltas={getattr(s.franka.gripper, '_action_deltas', None)}",
          flush=True)
except Exception as e:
    print("[dbg] gripper info err", e)

params, _ = D.parse(os.environ.get("DIST", "mass=2"))
for bx, by in [(0.42, 0.10), (0.45, 0.25)]:
    F.reset_episode(s, (bx, by))
    F.apply_episode_disturbance(s, params, D.episode_draws(params, 0, 0))
    print(f"[dbg] === box ({bx},{by}) applied {F.read_back(s)}", flush=True)
    s.franka.gripper.open()
    for _ in range(60):
        s.world.step(render=False)
    F.move_to_start(s)
    ctrl = PickPlaceController(name="pp", gripper=s.franka.gripper, robot_articulation=s.franka,
                               end_effector_initial_height=C.TABLE_Z + 0.3)
    pick, place = F.box_pos(s), F.tray_place_target()
    for step in range(2000):
        q = np.array(s.franka.get_joint_positions())
        a = ctrl.forward(picking_position=pick, placing_position=place, current_joint_positions=q)
        s.franka.apply_action(a)
        s.world.step(render=False)
        if step % 25 == 0 or ctrl.is_done():
            b = F.box_pos(s)
            ee = F.get_ee_position(s)
            gcmd = None if a.joint_positions is None else np.round(np.array(a.joint_positions, dtype=float)[-2:], 4)
            from pxr import UsdGeom
            tips = [UsdGeom.Xformable(s.stage.GetPrimAtPath(f"/World/Franka/{n}"))
                    .ComputeLocalToWorldTransform(0).ExtractTranslation()
                    for n in ("panda_leftfinger", "panda_rightfinger")]
            mid = [(tips[0][i] + tips[1][i]) / 2 for i in range(3)]
            print(f"[dbg]   finger mid=({mid[0]:.3f},{mid[1]:.3f},{mid[2]:.3f}) drives={s.finger_drives}", flush=True)
            print(f"[dbg] step {step:4d} event={ctrl.get_current_event()} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                  f"box=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) fingers={np.round(q[7:9], 4)} cmd={gcmd}", flush=True)
        if ctrl.is_done():
            break
    print(f"[dbg] cond {F.success_conditions(s, 1.0)}", flush=True)
simulation_app.close()
