# ik_check.py - IK 경유점 제어기의 외란별 성공률 (Isaac Sim 5.1)
#   CHECK_SPECS="clean;mass=3;com=3;friction=3" CHECK_N=6 /data/isaacsim/python.sh -u checks/ik_check.py
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factory_config as C
import disturbances as D

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})
import factory_scene as F
from ik_expert import IKExpert, apply

s = F.build_scene(simulation_app)
ex = IKExpert(s)
pts = C.eval_positions(n=int(os.environ.get("CHECK_N", "6")), seed=99)
specs = [x if x != "clean" else "" for x in os.environ.get("CHECK_SPECS", "clean").split(";")]
out = {}
for spec in specs:
    params, _ = D.parse(spec)
    res = []
    for k, (bx, by) in enumerate(pts):
        F.reset_episode(s, (bx, by))
        F.apply_episode_disturbance(s, params, D.episode_draws(params, 0, k))
        s.franka.gripper.open()
        for _ in range(60):
            s.world.step(render=False)
        F.move_to_start(s)
        ok = ex.reset(F.box_pos(s), F.tray_place_target())
        max_z = float(F.box_pos(s)[2])
        lost_at = None
        while ok and not ex.done:
            cmd, ph = ex.step(np.array(s.franka.get_joint_positions()))
            apply(s, cmd)
            s.world.step(render=False)
            z = float(F.box_pos(s)[2])
            max_z = max(max_z, z)
            if lost_at is None and ph in ("transfer", "lower") and z < C.BOX_Z + 0.01:
                lost_at = ph
        for _ in range(60):
            s.world.step(render=False)
        cond = F.success_conditions(s, max_z)
        res.append({"box": [bx, by], "success": bool(all(cond.values())), "cond": {a: bool(b) for a, b in cond.items()},
                    "ik_fail": ex.failed, "lost_at": lost_at, "final": [round(float(v), 3) for v in F.box_pos(s)]})
    n_ok = sum(r["success"] for r in res)
    out[D.label(spec)] = {"success": n_ok, "n": len(res), "episodes": res}
    marks = "".join("O" if r["success"] else ("K" if r["ik_fail"] else ("L" if r["lost_at"] else "x")) for r in res)
    print(f"IKCHECK {D.label(spec):11s} {n_ok}/{len(res)} {marks} | {F.read_back(s)}", flush=True)
    for r in res:
        if not r["success"]:
            print(f"IKCHECK    fail {r}", flush=True)
path = os.environ.get("CHECK_OUT", "/data/jinju/factory/ik_check.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("IKCHECK DONE", flush=True)
simulation_app.close()
