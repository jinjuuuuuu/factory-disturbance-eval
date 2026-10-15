# com_test.py - 무게중심 외란이 물리에 실제로 반영되는지 확인 (Isaac Sim 5.1)
# ---------------------------------------------------------------------------
# 폭 1cm 받침대 위에 5cm 상자를 올린다. 받침대 가장자리는 상자 중심에서 ±0.5cm.
#   무게중심이 받침대 위(|dx| < 0.5cm)  -> 서 있어야 한다
#   무게중심이 받침대 밖(|dx| > 0.5cm)  -> 쏠린 쪽으로 넘어져야 한다
# factory_scene.py 와 같은 방식(RigidPrim.set_coms, world.reset 이후 적용)을 쓴다.
#
# 실행: /data/isaacsim/python.sh -u checks/com_test.py
# ---------------------------------------------------------------------------
import json
import math
import os
import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import RigidPrim

BOX = 0.05
SUPPORT_W = 0.01
SUPPORT_H = 0.10
MASS = 0.6

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.scene.add(FixedCuboid(prim_path="/World/Support", name="support",
                            position=np.array([0.0, 0.0, SUPPORT_H / 2]),
                            scale=np.array([SUPPORT_W, 0.10, SUPPORT_H]), size=1.0))
box_z = SUPPORT_H + BOX / 2 + 0.0005
box = world.scene.add(DynamicCuboid(prim_path="/World/Box", name="box",
                                    position=np.array([0.0, 0.0, box_z]),
                                    size=BOX, mass=MASS))
view = world.scene.add(RigidPrim(prim_paths_expr="/World/Box", name="box_view"))
world.reset()

CASES = [("가운데", 0.0), ("+x 0.3cm (받침대 안)", 0.003), ("+x 1.5cm", 0.015),
         ("-x 1.5cm", -0.015), ("+x 2.0cm", 0.020), ("다시 가운데 (값이 남지 않는지)", 0.0)]


def tilt_deg(q):
    w, x, y, z = [float(v) for v in q]
    # 상자 로컬 z축이 월드 z축과 이루는 각
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


results = []
for name, dx in CASES:
    world.reset()
    box.set_world_pose(position=np.array([0.0, 0.0, box_z]), orientation=np.array([1.0, 0, 0, 0]))
    box.set_linear_velocity(np.zeros(3))
    box.set_angular_velocity(np.zeros(3))
    pos, ori = view.get_coms()
    pos = np.array(pos, dtype=np.float32).reshape(1, 1, 3)
    pos[0, 0, :] = [dx, 0.0, 0.0]
    view.set_coms(positions=pos, orientations=np.array(ori, dtype=np.float32).reshape(1, 1, 4))
    read = np.array(view.get_coms()[0]).reshape(-1)[:3]
    for _ in range(180):                      # 3초
        world.step(render=False)
    p, q = box.get_world_pose()
    t = tilt_deg(q)
    fell = t > 20.0 or float(p[2]) < SUPPORT_H
    side = "+x" if float(p[0]) > 0.005 else ("-x" if float(p[0]) < -0.005 else "-")
    expect_fall = abs(dx) > SUPPORT_W / 2
    ok = (fell == expect_fall) and (not fell or side == ("+x" if dx > 0 else "-x"))
    r = {"case": name, "com_dx_set_m": dx, "com_read_m": [round(float(v), 4) for v in read],
         "tilt_deg": round(t, 1), "final_pos": [round(float(v), 3) for v in p],
         "fell": fell, "fell_toward": side, "expected_fall": expect_fall, "pass": ok}
    results.append(r)
    print(f"[com] {name:28s} 설정 dx={dx*100:+.1f}cm 읽음={r['com_read_m']} "
          f"기울기={t:5.1f}° 넘어짐={fell} 방향={side} -> {'PASS' if ok else 'FAIL'}", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "com_test_result.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[com] {sum(r['pass'] for r in results)}/{len(results)} PASS -> {out}", flush=True)
app.close()
