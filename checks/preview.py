# preview.py - 외란을 적용한 장면을 카메라 두 대로 찍어 저장 (Isaac Sim 5.1)
#   DIST="light_scale=0.4,cam_shift=0.03,cam_rot_deg=3" PREVIEW_OUT=/tmp/pv \
#   /data/isaacsim/python.sh -u checks/preview.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factory_config as C
import disturbances as D

OUT = os.environ.get("PREVIEW_OUT", "/tmp/factory_preview")
os.makedirs(OUT, exist_ok=True)
params, _ = D.parse(os.environ.get("DIST", ""))
seed = int(os.environ.get("EVAL_SEED", "0"))

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})
from PIL import Image
import factory_scene as F

s = F.build_scene(simulation_app)
x, y = C.eval_positions(n=1, seed=seed)[0]
draws = D.episode_draws(params, seed, 0)
F.reset_episode(s, (x, y))
F.apply_episode_disturbance(s, params, draws)
F.move_to_start(s)
if params["flicker_prob"] > 0:
    F.set_light_scale(s, params["light_scale"] * D.FLICKER_LOW)   # 깜빡여 어두워진 순간
F.set_camera_shake_for_step(s, params, draws, 0)                  # 흔들린 한 순간
F.settle(s, 10)
Image.fromarray(F.grab_rgb(s, s.over_cam)).save(os.path.join(OUT, "over.png"))
Image.fromarray(F.grab_rgb(s, s.wrist_cam)).save(os.path.join(OUT, "wrist.png"))
print(f"PREVIEW_DONE {F.read_back(s)}", flush=True)
simulation_app.close()
