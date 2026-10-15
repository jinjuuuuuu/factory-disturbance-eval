# disturbance_video.py - 외란이 걸린 고정 카메라 영상을 정답 제어기 동작과 함께 녹화
# ---------------------------------------------------------------------------
#   VIDEO_DISTS="clean;shake=3;shake=5" VIDEO_OUT=/data/jinju/factory/videos \
#   /data/isaacsim/python.sh -u checks/disturbance_video.py
#
# 조건마다 같은 상자 위치에서 IK 정답 제어기가 상자를 옮기는 모습을 정책 관측 주기
# (20Hz)로 고정 카메라에서 찍는다. 조건별 영상과, 조건들을 가로로 붙인 비교 영상
# (compare.mp4 / compare.gif)을 만든다. 시각 외란(흔들림·깜빡임·조명·카메라)이 정책
# 입장에서 어떻게 보이는지 보여주는 용도다.
# ---------------------------------------------------------------------------
import os
import sys
import shutil
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factory_config as C
import disturbances as D

OUT = os.environ.get("VIDEO_OUT", "/data/jinju/factory/videos")
SPECS = [x if x != "clean" else "" for x in os.environ.get("VIDEO_DISTS", "clean;shake=3;shake=5").split(";")]
os.makedirs(OUT, exist_ok=True)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})
from PIL import Image, ImageDraw
import factory_scene as F
from ik_expert import IKExpert, apply

s = F.build_scene(simulation_app)
ex = IKExpert(s)
bx, by = C.eval_positions(n=1, seed=0)[0]
clips = {}
for spec in SPECS:
    params, _ = D.parse(spec)
    draws = D.episode_draws(params, 0, 0)
    F.reset_episode(s, (bx, by))
    F.apply_episode_disturbance(s, params, draws)
    s.franka.gripper.open()
    F.settle(s, 30)
    F.move_to_start(s)
    ex.reset(F.box_pos(s), F.tray_place_target())
    frames, step, k = [], 0, 0
    while not ex.done:
        if step % C.TRAIN_STRIDE == 0:            # 정책이 관측하는 시점마다 (20Hz)
            F.set_lights_for_step(s, params, draws, k)
            F.set_camera_shake_for_step(s, params, draws, k)
            k += 1
            frames.append(F.grab_rgb(s, s.over_cam))
        cmd, _ = ex.step(np.array(s.franka.get_joint_positions()))
        apply(s, cmd)
        s.world.step(render=True)
        step += 1
    name = D.label(spec)
    clips[name] = frames
    d = os.path.join(OUT, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(d, f"{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", os.path.join(d, "%04d.png"),
                    "-pix_fmt", "yuv420p", "-vcodec", "libx264", os.path.join(OUT, f"{name}.mp4")])
    print(f"VIDEO {name}: {len(frames)} frames -> {OUT}/{name}.mp4", flush=True)

# 비교 영상: 조건들을 가로로 붙이고 위에 라벨
names = list(clips)
n = min(len(c) for c in clips.values())
cmp_dir = os.path.join(OUT, "compare")
shutil.rmtree(cmp_dir, ignore_errors=True)
os.makedirs(cmp_dir)
gif = []
for i in range(n):
    row = np.concatenate([clips[nm][i] for nm in names], axis=1)
    canvas = Image.new("RGB", (row.shape[1], row.shape[0] + 22), "white")
    canvas.paste(Image.fromarray(row), (0, 22))
    dr = ImageDraw.Draw(canvas)
    for j, nm in enumerate(names):
        dr.text((j * C.IMG_W + 8, 5), nm, fill="black")
    canvas.save(os.path.join(cmp_dir, f"{i:04d}.png"))
    if i % 2 == 0:
        gif.append(canvas)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "20", "-i", os.path.join(cmp_dir, "%04d.png"),
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", os.path.join(OUT, "compare.mp4")])
gif[0].save(os.path.join(OUT, "compare.gif"), save_all=True, append_images=gif[1:], duration=100, loop=0)
print(f"VIDEO DONE compare.mp4 / compare.gif ({n} frames, {names})", flush=True)
simulation_app.close()
