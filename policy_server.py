#!/usr/bin/env python3
# policy_server.py - 학습된 LeRobot 정책(ACT / Diffusion / pi0)을 소켓으로 서비스
# ---------------------------------------------------------------------------
# conda `lerobot`(Python 3.12)에서 실행. Isaac Sim(3.11)과 한 프로세스에 못 올리므로
# 정책은 여기, 시뮬은 eval_client.py에서 돌고 소켓으로 대화한다.
# (~/pickplace/act_policy_server.py와 같은 프로토콜. 정책 종류만 일반화했다.)
#
#   MODEL_PATH=/data/jinju/factory/act_v1/checkpoints/last/pretrained_model \
#   POLICY_PORT=5555 python policy_server.py
#
# 정책 종류는 체크포인트의 config.json("type")에서 읽는다.
# ---------------------------------------------------------------------------
import os
import json
import socket
import struct
import pickle
import numpy as np
import torch

import factory_config as C

MODEL_PATH = os.environ["MODEL_PATH"]
HOST = os.environ.get("POLICY_BIND", "127.0.0.1")
PORT = int(os.environ.get("POLICY_PORT", "5555"))

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
try:
    from lerobot.common.control_utils import predict_action
except ImportError:
    from lerobot.utils.control_utils import predict_action

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    POLICY_TYPE = json.load(f)["type"]
policy = get_policy_class(POLICY_TYPE).from_pretrained(MODEL_PATH)
policy.config.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(policy.config.device)
policy.to(device).eval()
policy.reset()
pre, post = make_pre_post_processors(policy.config, pretrained_path=MODEL_PATH)
print(f"[server] {POLICY_TYPE} 로드: {MODEL_PATH} (device={device})", flush=True)

last = {"wrist": None, "over": None}


def needs_observation():
    """다음 호출에서 정책이 새 관측을 쓰는가 (= 행동 큐가 비었는가).
    ACT는 _action_queue, Diffusion·pi0는 _queues['action'] / _action_queue 를 쓴다.
    구조를 모르면 안전하게 매번 요청한다.
    Diffusion처럼 과거 여러 프레임을 보는 정책(n_obs_steps > 1)은 행동 큐와 상관없이
    매 스텝 관측 큐를 채우므로, 이미지를 건너뛰면 학습 때와 다른(낡은) 프레임을 보게 된다."""
    if getattr(policy.config, "n_obs_steps", 1) > 1:
        return True
    q = getattr(policy, "_action_queue", None)
    if q is None:
        qs = getattr(policy, "_queues", None)
        q = qs.get("action") if isinstance(qs, dict) else None
    if q is None:
        return True
    return len(q) == 0


def decode(payload, msg):
    if msg.get("enc") == "png":
        from PIL import Image
        import io
        return np.array(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
    import zlib
    return np.frombuffer(zlib.decompress(payload), np.uint8).reshape(tuple(msg["shape"])).copy()


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


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(1)
print(f"[server] listening on {HOST}:{PORT}", flush=True)
try:
    while True:
        conn, addr = srv.accept()
        with conn:
            while True:
                msg = recv(conn)
                if msg is None:
                    break
                cmd = msg.get("cmd")
                if cmd == "reset":
                    policy.reset()
                    pre.reset()
                    post.reset()
                    last["wrist"] = last["over"] = None
                    send(conn, {"ok": True, "policy_type": POLICY_TYPE})
                elif cmd == "act":
                    if "wrist" in msg:
                        last["wrist"] = decode(msg["wrist"], msg)
                        last["over"] = decode(msg["over"], msg)
                    elif last["wrist"] is None:
                        send(conn, {"error": "첫 호출에는 이미지가 필요하다"})
                        continue
                    obs = {"observation.images.wrist": last["wrist"],
                           "observation.images.over": last["over"],
                           "observation.state": np.asarray(msg["state"], dtype=np.float32)}
                    with torch.inference_mode():
                        a = predict_action(observation=obs, policy=policy, device=device,
                                           preprocessor=pre, postprocessor=post,
                                           use_amp=policy.config.use_amp, task=C.TASK,
                                           robot_type="franka")
                    send(conn, {"action": a.squeeze(0).cpu().numpy().astype(float).tolist(),
                                "need_obs": needs_observation()})
                elif cmd == "bye":
                    send(conn, {"ok": True})
                    break
                else:
                    send(conn, {"error": f"unknown cmd {cmd}"})
except KeyboardInterrupt:
    pass
finally:
    srv.close()
