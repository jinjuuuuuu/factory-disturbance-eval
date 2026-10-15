#!/usr/bin/env python3
# to_lerobot.py - 시연 npz를 솎아서 LeRobotDataset으로 변환 (conda lerobot)
# ---------------------------------------------------------------------------
# ~/pickplace/subsample_dataset.py 의 변환 부분을 공장 데이터에 맞춘 것.
# 60Hz 기록을 TRAIN_STRIDE(3)마다 솎아 20Hz로 만든다. 평가의 ACTION_REPEAT도 같은
# 값을 쓰므로 둘이 어긋날 수 없다(어긋나면 로봇이 3배 빠르게 움직이고 조용히 실패).
#
#   python to_lerobot.py --src /data/jinju/factory/demos_v1 --repo-id jinju/factory_box_v1
# 데이터셋은 HF_LEROBOT_HOME(기본 ~/.cache/huggingface/lerobot)/<repo-id> 에 생긴다.
# ---------------------------------------------------------------------------
import os
import glob
import argparse
import numpy as np

import factory_config as C

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="episode_*.npz 폴더 (여러 개 가능)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--stride", type=int, default=C.TRAIN_STRIDE)
    ap.add_argument("--limit", type=int, default=0)
    # 기본은 mp4. 이미지(PNG) 저장은 학습 때 데이터 읽기가 병목이 돼 GPU 0%, 초당 0.4스텝까지
    # 떨어졌다(mp4는 초당 11스텝). v1 변환이 5시간 걸린 원인은 mp4가 아니라 npz 반복 해제였다.
    ap.add_argument("--images", dest="videos", action="store_false", help="PNG로 저장 (학습 시 느림)")
    ap.add_argument("--writer-threads", type=int, default=16)
    args = ap.parse_args()

    files = sorted(f for src in args.src for f in glob.glob(os.path.join(src, "episode_*.npz")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"에피소드 없음: {args.src}")
    d0 = np.load(files[0])
    H, W = d0["images_wrist"].shape[1:3]
    fps = round(C.RECORD_HZ / args.stride)
    kind = "video" if args.videos else "image"
    feats = {
        "observation.images.wrist": {"dtype": kind, "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        "observation.images.over": {"dtype": kind, "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (9,), "names": None},
        "action": {"dtype": "float32", "shape": (9,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=feats,
                               robot_type="franka", use_videos=args.videos,
                               image_writer_threads=args.writer_threads)
    total = 0
    for i, f in enumerate(files):
        # [!] npz 안의 배열은 d["키"]로 꺼낼 때마다 전체를 다시 압축 해제한다. 프레임마다
        #   d["images_wrist"][t]로 꺼내면 710장짜리 배열을 매번 풀어서, 에피소드 하나에 약
        #   3분이 걸렸다(cProfile: 340초 중 270초가 zlib 해제). 에피소드마다 한 번만 꺼낸다.
        with np.load(f) as z:
            d = {k: z[k] for k in ("obs", "actions", "images_wrist", "images_over")}
        T = min(len(d["obs"]), len(d["actions"]), len(d["images_wrist"]), len(d["images_over"]))
        idx = list(range(0, T, args.stride))
        if idx[-1] != T - 1:
            idx.append(T - 1)              # 마지막(놓고 복귀한) 상태는 항상 남긴다
        for t in idx:
            ds.add_frame({"observation.images.wrist": d["images_wrist"][t],
                          "observation.images.over": d["images_over"][t],
                          "observation.state": d["obs"][t][:9].astype(np.float32),
                          "action": d["actions"][t][:9].astype(np.float32),
                          "task": C.TASK})
        ds.save_episode()
        total += len(idx)
        if i % 10 == 0 or i == len(files) - 1:
            print(f"[convert] {i + 1}/{len(files)} {os.path.basename(f)} {T} -> {len(idx)} (누적 {total})", flush=True)
    if hasattr(ds, "finalize"):
        ds.finalize()
    print(f"[convert] 완료: {len(files)} 에피소드, {total} 프레임, {fps}Hz -> {ds.root}", flush=True)


if __name__ == "__main__":
    main()
