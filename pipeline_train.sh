#!/usr/bin/env bash
# pipeline_train.sh - 시연 수집이 끝나면 변환하고 ACT·Diffusion Policy를 동시에 학습한다
# ---------------------------------------------------------------------------
#   nohup bash pipeline_train.sh > /data/jinju/factory/pipeline.log 2>&1 &
#
# 학습량 근거 (100 에피소드 x 약 237프레임 = 약 2.37만 프레임, 20Hz):
#   ACT       batch 32 x 30k step = 96만 샘플 ~ 40 에폭 (예전 프로젝트 최고점이 40~55 에폭 근처)
#   Diffusion batch 64 x 40k step = 256만 샘플 ~ 108 에폭 (DP는 원 논문도 수백 에폭)
# 체크포인트는 10k마다 남긴다 - 과적합되면 앞 체크포인트로 평가한다.
# ---------------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/jinju/factory
REPO=jinju/factory_box_v1
ROOT=$D/lerobot_ds/factory_box_v1
say() { echo "[pipe $(date +%m-%d\ %H:%M:%S)] $*"; }

say "수집 완료 대기"
while pgrep -f "[c]ollect_demos.py" > /dev/null; do sleep 60; done
na=$(ls $D/demos_v1a/episode_*.npz 2>/dev/null | wc -l)
nb=$(ls $D/demos_v1b/episode_*.npz 2>/dev/null | wc -l)
say "수집 끝: a=$na b=$nb"
if [ $((na + nb)) -lt 80 ]; then say "!! 시연이 80개 미만이라 멈춘다"; exit 1; fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd "$HERE"
if [ ! -d "$ROOT" ]; then
  say "LeRobot 변환 -> $ROOT"
  HF_LEROBOT_HOME=$D/lerobot_ds python to_lerobot.py --src $D/demos_v1a $D/demos_v1b --repo-id $REPO \
    || { say "!! 변환 실패"; exit 1; }
  # create()가 HF_LEROBOT_HOME/<repo_id>에 만든다 -> 고정 경로로 링크
  [ -d "$D/lerobot_ds/$REPO" ] && ln -sfn "$D/lerobot_ds/$REPO" "$ROOT"
fi

COMMON="--dataset.repo_id=$REPO --dataset.root=$ROOT --policy.push_to_hub=false --wandb.enable=false --save_freq=10000 --num_workers=8 --policy.device=cuda"

say "ACT 학습 시작"
lerobot-train $COMMON --policy.type=act --steps=30000 --batch_size=32 --policy.optimizer_lr=3e-5 \
  --output_dir=$D/act_v1 > $D/train_act.log 2>&1 &
pa=$!
sleep 60
say "Diffusion Policy 학습 시작"
lerobot-train $COMMON --policy.type=diffusion --steps=40000 --batch_size=64 \
  --policy.crop_shape="[216,288]" --output_dir=$D/dp_v1 > $D/train_dp.log 2>&1 &
pd=$!
wait $pa; say "ACT 끝 (exit $?)"
wait $pd; say "DP 끝 (exit $?)"
