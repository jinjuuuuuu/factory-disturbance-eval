#!/usr/bin/env bash
# train_seeds.sh - 같은 데이터로 학습 시드만 바꿔 ACT·DP를 추가 학습한다 (시드 2, 3)
#   nohup bash train_seeds.sh > /data/jinju/factory/train_seeds.log 2>&1 &
# 시드 1은 pipeline_train.sh의 act_v1 / dp_v1 (lerobot 기본 seed). 설정은 그것과 동일하고
# --seed와 출력 폴더만 다르다.
set -u
D=/data/jinju/factory
REPO=jinju/factory_box_v1
ROOT=$D/lerobot_ds/factory_box_v1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
COMMON="--dataset.repo_id=$REPO --dataset.root=$ROOT --policy.push_to_hub=false --wandb.enable=false --save_freq=10000 --num_workers=6 --policy.device=cuda"
say() { echo "[seeds $(date +%m-%d\ %H:%M:%S)] $*"; }
pids=()
for s in 2 3; do
  say "ACT seed $s 시작"
  lerobot-train $COMMON --seed=$s --policy.type=act --steps=30000 --batch_size=32 --policy.optimizer_lr=3e-5 \
    --output_dir=$D/act_v1_s$s > $D/train_act_s$s.log 2>&1 & pids+=($!)
  sleep 30
  say "DP seed $s 시작"
  lerobot-train $COMMON --seed=$s --policy.type=diffusion --steps=40000 --batch_size=64 \
    --policy.crop_shape="[216,288]" --output_dir=$D/dp_v1_s$s > $D/train_dp_s$s.log 2>&1 & pids+=($!)
  sleep 30
done
for p in "${pids[@]}"; do wait $p; say "pid $p 끝 (exit $?)"; done
say "전부 끝"
