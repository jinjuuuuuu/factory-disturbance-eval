#!/usr/bin/env bash
# pipeline_v2.sh - v2 배치(가로 컨베이어 + 옆 트레이) 전체 과정을 이어서 돌린다
# ---------------------------------------------------------------------------
#   nohup bash pipeline_v2.sh > /data/jinju/factory/pipeline_v2.log 2>&1 &
#
#   1) 시연 수집 100개 (수집기 2개 x 50, 시드 다름)
#   2) LeRobot 변환 (이미지 저장 - v1의 mp4 변환은 5시간 걸렸다)
#   3) 학습 6개 동시: ACT·DP x 학습 시드 1,2,3 (v1과 같은 설정)
#   4) 본 평가: 6모델 x 36조건 x 25회 (eval_all.sh)
#   5) 분석 (analyze.py)
# 각 단계는 결과가 이미 있으면 건너뛴다 -> 중간에 멈춰도 다시 실행하면 이어진다.
# v1 결과(demos_v1*, act_v1*, dp_v1*, eval/)는 건드리지 않는다.
# ---------------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/jinju/factory
REPO=jinju/factory_box_v2
ROOT=$D/lerobot_ds_v2/$REPO
say() { echo "[v2 $(date +%m-%d\ %H:%M:%S)] $*"; }
cd "$HERE"

# 1) 수집
for x in "a 1234" "b 5678"; do
  set -- $x
  n=$(ls $D/demos_v2$1/episode_*.npz 2>/dev/null | wc -l)
  if [ "$n" -lt 50 ]; then
    say "수집 $1 시작 (현재 $n개)"
    NUM_EPISODES=50 COLLECT_SEED=$2 SAVE_PATH=$D/demos_v2$1 \
      /data/isaacsim/python.sh -u collect_demos.py > $D/collect_v2$1.log 2>&1 &
    sleep 30
  fi
done
wait
na=$(ls $D/demos_v2a/episode_*.npz | wc -l); nb=$(ls $D/demos_v2b/episode_*.npz | wc -l)
say "수집 끝: a=$na b=$nb"
[ $((na + nb)) -ge 90 ] || { say "!! 시연 부족"; exit 1; }

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

# 2) 변환
# meta/info.json은 변환 '시작' 때 생기므로 완료 판단에 못 쓴다 (중단된 변환을 완료로
# 착각해 학습이 30초 만에 전부 죽은 적 있음). 끝까지 갔을 때만 .complete를 남긴다.
if [ ! -f "$ROOT/.complete" ]; then
  rm -rf "$ROOT"
  say "변환 시작"
  HF_LEROBOT_HOME=$D/lerobot_ds_v2 python to_lerobot.py --src $D/demos_v2a $D/demos_v2b --repo-id $REPO \
    > $D/convert_v2.log 2>&1 || { say "!! 변환 실패"; tail -20 $D/convert_v2.log; exit 1; }
  touch "$ROOT/.complete"
  say "변환 끝: $(tail -1 $D/convert_v2.log)"
fi

# 3) 학습 6개
COMMON="--dataset.repo_id=$REPO --dataset.root=$ROOT --policy.push_to_hub=false --wandb.enable=false --save_freq=10000 --num_workers=4 --policy.device=cuda"
for s in 1 2 3; do
  if [ ! -f $D/act_v2_s$s/checkpoints/030000/pretrained_model/config.json ]; then
    rm -rf $D/act_v2_s$s
    say "ACT seed $s 학습"
    lerobot-train $COMMON --seed=$s --policy.type=act --steps=30000 --batch_size=32 --policy.optimizer_lr=3e-5 \
      --output_dir=$D/act_v2_s$s > $D/train_act_v2_s$s.log 2>&1 &
    sleep 30
  fi
  if [ ! -f $D/dp_v2_s$s/checkpoints/040000/pretrained_model/config.json ]; then
    rm -rf $D/dp_v2_s$s
    say "DP seed $s 학습"
    lerobot-train $COMMON --seed=$s --policy.type=diffusion --steps=40000 --batch_size=64 \
      --policy.crop_shape="[216,288]" --output_dir=$D/dp_v2_s$s > $D/train_dp_v2_s$s.log 2>&1 &
    sleep 30
  fi
done
wait
say "학습 끝"

# 4) 본 평가 + 5) 분석
conda deactivate
VER=v2 EVAL_ROOT=$D/eval_v2 ACT_STEP=030000 DP_STEP=040000 bash eval_all.sh
say "전체 끝"
