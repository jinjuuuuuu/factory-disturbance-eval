#!/usr/bin/env bash
# eval_all.sh - 본 평가: ACT·DP x 학습 시드 3개 x 조건 36개 x 25회
# ---------------------------------------------------------------------------
#   VER=v2 EVAL_ROOT=/data/jinju/factory/eval_v2 ACT_STEP=030000 DP_STEP=040000 \
#     nohup bash eval_all.sh > /data/jinju/factory/eval_all_v2.log 2>&1 &
#
# - 체크포인트는 시드마다 고르지 않고 같은 학습 스텝(마지막)으로 고정한다 (시드별 최고를
#   고르면 결과가 부풀려진다).
# - 모든 모델이 같은 25개 상자 위치(EVAL_SEED=0)와 같은 에피소드별 외란 무작위값을 받는다.
#   -> 모델 간 차이는 학습 시드와 구조에서만 나온다.
# - 결과: $EVAL_ROOT/{act,dp}_s{1,2,3}/<조건>.json  (analyze.py가 읽는 규칙)
# - 학습이 아직 돌고 있으면 끝날 때까지 기다린다.
# ---------------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/jinju/factory
VER=${VER:-v2}
EVAL_ROOT=${EVAL_ROOT:-$D/eval_$VER}
ACT_STEP=${ACT_STEP:-030000}
DP_STEP=${DP_STEP:-040000}
NW=${NW:-2}                     # 모델마다 워커 수 (6모델 x 2 = Isaac Sim 12개)
say() { echo "[all $(date +%m-%d\ %H:%M:%S)] $*"; }

say "학습 종료 대기"
while pgrep -f "lerobot-trai[n]" > /dev/null; do sleep 120; done

declare -A CK
for s in 1 2 3; do
  CK[act_s$s]=$D/act_${VER}_s$s/checkpoints/$ACT_STEP/pretrained_model
  CK[dp_s$s]=$D/dp_${VER}_s$s/checkpoints/$DP_STEP/pretrained_model
done
for k in "${!CK[@]}"; do
  [ -f "${CK[$k]}/config.json" ] || { say "!! 체크포인트 없음: $k ${CK[$k]}"; exit 1; }
done
echo "{\"ACT seed1 ($ACT_STEP)\": \"${CK[act_s1]}\", \"DP seed1 ($DP_STEP)\": \"${CK[dp_s1]}\"}" > $D/models.json

cd "$HERE"
base=6000
for k in act_s1 dp_s1 act_s2 dp_s2 act_s3 dp_s3; do
  say "시작 $k (포트 $base~) ${CK[$k]}"
  EVAL_ROOT=$EVAL_ROOT PORT_BASE=$base EVAL_N=25 bash run_sweep.sh "$k" "${CK[$k]}" "$NW" > "$D/sweep_${VER}_$k.log" 2>&1 &
  base=$((base + 20))
  sleep 40
done
wait
say "본 평가 끝"
~/miniforge3/envs/lerobot/bin/python analyze.py --eval-root "$EVAL_ROOT" --out "$D/analysis_$VER" \
  && say "분석 완료 -> $D/analysis_$VER"
