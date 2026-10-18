#!/usr/bin/env bash
# record_fail_gifs.sh - 평가에서 실패한 에피소드를 다시 돌려 고정 카메라 GIF로 저장한다
# ---------------------------------------------------------------------------
#   nohup bash record_fail_gifs.sh <목록 파일> > /data/jinju/factory/fail_gifs/run.log 2>&1 &
#   목록 파일: 한 줄에 "이름|DIST|모델(act_s1 등)|에피소드 번호"
#
# 평가와 같은 상자 위치·외란 무작위값(EVAL_SEED=0, 에피소드 번호)으로 그 에피소드 하나만
# 다시 돌린다 (eval_client.py의 EVAL_ONLY_EP + RECORD_GIF). 결과 JSON도 따로 남기므로
# 다시 돌렸을 때도 실패했는지 확인할 수 있다 (DP는 샘플링이라 결과가 달라질 수 있다).
# 평가 결과 폴더(eval_v2)는 건드리지 않는다. 포트는 6900을 쓴다.
# ---------------------------------------------------------------------------
set -u
LIST=$(readlink -f "${1:?목록 파일}")
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/jinju/factory
OUT=${GIF_OUT:-$D/fail_gifs}
PORT=${PORT:-6900}
mkdir -p "$OUT"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd "$HERE"

while IFS='|' read -r name dist model ep; do
  [ -z "$name" ] && continue
  case $model in act_*) step=030000 ;; dp_*) step=040000 ;; esac
  ck=$D/${model%_s*}_v2_s${model##*_s}/checkpoints/$step/pretrained_model
  echo "[gif $(date +%H:%M:%S)] $name | DIST=$dist | $model ep$ep"
  MODEL_PATH="$ck" POLICY_PORT=$PORT python policy_server.py > "$OUT/server.log" 2>&1 < /dev/null &
  spid=$!
  for _ in $(seq 1 60); do grep -q "listening" "$OUT/server.log" 2>/dev/null && break; sleep 2; done
  DIST="$dist" EVAL_ONLY_EP=$ep RECORD_GIF="$OUT/$name.gif" OUT_JSON="$OUT/$name.json" \
    MODEL_LABEL="$model" POLICY_PORT=$PORT EVAL_N=25 \
    /data/isaacsim/python.sh -u eval_client.py > "$OUT/$name.log" 2>&1 < /dev/null
  grep -a "DONE" "$OUT/$name.log" || echo "[gif] !! 실패 $name (로그 $OUT/$name.log)"
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
done < "$LIST"
echo "[gif $(date +%H:%M:%S)] 끝"
