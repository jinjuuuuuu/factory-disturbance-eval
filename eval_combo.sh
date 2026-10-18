#!/usr/bin/env bash
# eval_combo.sh - 조건 목록 파일 하나를 ACT·DP 시드 3개씩(모델 6개) 모두에 평가한다
# ---------------------------------------------------------------------------
#   nohup bash eval_combo.sh <조건 목록 파일> > /data/jinju/factory/eval_combo.log 2>&1 &
#   목록 파일: 한 줄에 DIST 하나 (예: friction=3,com=3)
#
# 단일 외란·7종 동시 이후에 추가하는 복합 외란(현장 조합) 평가용. 평가 규칙은
# eval_stage.sh와 같다 (25회, EVAL_SEED=0, ACT 3만·DP 4만 스텝 체크포인트).
# 모델마다 워커 1개 = Isaac Sim 6개 (12개는 GPU 96GB가 차서 OOM으로 멈춘 적이 있다). 포트는 6600부터 모델마다 20씩 띄운다.
# 결과는 eval_v2에 같이 저장된다 (이미 있는 조건은 run_sweep.sh가 건너뛴다).
# ---------------------------------------------------------------------------
set -u
LIST=$(readlink -f "${1:?조건 목록 파일}")
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/jinju/factory
EVAL_ROOT=${EVAL_ROOT:-$D/eval_v2}
NW=${NW:-1}
base=${PORT_BASE:-6600}
say() { echo "[combo $(date +%m-%d\ %H:%M:%S)] $*"; }

cd "$HERE"
say "조건 $(grep -c . "$LIST")개: $(paste -sd' ' "$LIST")"
for m in act dp; do
  case $m in act) step=030000 ;; dp) step=040000 ;; esac
  for s in 1 2 3; do
    ck=$D/${m}_v2_s$s/checkpoints/$step/pretrained_model
    say "평가 시작 ${m}_s$s (포트 $base~, 워커 $NW)"
    EVAL_ROOT=$EVAL_ROOT PORT_BASE=$base EVAL_N=25 bash run_sweep.sh "${m}_s$s" "$ck" "$NW" "$LIST" \
      > "$D/sweep_combo_${m}_s${s}_$(basename "$LIST" .txt).log" 2>&1 &
    base=$((base + 20))
    sleep 40
  done
done
wait
say "끝"
