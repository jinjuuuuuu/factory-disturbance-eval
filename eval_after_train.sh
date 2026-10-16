#!/usr/bin/env bash
# eval_after_train.sh - DP 학습이 끝난 뒤 ACT(남은 조건)·DP 평가를 함께 돌린다
# ---------------------------------------------------------------------------
#   nohup bash eval_after_train.sh > /data/jinju/factory/eval_after_train.log 2>&1 &
#
# 10/15 밤: DP 학습 3개 + ACT 평가 9개가 GPU 메모리 96GB를 채워 둘 다 수십 배 느려졌다
# (에피소드당 20~50분). 그래서 학습과 평가를 겹치지 않게 하고, 평가 동시 실행 수도
# 모델당 2개(총 Isaac Sim 12개)로 줄인다. 이미 끝난 조건(json)은 건너뛴다.
# ---------------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
D=/data/jinju/factory
ST=$D/overnight_status.txt
say() { echo "[eval2 $(date +%m-%d\ %H:%M)] $*" | tee -a $ST; }

say "DP 학습 종료 대기"
while pgrep -f "lerobot-trai[n]" > /dev/null; do sleep 120; done
say "DP 학습 끝. 평가 시작 (ACT·DP 각 모델 워커 2개)"
NW=${NW:-2}
MODEL=act STEP=030000 NW=$NW bash eval_stage.sh > $D/eval_stage_act3.log 2>&1 &
sleep 120
MODEL=dp  STEP=040000 NW=$NW bash eval_stage.sh > $D/eval_stage_dp.log 2>&1 &
while pgrep -f "eval_stage.s[h]" > /dev/null; do
  sleep 1800
  say "진행 ACT $(ls $D/eval_v2/act_s*/*.json 2>/dev/null | wc -l)/123, DP $(ls $D/eval_v2/dp_s*/*.json 2>/dev/null | wc -l)/123"
done
~/miniforge3/envs/lerobot/bin/python analyze.py --eval-root $D/eval_v2 --out $D/analysis_v2 >> $ST 2>&1
say "평가·분석 끝 -> $D/analysis_v2"
