#!/usr/bin/env bash
# eval_stage.sh - 한 모델 종류(act 또는 dp)의 시드 3개를 학습이 끝나는 대로 평가한다
# ---------------------------------------------------------------------------
#   MODEL=act STEP=030000 NW=3 nohup bash eval_stage.sh > /data/jinju/factory/eval_stage_act.log 2>&1 &
#   MODEL=dp  STEP=040000 NW=4 nohup bash eval_stage.sh > /data/jinju/factory/eval_stage_dp.log 2>&1 &
#
# ACT를 먼저 평가해 실험 설계(외란 단계가 붕괴 구간을 덮는지, 기준 성공률, 버그)를 점검한
# 뒤 DP를 평가하기 위해 eval_all.sh를 모델 종류별로 나눈 것. 평가 조건·횟수·시드 규칙은
# eval_all.sh와 같다 (조건 36개 x 25회, EVAL_SEED=0, 마지막 체크포인트 고정).
# ---------------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=/data/jinju/factory
VER=${VER:-v2}
EVAL_ROOT=${EVAL_ROOT:-$D/eval_$VER}
MODEL=${MODEL:?act 또는 dp}
STEP=${STEP:?체크포인트 스텝}
NW=${NW:-3}
case $MODEL in act) base=6000 ;; dp) base=6100 ;; *) base=6200 ;; esac
say() { echo "[$MODEL $(date +%m-%d\ %H:%M:%S)] $*"; }

# 시드 3개의 마지막 체크포인트가 생기고 학습 프로세스가 끝날 때까지 기다린다
for s in 1 2 3; do
  ck=$D/${MODEL}_${VER}_s$s/checkpoints/$STEP/pretrained_model
  until [ -f "$ck/model.safetensors" ] && ! pgrep -f "output_dir=$D/${MODEL}_${VER}_s$s\b" > /dev/null; do sleep 120; done
  say "seed $s 학습 끝 확인: $ck"
done

cd "$HERE"
for s in 1 2 3; do
  ck=$D/${MODEL}_${VER}_s$s/checkpoints/$STEP/pretrained_model
  say "평가 시작 ${MODEL}_s$s (포트 $base~, 워커 $NW)"
  EVAL_ROOT=$EVAL_ROOT PORT_BASE=$base EVAL_N=25 bash run_sweep.sh "${MODEL}_s$s" "$ck" "$NW" \
    > "$D/sweep_${VER}_${MODEL}_s$s.log" 2>&1 &
  base=$((base + 20))
  sleep 40
done
wait
n=$(ls $EVAL_ROOT/${MODEL}_s*/*.json 2>/dev/null | wc -l)
say "평가 끝: 결과 $n개 (기본 조건 41개 x 시드 3 = 123)"
~/miniforge3/envs/lerobot/bin/python analyze.py --eval-root "$EVAL_ROOT" --out "$D/analysis_$VER" \
  && say "분석 완료 -> $D/analysis_$VER"
