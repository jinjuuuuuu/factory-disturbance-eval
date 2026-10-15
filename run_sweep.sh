#!/usr/bin/env bash
# run_sweep.sh - 한 모델에 대해 외란 조건 전체를 평가한다 (병렬 워커)
# ---------------------------------------------------------------------------
#   bash run_sweep.sh <모델라벨> <체크포인트 폴더> [워커 수=4] [조건 목록 파일]
#   PORT_BASE=5600   여러 sweep을 동시에 돌릴 때는 반드시 서로 다른 값(예 5600/5700/5800)
#                    - 같으면 다른 sweep의 정책 서버에 붙어 엉뚱한 모델로 평가된다.
#   EVAL_N=50        조건당 평가 에피소드 수
#   예) nohup bash run_sweep.sh act /data/jinju/factory/act_v1/checkpoints/last/pretrained_model 4 \
#         > /data/jinju/factory/sweep_act.log 2>&1 &
#
# 워커마다 정책 서버 1개 + 평가 클라이언트 1개가 짝을 이루고(포트가 다르다), 조건
# 목록을 나눠 맡는다. 결과: /data/jinju/factory/eval/<라벨>/<조건>.json
# 이미 결과 파일이 있는 조건은 건너뛴다(중단 후 다시 돌려도 이어진다).
# ---------------------------------------------------------------------------
set -u
LABEL=$1
CKPT=$2
NW=${3:-4}
LIST=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${EVAL_ROOT:-/data/jinju/factory/eval}/$LABEL
mkdir -p "$OUT"
source ~/miniforge3/etc/profile.d/conda.sh

if [ -n "$LIST" ]; then
  mapfile -t CONDS < "$LIST"
else
  CONDS=("")
  for f in friction light flicker camera shake mass com; do
    for l in 1 2 3 4 5; do CONDS+=("$f=$l"); done
  done
  for l in 1 2 3 4 5; do CONDS+=("all=$l"); done
fi
echo "[sweep] $LABEL | 조건 ${#CONDS[@]}개 | 워커 $NW | $CKPT"

worker() {
  local w=$1 port=$((${PORT_BASE:-5600} + $1)) i
  conda activate lerobot
  cd "$HERE"
  if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
    echo "[w$w] !! 포트 $port 가 이미 사용 중 - PORT_BASE를 바꿔서 다시 실행"; return 1
  fi
  MODEL_PATH="$CKPT" POLICY_PORT=$port python policy_server.py > "$OUT/server_$w.log" 2>&1 &
  local spid=$!
  for _ in $(seq 1 60); do grep -q "listening" "$OUT/server_$w.log" 2>/dev/null && break; sleep 2; done
  for ((i = w; i < ${#CONDS[@]}; i += NW)); do
    local d="${CONDS[$i]}"
    local lab
    lab=$(python -c "import disturbances as D; print(D.label('$d'))")
    if [ -f "$OUT/$lab.json" ]; then echo "[w$w] 건너뜀 $lab"; continue; fi
    echo "[w$w] $(date +%H:%M:%S) 시작 $lab"
    DIST="$d" OUT_JSON="$OUT/$lab.json" MODEL_LABEL="$LABEL" POLICY_PORT=$port EVAL_N=${EVAL_N:-50} \
      /data/isaacsim/python.sh -u eval_client.py > "$OUT/$lab.log" 2>&1
    grep "DONE" "$OUT/$lab.log" | sed "s/^/[w$w] /" || echo "[w$w] !! 실패 $lab (로그 $OUT/$lab.log)"
  done
  kill $spid 2>/dev/null
}

for ((w = 0; w < NW; w++)); do worker $w & sleep 20; done
wait
echo "[sweep] 끝: $(ls "$OUT"/*.json 2>/dev/null | wc -l)개 결과"
