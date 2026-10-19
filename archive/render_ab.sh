#!/usr/bin/env bash
# render_ab.sh - 렌더링 생략(RENDER_LOOKAHEAD)이 결과를 바꾸지 않는지 같은 에피소드로 비교
#   bash checks/render_ab.sh <체크포인트> [에피소드 목록=0,1,2] [DIST=""]
set -u
CK=$1; EPS=${2:-0,1,2}; DIST=${3:-}
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
T=/data/jinju/factory/render_ab; rm -rf $T; mkdir -p $T
source ~/miniforge3/etc/profile.d/conda.sh; conda activate lerobot
MODEL_PATH=$CK POLICY_PORT=6500 python policy_server.py > $T/server.log 2>&1 &
SP=$!
for _ in $(seq 1 60); do grep -q listening $T/server.log && break; sleep 2; done
for la in 0 1 2; do
  t0=$(date +%s)
  EVAL_ONLY_EP=$EPS DIST="$DIST" RENDER_LOOKAHEAD=$la SAVE_OBS_DIR=$T/obs_la$la OUT_JSON=$T/la$la.json POLICY_PORT=6500 \
    /data/isaacsim/python.sh -u eval_client.py > $T/la$la.log 2>&1
  echo "LA=$la $(( $(date +%s) - t0 ))s"
  grep -a "^\[eval\] ep" $T/la$la.log | cut -c1-120
done
kill $SP
python - <<'PY'
import glob, os, numpy as np
from PIL import Image
T = "/data/jinju/factory/render_ab"
base = sorted(glob.glob(f"{T}/obs_la0/*.png"))
for la in (1, 2):
    diffs = []
    for f in base:
        g = f.replace("obs_la0", f"obs_la{la}")
        if os.path.exists(g):
            a = np.asarray(Image.open(f), float); b = np.asarray(Image.open(g), float)
            diffs.append(abs(a - b).mean())
    print(f"IMG LA={la} vs LA=0: {len(diffs)} frames, mean abs diff per frame = {[round(d, 2) for d in diffs]}")
PY
