#!/usr/bin/env bash
# overnight.sh - 사람 없이 밤사이 진행: 렌더링 생략 판정 -> ACT 평가 재시작 -> ACT 점검 -> DP 평가
# ---------------------------------------------------------------------------
#   nohup bash overnight.sh > /data/jinju/factory/overnight.log 2>&1 &
# 진행 상황: /data/jinju/factory/overnight_status.txt (한 줄 요약이 계속 덧붙는다)
#
# 1) checks/render_ab.sh 결과로 RENDER_LOOKAHEAD를 고른다.
#    같은 에피소드의 성공 여부·스텝 수가 기존(0)과 같고, 정책이 받은 이미지 차이가
#    프레임 평균 1.0(0~255 기준) 미만인 가장 작은 값. 없으면 0(기존 방식).
# 2) 값이 0이 아니면 ACT 평가를 그 값으로 다시 띄운다. 이미 끝난 조건(json)은 건너뛰고,
#    조건 목록은 run_sweep.sh 기본(41개, 흔들림 포함)이다.
# 3) ACT 평가가 끝나면 점검: 시드마다 외란 없음 성공률 >= 80%, 검은 프레임 0,
#    외란 5단계 중 하나 이상에서 성공률이 기준보다 낮아짐. 통과하면 DP 평가 시작,
#    실패하면 멈추고 HOLD를 남긴다 (사람이 보고 결정).
# DP 학습(lerobot-train)은 건드리지 않는다.
# ---------------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
D=/data/jinju/factory
ST=$D/overnight_status.txt
say() { echo "[night $(date +%m-%d\ %H:%M)] $*" | tee -a $ST; }
PY=~/miniforge3/envs/lerobot/bin/python

# 1) 렌더링 생략 판정
say "렌더링 비교 결과 대기"
until grep -q "IMG LA=2" $D/render_ab.out 2>/dev/null; do
  pgrep -f "render_a[b].sh" > /dev/null || break
  sleep 60
done
LA=$($PY - <<'PY'
import json, glob, os
import numpy as np
from PIL import Image
T = "/data/jinju/factory/render_ab"
def eps(la):
    p = f"{T}/la{la}.json"
    if not os.path.exists(p):
        return None
    return [(e["episode"], e["success"], e["steps"]) for e in json.load(open(p))["episodes"]]
base = eps(0)
pick = 0
if base:
    for la in (1, 2):
        e = eps(la)
        if e != base:
            continue
        diffs = []
        for f in sorted(glob.glob(f"{T}/obs_la0/*.png")):
            g = f.replace("obs_la0", f"obs_la{la}")
            if not os.path.exists(g):
                diffs.append(999); continue
            a = np.asarray(Image.open(f), float); b = np.asarray(Image.open(g), float)
            diffs.append(abs(a - b).mean())
        if diffs and max(diffs) < 1.0:
            pick = la
            break
print(pick)
PY
)
say "RENDER_LOOKAHEAD=$LA 로 결정 (0이면 기존 방식 유지)"
grep -a "^LA=\|^IMG" $D/render_ab.out | tee -a $ST

# 2) ACT 평가 재시작 (생략 적용 시)
if [ "$LA" != "0" ]; then
  say "ACT 평가를 RENDER_LOOKAHEAD=$LA 로 재시작 (끝난 조건은 유지)"
  for p in $(pgrep -f "eval_stage.sh|run_sweep.sh|eval_client.py|policy_server.py"); do
    [ "$p" != "$$" ] && kill $p 2>/dev/null
  done
  sleep 20
  RENDER_LOOKAHEAD=$LA MODEL=act STEP=030000 NW=3 bash eval_stage.sh > $D/eval_stage_act.log 2>&1 &
else
  say "ACT 평가는 기존 프로세스 그대로 진행"
fi
export RENDER_LOOKAHEAD=$LA

# ACT 끝 대기: 시드 3개 x 41조건
until [ "$(ls $D/eval_v2/act_s*/*.json 2>/dev/null | wc -l)" -ge 123 ] && ! pgrep -f "eval_stage.s[h]" > /dev/null; do
  sleep 300
  say "ACT 진행 $(ls $D/eval_v2/act_s*/*.json 2>/dev/null | wc -l)/123"
  # 기존 프로세스(36조건)로 진행 중이었다면 끝난 뒤 흔들림 조건을 채우러 다시 띄운다
  if ! pgrep -f "eval_stage.s[h]" > /dev/null && [ "$(ls $D/eval_v2/act_s*/*.json 2>/dev/null | wc -l)" -lt 123 ]; then
    say "남은 ACT 조건(흔들림 등) 평가 시작"
    MODEL=act STEP=030000 NW=3 bash eval_stage.sh > $D/eval_stage_act2.log 2>&1 &
    sleep 60
  fi
done
say "ACT 평가 끝"

# 3) ACT 점검
VERDICT=$($PY - <<'PY'
import json, glob
D = "/data/jinju/factory/eval_v2"
ok, notes = True, []
for s in (1, 2, 3):
    files = glob.glob(f"{D}/act_s{s}/*.json")
    res = {json.load(open(f))["label"]: json.load(open(f)) for f in files}
    clean = res.get("clean")
    if not clean or clean["success_rate"] < 0.8:
        ok = False; notes.append(f"act_s{s} clean={clean and clean['success_rate']}")
    blank = sum(r["blank_camera_frames"] for r in res.values())
    if blank:
        ok = False; notes.append(f"act_s{s} blank={blank}")
    if clean:
        drop = [k for k, r in res.items() if k.endswith("5") and r["success_rate"] < clean["success_rate"] - 0.2]
        if not drop:
            ok = False; notes.append(f"act_s{s} no level-5 drop")
        notes.append(f"act_s{s} clean={clean['success_rate']:.2f} drops@5={len(drop)}")
print(("PASS " if ok else "HOLD ") + "; ".join(notes))
PY
)
say "ACT 점검: $VERDICT"
$PY analyze.py --eval-root $D/eval_v2 --out $D/analysis_v2 >> $ST 2>&1 && say "ACT 분석 그림 -> $D/analysis_v2"

case $VERDICT in
  PASS*)
    say "DP 평가 시작 (DP 학습이 끝나기를 기다렸다가)"
    RENDER_LOOKAHEAD=$LA MODEL=dp STEP=040000 NW=3 bash eval_stage.sh > $D/eval_stage_dp.log 2>&1
    say "DP 평가 끝: $(ls $D/eval_v2/dp_s*/*.json 2>/dev/null | wc -l)/123"
    ;;
  *)
    say "점검 실패 - DP 평가를 시작하지 않고 멈춘다 (HOLD)"
    ;;
esac
say "끝"
