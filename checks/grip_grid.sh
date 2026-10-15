#!/usr/bin/env bash
# grip_grid.sh - 그리퍼 힘 설정별 정답 제어기 성공률 (clean / mass=3 / com=3)
cd ~/factory_robustness
for cfg in "7.2 400" "20 1000" "40 1000" "40 2000"; do
  set -- $cfg
  out=/data/jinju/factory/grip_${1}_${2}
  FACTORY_BG=0 CHECK=expert CHECK_N=6 CHECK_SPECS="clean;mass=3;com=3;friction=3" FINGER_MAX_FORCE=$1 FINGER_STIFFNESS=$2 \
    CHECK_OUT=$out /data/isaacsim/python.sh -u checks/scene_check.py > $out.log 2>&1
  python3 - "$out" "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1] + "/scene_check.json"))["expert"]
print(f"[grid] force={sys.argv[2]:>4} stiff={sys.argv[3]:>5} | " + " | ".join(
    f"{k} {v['success']}/{v['n']} " + "".join("O" if e["success"] else "".join(c[0] for c, ok in e["cond"].items() if not ok)[:1] for e in v["episodes"])
    for k, v in d.items()), flush=True)
PY
done
echo "[grid] DONE"
