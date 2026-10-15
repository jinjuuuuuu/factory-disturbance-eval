#!/usr/bin/env python3
# analyze.py - 일괄 평가 결과를 모아 논문용 표·그림을 만든다
# ---------------------------------------------------------------------------
#   python analyze.py                   # /data/jinju/factory/eval/<모델>_s<시드>/*.json
#   python analyze.py --out /data/jinju/factory/analysis
#
# 결과 폴더 이름 규칙: <모델>_s<시드> (예: act_s1, dp_s3). 같은 모델의 시드끼리 묶어
# 평균 ± 표준편차를 낸다.
#
# 붕괴 지점(breakdown point) 정의 - 논문 Ⅲ장에 그대로 쓴다:
#   외란 단계 1..5 중, 시드 평균 성공률이 '외란 없음' 시드 평균의 50% 미만으로 처음
#   떨어지는 단계. 5단계까지 떨어지지 않으면 '>5'.
#
# 출력: summary.csv (모델·외란·단계별 평균/표준편차), breakdown.csv, failure_modes.csv,
#       fig_curves.png (외란별 성공률 곡선), fig_curves.pdf
# ---------------------------------------------------------------------------
import os
import re
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

import disturbances as DI

FAMS = DI.ORDER + ["all"]
NAMES_KO = {"friction": "마찰", "light": "조명 밝기", "flicker": "조명 깜빡임", "camera": "카메라 틀어짐",
            "shake": "화면 흔들림", "mass": "질량", "com": "무게중심 쏠림", "all": "7종 동시"}
NAMES_EN = {"friction": "Friction", "light": "Light dimming", "flicker": "Light flicker",
            "camera": "Camera offset", "shake": "Camera shake", "mass": "Mass", "com": "CoM shift",
            "all": "All seven"}


def family_level(levels):
    on = {k: v for k, v in levels.items() if v > 0}
    if not on:
        return "clean", 0
    if set(on) == set(DI.ALL_FAMS) and len(set(on.values())) == 1:
        return "all", next(iter(on.values()))
    if len(on) == 1:
        k = next(iter(on))
        return k, on[k]
    return None, None


def load(eval_root):
    """{(model, fam, level): {seed: rate}} 와 실패 유형 집계."""
    rates = defaultdict(dict)
    modes = defaultdict(lambda: defaultdict(int))
    grasp = defaultdict(lambda: {"attempts": [], "retried": 0, "recovered": 0, "n": 0})
    for d in sorted(glob.glob(os.path.join(eval_root, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in ("analysis",) or os.path.basename(d).startswith("ckpt_"):
            continue
        # 폴더 이름 <모델>_s<시드> 이면 시드끼리 묶고, 시드 표시가 없으면 시드 1개로 본다
        m = re.match(r"(.+?)(?:_s(\d+))?$", os.path.basename(d))
        model, seed = m.group(1), int(m.group(2) or 1)
        for f in glob.glob(os.path.join(d, "*.json")):
            s = json.load(open(f))
            if s.get("blank_camera_frames"):
                print(f"[!] 검은 프레임 있음 - 제외: {f}")
                continue
            fam, lv = family_level(s["levels"])
            if fam is None:
                continue
            rates[(model, fam, lv)][seed] = s["success_rate"]
            for k, v in s["failure_modes"].items():
                modes[(model, fam, lv)][k] += v
            g = grasp[(model, fam, lv)]
            for e in s["episodes"]:
                if "n_grasp_attempts" in e:
                    g["attempts"].append(e["n_grasp_attempts"])
                    g["retried"] += e["n_grasp_attempts"] > 1
                    g["recovered"] += bool(e.get("recovered"))
                    g["n"] += 1
    return rates, modes, grasp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="/data/jinju/factory/eval")
    ap.add_argument("--out", default="/data/jinju/factory/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rates, modes, grasp = load(args.eval_root)
    models = sorted({k[0] for k in rates})
    if not models:
        raise SystemExit("결과 없음")

    # 요약표
    rows = []
    for model in models:
        for fam in FAMS:
            for lv in range(0, DI.N_LEVELS + 1):
                key = (model, "clean", 0) if lv == 0 else (model, fam, lv)
                r = rates.get(key, {})
                if not r:
                    continue
                v = np.array(list(r.values())) * 100
                rows.append((model, fam, lv, len(v), v.mean(), v.std(ddof=0), " ".join(f"{x:.0f}" for x in v)))
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("model,disturbance,level,n_seeds,mean_pct,std_pct,per_seed\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.1f},{r[5]:.1f},{r[6]}\n")

    # 붕괴 지점
    bp = {}
    with open(os.path.join(args.out, "breakdown.csv"), "w") as f:
        f.write("model,disturbance,clean_mean_pct,breakdown_level,breakdown_value\n")
        for model in models:
            base = np.mean(list(rates.get((model, "clean", 0), {0: np.nan}).values())) * 100
            for fam in FAMS:
                lvl = ">5"
                for lv in range(1, DI.N_LEVELS + 1):
                    r = rates.get((model, fam, lv))
                    if r and np.mean(list(r.values())) * 100 < 0.5 * base:
                        lvl = lv
                        break
                val = ""
                if isinstance(lvl, int) and fam in DI.LEVELS:
                    val = DI.LEVELS[fam]["values"][lvl]
                bp[(model, fam)] = lvl
                f.write(f"{model},{fam},{base:.1f},{lvl},{val}\n")

    # 실패 유형
    with open(os.path.join(args.out, "failure_modes.csv"), "w") as f:
        allm = sorted({k for v in modes.values() for k in v})
        f.write("model,disturbance,level," + ",".join(allm) + "\n")
        for (model, fam, lv), c in sorted(modes.items()):
            f.write(f"{model},{fam},{lv}," + ",".join(str(c.get(k, 0)) for k in allm) + "\n")

    # 재시도: 파지 시도 횟수 평균, 재시도한 에피소드 비율, 떨어뜨린 뒤 회복해 성공한 수
    with open(os.path.join(args.out, "grasp_stats.csv"), "w") as f:
        f.write("model,disturbance,level,episodes,mean_grasp_attempts,retried_pct,recovered\n")
        for (model, fam, lv), g in sorted(grasp.items()):
            if g["n"]:
                f.write(f"{model},{fam},{lv},{g['n']},{np.mean(g['attempts']):.2f},"
                        f"{100 * g['retried'] / g['n']:.1f},{g['recovered']}\n")

    # 그림: 외란 7개 패널, 모델별 평균 ± 표준편차
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 시각 외란 윗줄, 물리 외란 + 동시 외란 아랫줄 (8칸 = 시각 4 + 물리 3 + 동시 1)
    FAMS = DI.VISUAL + DI.PHYSICAL + ["all"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharey=True)
    colors = {"act": "#2b6cb0", "dp": "#c05621", "pi0": "#2f855a"}
    for ax, fam in zip(axes.flat, FAMS):
        for model in models:
            xs, ms, ss = [], [], []
            for lv in range(0, DI.N_LEVELS + 1):
                key = (model, "clean", 0) if lv == 0 else (model, fam, lv)
                r = rates.get(key)
                if r:
                    v = np.array(list(r.values())) * 100
                    xs.append(lv), ms.append(v.mean()), ss.append(v.std())
            if xs:
                c = colors.get(model, None)
                ax.errorbar(xs, ms, yerr=ss, marker="o", ms=4, capsize=3, lw=1.6, color=c,
                            label=model.upper())
                b = bp.get((model, fam))
                if isinstance(b, int):
                    ax.axvline(b, color=c, ls=":", lw=1)
        ax.set_title(NAMES_EN[fam], fontsize=10)
        ax.set_xticks(range(0, DI.N_LEVELS + 1))
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9, loc="lower left")
    axes[0, 0].set_ylabel("Visual — success (%)")
    axes[1, 0].set_ylabel("Physical — success (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("Disturbance level")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_curves.{ext}"), dpi=200)
    print(f"[analyze] 모델 {models} | 결과 -> {args.out}")
    for (model, fam), lvl in sorted(bp.items()):
        print(f"  {model:5s} {fam:9s} 붕괴 단계 {lvl}")


if __name__ == "__main__":
    main()
