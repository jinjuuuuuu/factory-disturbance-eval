#!/usr/bin/env python3
# run_experiment.py - YAML 실험 설정 하나로 외란 평가 -> 결과 저장 -> 분석까지 실행
# ---------------------------------------------------------------------------
#   python run_experiment.py --config experiments/quick_demo.yaml            # 실행
#   python run_experiment.py --config experiments/quick_demo.yaml --dry-run  # 계획만 출력
#   python run_experiment.py --list-disturbances                              # 외란 단계표
#
# (conda lerobot 환경의 python으로 실행한다. 시뮬레이션은 Isaac Sim, 정책은 LeRobot
#  프로세스가 따로 돌고, 이 파일은 그 둘을 조건별로 띄우고 결과를 모은다.)
#
# 설정 파일 형식 (experiments/*.yaml 참고):
#   name: 실험 이름 (결과 폴더 이름)
#   output: 결과 상위 폴더                  (기본 /data/jinju/factory/experiments)
#   episodes: 조건당 평가 횟수              (기본 25)
#   eval_seed: 상자 위치·외란 무작위값 시드 (기본 0 - 모든 모델이 같은 조건을 받는다)
#   workers_per_model: 모델당 동시 실행 수   (기본 2)
#   models: {이름: 체크포인트 폴더, ...}     이름이 <모델>_s<시드> 형식이면 분석에서
#                                           시드끼리 묶어 평균 ± 표준편차를 낸다
#   runs:                                  실험 조건 묶음. 각 항목은 셋 중 하나:
#     baseline: {}                                   외란 없음
#     friction: {sweep: friction, levels: [1,2,3]}    한 외란을 여러 단계로
#     combined: {sweep: all, levels: [1,2,3,4,5]}     7종을 같은 단계로 동시에
#     night:    {sweep: [light, flicker]}             조합: 목록의 외란을 같은 단계로 함께 올림
#     control:  {sweep: friction, fixed: "mass=3"}    한 외란만 올리고 다른 외란은 고정
#     custom:   {disturbance: "light=2,com=4"}        조건 하나를 직접 지정 (단계 또는 물리값)
#   analyze: true                          끝나면 analyze.py로 그래프·표 생성
# ---------------------------------------------------------------------------
import os
import sys
import time
import socket
import argparse
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import disturbances as D


def conditions_from_runs(runs):
    """runs 설정 -> [(묶음 이름, DIST 문자열)] (중복 조건은 한 번만)."""
    out, seen = [], set()
    for run_name, spec in (runs or {}).items():
        spec = spec or {}
        if "sweep" in spec:
            # sweep: 외란 하나, 또는 목록(같은 단계로 함께 올리는 조합). fixed: 단계와 무관하게 고정할 외란
            fams = spec["sweep"] if isinstance(spec["sweep"], list) else [spec["sweep"]]
            for fam in fams:
                if fam != "all" and fam not in D.MAIN:
                    raise SystemExit(f"[config] runs.{run_name}: 외란 '{fam}'은 쓸 수 없다. 가능: {D.MAIN + ['all']}")
            fixed = str(spec.get("fixed", "")).strip()
            dists = [",".join([f"{f}={lv}" for f in fams] + ([fixed] if fixed else []))
                     for lv in spec.get("levels", range(1, D.N_LEVELS + 1))]
        elif "disturbance" in spec:
            dists = [str(spec["disturbance"])]
        else:
            dists = [""]
        for d in dists:
            D.parse(d)                                   # 문법 오류는 실행 전에 잡는다
            key = D.label(d)
            if key not in seen:
                seen.add(key)
                out.append((run_name, d))
    return out


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    ap = argparse.ArgumentParser(description="공장 외란 주입 검증 프레임워크 - 실험 실행기")
    ap.add_argument("--config", help="실험 설정 YAML")
    ap.add_argument("--dry-run", action="store_true", help="실행하지 않고 계획만 출력")
    ap.add_argument("--list-disturbances", action="store_true", help="외란 종류와 단계별 값 출력")
    ap.add_argument("--port-base", type=int, default=7100, help="정책 서버 포트 시작 번호")
    args = ap.parse_args()

    if args.list_disturbances:
        subprocess.run([sys.executable, os.path.join(HERE, "disturbances.py")])
        return
    if not args.config:
        ap.error("--config 가 필요하다")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    name = cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    root = os.path.join(cfg.get("output", "/data/jinju/factory/experiments"), name)
    n_ep = int(cfg.get("episodes", 25))
    seed = int(cfg.get("eval_seed", 0))
    nw = int(cfg.get("workers_per_model", 2))
    models = cfg.get("models") or {}
    if not models:
        raise SystemExit("[config] models 가 비어 있다")
    missing = [m for m, path in models.items() if not os.path.isfile(os.path.join(path, "config.json"))]
    for m in missing:
        print(f"[config] models.{m}: 체크포인트가 아직 없다 -> {models[m]}")
    if missing and not args.dry_run:
        raise SystemExit("[config] 체크포인트가 없는 모델이 있어 멈춘다")
    conds = conditions_from_runs(cfg.get("runs"))

    total = len(models) * len(conds) * n_ep
    print(f"[exp] 실험 '{name}' -> {root}")
    print(f"[exp] 모델 {len(models)}개 x 조건 {len(conds)}개 x {n_ep}회 = 에피소드 {total}회 "
          f"(모델당 워커 {nw}, 평가 시드 {seed})")
    for run_name, d in conds:
        print(f"        {run_name:12s} {D.label(d):40s} {D.parse(d)[0] if d else '(외란 없음)'}")
    if args.dry_run:
        return

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config_used.yaml"), "w", encoding="utf-8") as f:      # 재현용 기록
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    list_file = os.path.join(root, "conditions.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(d for _, d in conds) + "\n")

    procs, base = [], args.port_base
    for m, path in models.items():
        while not all(port_free(base + i) for i in range(nw)):
            base += 10
        env = dict(os.environ, EVAL_ROOT=root, PORT_BASE=str(base), EVAL_N=str(n_ep), EVAL_SEED=str(seed))
        log = open(os.path.join(root, f"sweep_{m}.log"), "w")
        print(f"[exp] {m}: 포트 {base}~{base + nw - 1} | 로그 {log.name}")
        procs.append((m, subprocess.Popen(["bash", os.path.join(HERE, "run_sweep.sh"), m, path, str(nw), list_file],
                                          cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)))
        base += max(nw, 10)
        time.sleep(20)

    t0 = time.time()
    while any(p.poll() is None for _, p in procs):
        done = sum(len([f for f in os.listdir(os.path.join(root, m)) if f.endswith(".json")])
                   for m, _ in procs if os.path.isdir(os.path.join(root, m)))
        print(f"[exp] 진행 {done}/{len(models) * len(conds)} 조건 | {(time.time() - t0) / 60:.0f}분", flush=True)
        time.sleep(120)
    print(f"[exp] 평가 끝 ({(time.time() - t0) / 60:.0f}분)")

    if cfg.get("analyze", True):
        subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), "--eval-root", root,
                        "--out", os.path.join(root, "analysis")], cwd=HERE)
        print(f"[exp] 결과: {os.path.join(root, 'analysis')} (fig_curves.png, breakdown.csv, summary.csv, "
              f"failure_modes.csv, grasp_stats.csv)")


if __name__ == "__main__":
    main()
