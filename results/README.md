# 평가 결과 (2026-10-18)

워크스테이션 `/data/jinju/factory/eval_v2`(평가 결과)와 `/data/jinju/factory/analysis_v2_final`(분석)을 옮겨 온 것이다.

- 정책: ACT(3만 스텝), Diffusion Policy(4만 스텝), 각각 학습 시드 3개 → `act_s1~3`, `dp_s1~3`
- 조건마다 25회, 모든 정책이 같은 상자 위치·같은 외란 무작위값(EVAL_SEED=0)

## `eval_v2_json.tgz` → `eval_v2/<정책>/<조건>.json`

평가 결과 JSON 486개를 묶은 것 (`tar xzf eval_v2_json.tgz`로 풀면 `eval_v2/` 폴더가 생긴다).
압축해 둔 이유: `com1.json` 같은 이름이 Windows 예약어(COM1)라 Windows에서는 파일로 저장되지 않는다.

조건 하나의 평가 결과. 파일 이름은 켜진 외란과 단계 (예: `camera2.json`, `friction3_com3.json`, `clean.json`은 외란 없음).

| 키 | 내용 |
| --- | --- |
| `success_rate`, `n_success`, `n_episodes` | 성공률 |
| `failure_modes` | 실패 유형 집계 (`grasp_no_lift` 들어 올리지 못함, `dropped_in_transit` 옮기다 떨어뜨림, `not_released` 놓지 못함, `missed_tray` 트레이 밖) |
| `levels`, `params` | 외란 단계와 실제 물리값 |
| `episodes` | 에피소드별 상자 위치, 성공 여부, 실패 유형, 파지 시도 횟수, 적용된 외란 값 |
| `blank_camera_frames` | 검은 프레임 수 (0이 아니면 분석에서 제외) |

같은 외란이 낮은 단계에서 0/25였던 정책의 더 센 단계는 평가하지 않고 0%로 기록했다 (`episodes`가 비어 있고 `skipped_after`에 그 낮은 단계가 적혀 있다).

## `analysis/`

| 파일 | 내용 |
| --- | --- |
| `summary.csv` | 모델·외란·단계별 성공률 평균, 표준편차, 시드별 값 |
| `breakdown.csv` | 붕괴 지점 (외란 없음의 50% 미만으로 처음 떨어지는 단계, 없으면 `>5`) |
| `failure_modes.csv` | 실패 유형 집계 |
| `grasp_stats.csv` | 파지 재시도 통계 |
| `fig_curves.png/pdf` | 단일 외란 7종 + 7종 동시 |
| `fig_combos.png/pdf` | 현장 조합 (마찰+무게중심, 야간·조명 노후, 설비 진동, 무겁고 쏠린 제품, 마찰+무게 3배 대조) |
| `fig_vibration.png/pdf` | 무작위 흔들림(윗줄) vs 연속 진동(아랫줄): 단독, 카메라 틀어짐과 함께, 7종 동시 |

분석 다시 만들기 (Linux): `cd results && tar xzf eval_v2_json.tgz && cd .. && python analyze.py --eval-root results/eval_v2 --out results/analysis`
