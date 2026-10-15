# 공장 외란 주입 검증 프레임워크

Isaac Sim 공장 장면에서 학습된 로봇 정책(ACT, Diffusion Policy 등 LeRobot 정책)에 **외란 7종을 강도 5단계로 주입**하고, 단계별 성공률 곡선과 **붕괴 지점**(성공률이 기준의 50% 아래로 처음 떨어지는 단계)을 자동으로 측정한다. 정책은 재학습하지 않는다.

외란은 정책이 **보는 것**을 바꾸는 시각 외란 4종과, 상자가 **움직이는 방식**을 바꾸는 물리 외란 3종으로 나뉜다. 단계 0은 외란 없음이다.

### 시각 외란 — 카메라 영상이 달라진다

| 외란 | 현장 요인 | 단계 0 → 5 | 비고 |
| --- | --- | --- | --- |
| 조명 밝기 저하 `light` | 주야간 변화, 조명 노후 | 밝기 100, 75, 55, 40, 25, 12% | |
| 조명 깜빡임 `flicker` | 안정기 노후, 전압 변동 | 관측마다 깜빡일 확률 0, 5, 10, 20, 30, 50% (깜빡일 때 밝기 20%) | 기존 연구에 없음 |
| 카메라 위치·각도 틀어짐 `camera` | 거치대 충격, 보정 어긋남 | 에피소드 동안 고정된 이동 0~5cm + 회전 0~5° (1cm·1° 간격) | |
| 화면 흔들림 `shake` | 프레스·컨베이어 진동 | 관측마다 바뀌는 떨림, RMS 이동 0, 0.25, 0.5, 1, 2, 3cm + 회전 0, 0.25, 0.5, 1, 2, 3° | 기존 연구에 없음 |

조명과 카메라는 고정된 변화(밝기 저하, 틀어짐)와 시시각각 바뀌는 변화(깜빡임, 흔들림)를 짝으로 둔다.

### 물리 외란 — 상자를 잡고 옮기는 힘이 달라진다

| 외란 | 현장 요인 | 단계 0 → 5 | 비고 |
| --- | --- | --- | --- |
| 상자 마찰 `friction` | 그리퍼 패드 마모, 절삭유 | 마찰계수 1.0, 0.8, 0.6, 0.4, 0.25, 0.1 | |
| 상자 질량 `mass` | 내용물 무게 편차 | 기준(0.2kg)의 1, 1.5, 2, 3, 4.5, 6배 | |
| 상자 무게중심 쏠림 `com` | 내용물 쏠림 | 0, 0.4, 0.8, 1.2, 1.6, 2.0cm (단독 시험 때 질량 3배) | 기존 연구에 없음 |

### 외란별로 가정한 현장 상황

**시각 외란**
- **조명 밝기 저하:** 야간 근무라 공장이 어둡거나, 천장 형광등 몇 개가 나간 상황
- **조명 깜빡임:** 오래된 형광등 안정기가 고장 나 깜빡이거나, 큰 설비가 켜질 때 전압이 떨어져 조명이 순간 어두워지는 상황
- **카메라 틀어짐:** 지게차나 작업자가 카메라 거치대를 쳐서 각도가 틀어진 채로 계속 쓰는 상황
- **화면 흔들림:** 옆 프레스가 찍거나 컨베이어가 돌아서 카메라가 계속 떨리는 상황

**물리 외란**
- **상자 마찰:** 그리퍼 고무 패드가 닳았거나, 상자 표면에 절삭유·먼지가 묻어 미끄러운 상황 (단계가 높을수록 마찰계수가 낮아져 더 미끄럽다)
- **상자 질량:** 같은 상자인데 안에 든 부품 개수가 달라 무게가 제각각인 상황
- **무게중심 쏠림:** 운반 중에 상자 안 부품이 한쪽 구석으로 몰린 상황

**7종 동시:** 야간에, 낡은 설비 옆에서, 기름 묻은 무거운 상자를 다루는 것처럼 여러 문제가 한꺼번에 겹친 상황

시각 외란은 영상만 바꾸고 물리는 그대로 두며, 물리 외란은 영상은 그대로 두고 물리만 바꾼다 (상자 겉모습은 같다). 두 종류를 섞으려면 `DIST="light=2,com=4"`처럼 쓰고, 7종을 같은 단계로 한꺼번에 주려면 `DIST="all=3"`이다. 전체 값은 `python run_experiment.py --list-disturbances`로 볼 수 있다.

---

## 사용 방법

### Step 0. 준비

- NVIDIA Isaac Sim 5.1 (`/data/isaacsim/python.sh`)
- LeRobot이 설치된 conda 환경 `lerobot` (정책 서버, 학습, 분석)
- 이 폴더 (`~/factory_robustness`)

```bash
conda activate lerobot
cd ~/factory_robustness
```

### Step 1. 장면 띄워 보기

외란을 넣은 장면을 카메라 두 대로 찍어 본다.

```bash
DIST="light=4,camera=3" PREVIEW_OUT=/tmp/pv /data/isaacsim/python.sh -u checks/preview.py
```

같은 일을 웹 화면에서 슬라이더로 할 수 있다 (Step 4).

### Step 2. 실험 설정 파일 작성

```yaml
name: my_experiment
episodes: 25                 # 조건당 평가 횟수
workers_per_model: 2         # 모델당 동시 실행 수

models:                      # 이름: 체크포인트 폴더 (이름이 <모델>_s<시드>면 시드끼리 묶어 평균 ± 표준편차)
  act: /data/jinju/factory/act_v2_s1/checkpoints/030000/pretrained_model
  dp:  /data/jinju/factory/dp_v2_s1/checkpoints/040000/pretrained_model

runs:
  baseline: {}                                    # 외란 없음
  friction: {sweep: friction, levels: [1,2,3,4,5]}    # 한 외란을 단계별로
  combined: {sweep: all, levels: [1,2,3,4,5]}         # 7종을 같은 단계로 동시에
  custom:   {disturbance: "light=2,com=4"}            # 조합 직접 지정
  exact:    {disturbance: "friction_mu=0.33"}         # 단계 대신 물리값 지정
```

예시: `experiments/quick_demo.yaml` (약 15분), `experiments/paper_main.yaml` (논문 본 실험).

### Step 3. 실행

```bash
python run_experiment.py --config experiments/quick_demo.yaml --dry-run   # 계획만 확인
python run_experiment.py --config experiments/quick_demo.yaml             # 실행
```

결과는 `/data/jinju/factory/experiments/<name>/` 아래에 생긴다.

| 파일 | 내용 |
| --- | --- |
| `<모델>/<조건>.json` | 에피소드별 성공 여부, 실패 유형, 파지 시도 횟수, 적용된 물리값 |
| `analysis/fig_curves.png` | 외란별 단계-성공률 곡선 (모델별, 시드 평균 ± 표준편차, 붕괴 지점 점선) |
| `analysis/breakdown.csv` | 모델 × 외란 붕괴 지점 |
| `analysis/failure_modes.csv` | 실패 유형 집계 |
| `analysis/grasp_stats.csv` | 재시도 통계 (평균 파지 시도, 재시도 비율, 회복 성공) |
| `config_used.yaml` | 실행한 설정 (재현용) |

### Step 4. 웹 화면 (선택)

```bash
~/miniforge3/envs/cosmosgui/bin/python ui_app.py      # 워크스테이션
ssh -L 7870:localhost:7870 ws                        # 내 PC에서 접속 후 http://localhost:7870
```

슬라이더로 외란 값을 고르고 [미리보기]로 카메라 화면을, [평가 실행]으로 성공률과 첫 에피소드 영상을 본다. "일괄 실험 결과" 탭에서 성공률 곡선을 본다.

---

## 구성

| 파일 | 역할 |
| --- | --- |
| `factory_config.py` | 장면의 모든 숫자 (작업대, 컨베이어, 상자, 트레이, 카메라, 조명, 성공 판정) |
| `disturbances.py` | 외란 종류·단계 정의, `DIST` 문자열 해석, 에피소드별 무작위값(시드 고정) |
| `factory_scene.py` | 장면 구성과 외란 적용 (수집·평가·점검이 모두 이것만 쓴다) |
| `eval_client.py` | 외란을 넣고 정책을 돌려 성공 판정 (Isaac Sim) |
| `policy_server.py` | 학습된 LeRobot 정책을 소켓으로 서비스 (ACT·DP·π0 자동 인식) |
| `run_sweep.sh` | 모델 하나에 대해 조건 목록을 병렬 평가 |
| `run_experiment.py` | YAML 설정 → 평가 → 분석 |
| `analyze.py` | 결과 JSON → 곡선·붕괴 지점·실패 유형·재시도 통계 |
| `collect_demos.py`, `ik_expert.py`, `to_lerobot.py` | 정답 제어기 시연 수집 → LeRobot 데이터셋 변환 |
| `checks/` | 외란 검증(무게중심 넘어짐, 정답 제어기 성공률, 장면 사진) |
| `ui_app.py` | 웹 화면 |

## 설계 원칙

- **공정한 비교:** 모든 모델이 같은 상자 위치와 같은 에피소드별 외란 무작위값(카메라가 틀어지는 방향, 내용물 쏠림 방향, 깜빡임 시점)을 받는다 (`eval_seed`).
- **물리적 한계 구분:** 정답 제어기(`ik_expert.py`)로 같은 외란을 먼저 시험한다. 정답 제어기도 실패하는 단계는 정책의 붕괴가 아니라 물리적 한계로 본다.
- **설정했다고 믿지 않고 잰다:** 매 에피소드 실제로 적용된 마찰·질량·무게중심·조명 값을 결과에 기록하고, 검은(렌더 안 된) 프레임이 있으면 그 결과는 분석에서 뺀다.
