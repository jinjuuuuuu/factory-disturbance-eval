# 인계 문서 (2026-10-16 11:10 작성, 회사 PC의 Claude → 노트북의 Claude)

이 문서를 처음부터 읽고 이어서 작업한다. 사용자는 논문을 처음 쓰는 연구원이며, **답은 짧고 쉽게** 해야 한다 (긴 표·설명은 요청할 때만).

## 1. 프로젝트 한 줄

Isaac Sim 5.1 공장 장면(멈춘 가로 컨베이어 위 5cm 상자 → 로봇 옆 파란 트레이)에서 LeRobot으로 학습한 ACT·Diffusion Policy(DP)에 **외란 7종 × 5단계**를 넣어 단계별 성공률 곡선과 **붕괴 지점**(시드 평균 성공률이 외란 없음의 50% 미만으로 처음 떨어지는 단계)을 측정한다. 재학습 없음.

- 마감: **2026-10-23**, 한국정보기술학회 추계 학술대회, **포스터 + 4쪽 논문**
- 외란: 시각 4종(조명 밝기 `light`, 깜빡임 `flicker`, 카메라 틀어짐 `camera`, 화면 흔들림 `shake`) + 물리 3종(마찰 `friction`, 질량 `mass`, 무게중심 `com`) + 7종 동시 `all`
- 평가 조건 41개 = 외란 없음 1 + 7종×5단계 35 + 7종 동시×5단계 5
- 모델: ACT, DP 각각 **학습 시드 3개**, 조건마다 **25회** 평가 (Colosseum·REALM과 같은 횟수), 체크포인트는 마지막(ACT 3만, DP 4만 스텝)으로 고정, 모든 모델이 같은 상자 위치·외란 시드(EVAL_SEED=0)
- 기존 연구에 없는 외란은 **무게중심 쏠림, 조명 깜빡임, 화면 흔들림**만 (카메라 회전은 Colosseum에도 있음 - 주장하지 말 것)
- 정답(IK) 제어기 점검: 외란 없음·질량 6배·무게중심 2cm·마찰 0.4/0.25 모두 8/8. 마찰 0.1은 물리적 한계(이전 배치 0/6) → 논문에서 따로 표시

## 2. 지금 상태 (10/16 11:10)

- 시연 100개 수집 → LeRobot 변환 → 학습: **ACT 시드 3개 완료**, **DP 시드 3개 학습 중 (12:30쯤 끝날 예정)**
- ACT 평가: 123개(시드3×41) 중 **18개 완료** 후 중단 (어젯밤 DP 학습과 GPU를 나눠 써서 에피소드당 20~50분으로 느려짐)
- 워크스테이션에서 `eval_after_train.sh`가 대기 중: DP 학습이 끝나면 **ACT 남은 조건 + DP 전체 평가를 자동 시작** (모델당 워커 2개 = Isaac Sim 12개), 끝나면 분석까지 자동
- 예상: 평가 완료는 10/17(토) 오전 전후 (실제 속도는 평가 시작 후 1시간 지나서 확인 필요)

## 3. 워크스테이션 접속과 확인

- SSH 별칭 `ws` (사용자 hjinju828, 키 인증, sudo 없음). 노트북에 `~/.ssh/config`와 키가 없으면 사용자가 직접 설정해야 한다 (키·비밀번호는 Claude가 입력하지 않는다).
- **접속이 자주 끊긴다** (사내 NAC/보안장비 추정). 1분 안에 두 번 접속하면 끊기므로, 명령은 한 번의 ssh로 묶고, 확인은 90초 이상 간격으로, 실패하면 재시도 루프를 쓴다.
- 진행 확인 (한 번에):

```bash
ssh ws 'tail -5 /data/jinju/factory/overnight_status.txt; for m in act dp; do for s in 1 2 3; do printf "%s_s%s:%s " $m $s $(ls /data/jinju/factory/eval_v2/${m}_s$s/*.json 2>/dev/null | wc -l); done; echo; done; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

- 에피소드 속도 확인: `grep -a "^\[eval\] ep" /data/jinju/factory/eval_v2/act_s1/<조건>.log | tail -3` (끝의 누적 초)
- 결과: `/data/jinju/factory/eval_v2/{act,dp}_s{1,2,3}/<조건>.json`, 분석은 `/data/jinju/factory/analysis_v2/` (fig_curves.png, breakdown.csv, summary.csv, failure_modes.csv, grasp_stats.csv)
- 분석 수동 실행: `cd ~/factory_robustness && ~/miniforge3/envs/lerobot/bin/python analyze.py --eval-root /data/jinju/factory/eval_v2 --out /data/jinju/factory/analysis_v2`
- 웹 화면: `ssh -L 7870:localhost:7870 ws` 후 http://localhost:7870 (`ui_app.py`, conda `cosmosgui`로 실행 중일 수 있음; 7860은 사용자의 Cosmos3 GUI용이라 쓰지 말 것)

## 4. 꼭 지킬 것 / 겪은 함정

- `~/pickplace`, `/data/jinju`의 예전 결과, `/data/lerobot`(공용 설치), conda `lerobot`·`cosmosgui` 환경은 **수정 금지**
- 학습과 평가를 동시에 돌리지 말 것 (GPU 96GB가 차면 둘 다 수십 배 느려짐). 평가는 Isaac Sim 12개 이하.
- 렌더링 생략(`RENDER_LOOKAHEAD`)은 **쓰지 않는다** — 이미지가 달라져 결과가 바뀌었다(A/B 비교로 확인).
- `pgrep -f`/`pkill -f`는 원격 명령줄 자신과도 매칭된다 → 자기 PID(`$$`, `$PPID`) 제외하거나 `[p]attern` 형태로.
- 동시 실행 sweep은 `PORT_BASE`를 서로 다르게 (같으면 다른 모델 서버에 붙음).
- DP는 n_obs_steps=2라 정책 서버가 매 스텝 이미지를 받는다 (ACT보다 평가가 느림).
- 외란 정의를 바꿀 때 기존 조건의 무작위값이 바뀌지 않게 새 값은 `episode_draws`의 **뒤에서** 뽑는다.

## 5. 남은 일 (순서대로)

1. 평가 시작 1시간 후 속도 확인 → 완료 예상 시각을 사용자에게 짧게 보고
2. 결과가 나오면 ACT 먼저 점검(외란 없음 ≥80%, 검은 프레임 0, 5단계에서 하락) → 그림·표를 사용자에게 보여주기
3. 논문 Ⅳ(결과)·Ⅴ(결론) 작성, 요약의 ▢ 채우기 → 아래 논문 초안 문서에 직접 작성
4. 포스터 (3단, 결과 그래프를 가장 크게 - 기획안 문서의 포스터 레이아웃 참고)
5. 미결정/선택: 무게중심+마찰 조합 조건(사용자 미응답), 화면 흔들림 5단계 비교 영상(GPU 한가할 때 `checks/disturbance_video.py`), π0는 시간 남으면

## 6. 문서·자료 위치

- 코드: https://github.com/jinjuuuuuu/factory-disturbance-eval (워크스테이션 `~/factory_robustness`와 같음; README에 사용법·외란 설명)
- 연구 기획안(차별성 비교표, 포스터 레이아웃): https://claude.ai/code/artifact/5e575deb-99d5-40d3-b795-7eb23147153c
- 논문 초안(Ⅰ~Ⅲ장 완성, Ⅳ·Ⅴ 자리): https://claude.ai/code/artifact/b4c18cd1-e24a-4de5-8b2e-df9dea52592a
- 세미나 PPT: 회사 PC `Desktop\세미나 준비\공장외란_붕괴지점_연구요약.pptx` (노트북에는 없을 수 있음)
- 산출물은 PC의 `Desktop`(세미나 자료는 `Desktop\세미나 준비`)에 저장, G: 공유드라이브에는 저장하지 않는다.

## 7. 사용자 선호

- 한국어, **짧고 쉽게**, 결론 먼저
- 요청하지 않은 항목(벤치마크 등)을 임의로 추가하지 않는다
- 비용·방향이 바뀌는 결정은 먼저 묻는다 (실험 설계 변경, 재수집·재학습 등)
