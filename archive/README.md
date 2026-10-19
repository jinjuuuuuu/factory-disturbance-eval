# archive - 지금은 쓰지 않는 스크립트

실험 과정에서 한 번 쓰고 더는 쓰지 않는 파일이다. 기록으로만 남긴다 (실행하지 말 것).

| 파일 | 썼던 이유 / 안 쓰는 이유 |
| --- | --- |
| `pipeline_train.sh`, `train_seeds.sh` | 첫 배치(v1: 컨베이어가 로봇과 겹치던 장면)의 학습. 지금은 `pipeline_v2.sh` |
| `eval_all.sh`, `eval_after_train.sh` | 10/15~16 평가 실행. 지금은 `run_experiment.py` |
| `overnight.sh`, `render_ab.sh` | 렌더링 생략(관측 직전만 렌더링) 판정용. 이미지가 달라져 결과가 바뀌어 쓰지 않기로 했고, 옵션도 `eval_client.py`에서 뺐다 |
| `conditions/*.txt` | 10/17~18 현장 조합·연속 진동 평가에 쓴 조건 목록. 지금은 `experiments/paper_main.yaml`이 같은 조건을 담는다. `combo_factory.txt`의 `camera=k,shake=k`는 사용 중단된 무작위 흔들림 조합이다 |
