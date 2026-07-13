# SurvAlign-P 코드 전면 수정 작업 지시서 (Claude Code용)

> 이 문서는 SurvAlign-P 레포(`https://github.com/ynrch1014/SurvAlign-P-revisedv1`)를 최종 연구계획서 기준으로 전면 수정하고, 학습 속도를 개선하기 위한 작업 지시서입니다. 각 항목은 "무엇을/왜/완료 기준"으로 구성되어 있어, 순서대로 처리하면 됩니다.

---

## 0. 시작 전 확인 사항

1. `git log`로 지금까지 반영된 패치(M0 상관관계 추가, `inprocess_attacks.py`, `tools/prefetch_all_weights.py`, `run_encodec.py`/`run_vocos.py` 버그 수정)가 이미 적용돼 있는지 확인
2. `python smoke_test.py`로 asset-free 스모크 테스트가 통과하는지 먼저 확인 (베이스라인 확보)
3. `weight.pth`, SpeechTokenizer 가중치가 로컬에 있는지 확인 (없으면 `python tools/prefetch_all_weights.py`)
4. GPU 인식 확인: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 1. 정확성/무결성 수정 (속도와 무관, 최우선)

### 1-A. Held-out 무결성 검사를 Phase 2까지 확장 【신규, 중요】
**무엇을**: `phase2_training.py`가 시작할 때, `args.survival_attack_names`(Gate 입력 Survival Map을 만드는 공격)와 cross-codec held-out 테스트 코덱 목록(FACodec, ClearerVoice, DAC, Vocos — 논문에서 "훈련 미노출"이라 주장할 코덱들)을 `experiment_utils.overlapping_attack_families()`로 자동 대조해서, family가 겹치면 경고 또는 `--strict_heldout` 시 중단하도록 추가.

**왜**: Gate의 입력 피처(Survival Map)를 만들 때 쓴 공격이 나중에 "held-out 일반화 증거"로 쓰일 코덱과 겹치면, 논문의 cross-codec 일반화 주장(C10) 자체가 무효화되는 데이터 누출입니다. 지금은 기본값이 우연히 안전할 뿐, 강제되지 않습니다.

**완료 기준**: `survival_attacks`에 실수로 `facodec_proxy` 등을 넣으면 자동으로 경고/중단되는지 유닛 테스트로 확인.

### 1-B. utility_attacks ↔ eval_attacks 기본값 분리 (Phase 1)
**무엇을**: `phase1_attribution.py`의 `--utility_attacks`, `--eval_attacks` 기본값에서 `strong_speechtokenizer` 중복 제거. `--strict_heldout`을 기본 `True`로.

**완료 기준**: 기본 옵션으로 실행 시 경고 없이 끝까지 진행됨.

### 1-C. equal 에너지 모드 기준 노름 재정의
**무엇을**: 현재 "조건 중 최소 노름" 기준을 "Full 잔차 노름의 고정 비율(예: √0.2)"로 변경. bin당 증폭 배율을 로그에 남기는 코드 추가.

### 1-D. 통계 보강
**무엇을**: 마스킹 개입 실험의 다중 비교에 Holm–Bonferroni 보정 추가, Cliff's delta 효과크기 계산 추가, 95% CI 병기, Wilcoxon 검정 시 유효 n(zero-difference 제외 후 실제 표본 수) 별도 보고.

---

## 2. 신규 실험 스크립트

### 2-A. 실험 3: Survival Map 자기 검증 (leave-one-attack-out) 【신규 스크립트】
**무엇을**: `phase1_experiment3_selfcheck.py` 신규 작성. `survival_attacks`에서 공격을 하나씩 빼고 Map을 재생성 → 제외된 공격에서의 실제 retention 예측력(Spearman)을 측정 → 5~6개 공격에 대해 반복.

**참고**: A_surv가 5~6개뿐이라 통계적 검정력이 낮으므로, 정성적 안정성 확인 수준으로 결과를 보고.

### 2-B. M3(유한차분 Codec-Utility) 소규모 서브셋 검증
**무엇을**: `compute_decoder_utility_map`(M2, 1차 근사)과 비교할 유한차분 버전(M3) 추가 — 실제로 top-k만 남기고 공격 후 loss를 직접 측정. 연산량이 크므로 20~30샘플 서브셋에만 적용하는 옵션으로.

---

## 3. 학습 속도 개선 (Phase 2 중심, RunPod RTX 3090 기준)

### 3-A. Survival Map 사전 계산·캐싱 【속도 개선 우선순위 1위로 추정】
**무엇을**: 현재 Gate 훈련 루프가 매 epoch, 심지어 매 배치마다 Survival Map을 새로 계산하고 있는지 확인. 만약 그렇다면, **훈련 시작 전에 전체 데이터셋에 대해 Survival Map을 한 번만 계산해서 디스크(또는 메모리)에 캐싱**하고, 훈련 중에는 캐시에서 불러오도록 변경.

**왜**: Survival Map 생성 자체가 여러 공격(3~5개)을 실제로 걸어보는 무거운 연산입니다(순전파 여러 번). 이게 매 epoch 반복되면 5 epoch만 돼도 5배 낭비입니다. 원본 오디오가 바뀌지 않는 한 Survival Map도 안 바뀌므로 캐싱이 안전합니다.

**확인 방법**: 캐싱 전/후로 1 epoch 소요 시간을 실측 비교.

### 3-B. DataLoader 병목 확인
**무엇을**: `num_workers`, `pin_memory=True`, `persistent_workers=True` 설정 여부 확인. 현재 기본값이 뭔지 확인 후, GPU가 놀고 있는 시간이 있는지(`nvidia-smi -l 1`로 GPU 사용률 관찰) 체크.

### 3-C. Mixed Precision (AMP) 적용 검토
**무엇을**: `torch.cuda.amp.autocast` + `GradScaler` 적용 가능 여부 확인. RTX 3090은 Tensor Core를 지원하므로 AMP로 상당한 속도 이득이 기대됨. 단, decode loss 계산의 수치 안정성(특히 CE loss)에 문제가 없는지 소규모 실행으로 먼저 검증.

### 3-D. train_attacks 8종을 매 스텝 전부 도는 구조 재검토
**무엇을**: 현재 코드는 매 스텝마다 같은 배치에 대해 `train_attacks`(8개)를 **전부** 순차 실행하고 loss를 평균 냅니다(무작위로 하나만 뽑는 게 아님). 이건 의도된 설계(모든 조건을 균일하게 경험)이므로 **바꾸지 말 것** — 대신 이 8개의 공격 실행 자체를 배치 차원으로 합쳐서 한 번의 순전파로 처리할 수 있는지(예: `torch.vmap` 또는 배치를 8배로 늘려 병렬 처리) 검토. 안 되면 최소한 8개 공격 각각이 불필요하게 모델을 다시 로딩하고 있지 않은지만 확인(이전에 encodec/vocos에서 겪은 것과 같은 종류의 병목이 다른 공격에도 있는지).

### 3-E. encodec/vocos ID 트랙 완전 인프로세스화 재확인
**무엇을**: 지난 세션에서 만든 `inprocess_attacks.py`가 Phase 2 훈련 루프(`phase2_training.py`)에서도 쓰이고 있는지 확인 — 혹시 Phase 1에만 연결돼 있고 Phase 2는 여전히 예전 서브프로세스 방식(`--encodec_command`)을 쓰고 있다면 동일하게 연결.

### 3-F. 배치 크기 튜닝
**무엇을**: RTX 3090(24GB) 기준으로 VRAM 사용률을 보며 배치 크기를 늘릴 수 있는 만큼 늘려서 처리량(throughput) 확인. 단, 배치 크기가 바뀌면 learning rate도 함께 조정이 필요할 수 있음(선형 스케일링 규칙 고려).

**전체 완료 기준**: 3-A~3-F 적용 전/후로 "1 epoch, 동일 데이터 규모" 소요 시간을 실측 비교한 표를 만들어서 어떤 최적화가 실제로 효과 있었는지 정량적으로 보고.

---

## 4. 문서 정정 (코드 수정과 별개, 마지막에)

- "전적으로" → "지배적으로(λ=4.0)"
- "Shortcut Learning" → "훈련 분포 내 지역 최적점 + 미분 불가능 실공격의 구조적 학습 불가능성"
- "물리적 prior" → "직접 측정 기반 prior"
- "decoder-free" → "Map 생성: decoder-free / Gate 학습: decoder-dependent"로 분리
- `experiment_utils.py`의 "N=100" 주석 → "N=300"으로 수정

---

## 5. 작업 방식 제안

1. **1번(무결성)부터 순서대로**: 정확성이 속도보다 우선. 잘못된 속도 최적화보다 틀린 결과가 훨씬 나쁨.
2. **각 항목마다 "고치기 전/후 비교"를 남길 것**: 특히 3번(속도) 항목들은 각각 독립적으로 켜고 꺼보면서 어떤 게 실제로 효과 있었는지 표로 기록 — 나중에 논문 재현성 섹션이나 팀 공유 시 필요.
3. **작은 규모로 먼저 검증 후 확대**: 모든 변경사항을 `--max_samples 20` 같은 소규모로 먼저 돌려서 에러 없는지 확인한 뒤, 전체 규모로 확대.
4. **기존 `smoke_test.py` 패턴 활용**: 새로 만드는 기능(1-A, 2-A, 2-B)에도 가능하면 가짜 모델(mock)로 로직만 검증하는 asset-free 테스트를 추가해두면, GPU 없이도 빠르게 회귀 확인 가능.
