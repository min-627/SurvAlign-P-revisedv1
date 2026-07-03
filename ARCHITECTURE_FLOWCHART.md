# SurvAlign-P 전체 시스템 파이프라인 (Architecture Flowchart)

본 문서는 오리지널 AlignMark 모델과 새로 구축된 SurvAlign-P 파이프라인의 **파일 간 상호작용 및 역할 분담**을 한눈에 파악하기 위한 구조도입니다.

## 📦 전체 코드 흐름도

```text
[오리지널 AlignMark 폴더] (블랙박스/건드리지 않음)
         │
         ▼
[ 1. survalign_p.py ] (기반 엔진 / 뼈대)
         │  - 기존 AlignMark를 부르기 쉽게 포장 (AlignMarkManager)
         │  - 화자 격리 분할 및 다중 데이터셋 로딩 (UnifiedSpeechDataset)
         │  - 6가지 오디오 왜곡 필터 구현 (DifferentiableDistortion)
         │  - *단독으로 실행하는 파일이 아니며, 부품 창고 역할 수행*
         │
         ├──▶ [ 2. phase1_attribution.py ] (Phase 1 실행기 / 분석기)
         │       - survalign_p.py에서 엔진을 빌려와 "상관관계 분석" 수행
         │       - Survival Map과 Gradient Map 비교 및 인과 마스킹 테스트
         │       - (실행 리모컨: run_phase1.bat)
         │
         └──▶ [ 3. phase2_training.py ] (Phase 2 실행기 / 훈련 및 평가기)
                 - survalign_p.py에서 엔진을 빌려와 "게이트(Survival Gate) 학습" 수행
                 - 5가지 시나리오(Baseline, Uniform, Random, Survival, Gradient) 작동
                 - 평가(Testing)까지 단일 파이프라인에서 연속 수행하여 CSV 파일에 로깅
                 - (실행 리모컨: run_all_experiments.bat, test_all_experiments.bat 등)
```

## 🔍 파일별 명확한 역할 정의

### 1. `survalign_p.py` (우리가 만든 심장부)
* **새롭게 작성된 베이스캠프**: 기존 AlignMark 구조를 수정하는 것이 아니라, 편하게 불러오고 조작하기 위해 모델을 감싸고(Wrapper) 있는 기반 코드입니다.
* 데이터셋을 정교하게 나누고(Train/Calib/Test), 파이토치(PyTorch) 기반 미분 가능한(Differentiable) 6가지 음향 왜곡 필터를 제공합니다.

### 2. `phase1_attribution.py` (통계 및 원인 분석)
* `survalign_p.py` 모듈을 Import하여 오직 **가설 검증**만을 수행합니다.
* "Survival Map(물리적 생존 구역)이 디코더의 실제 힌트(Gradient)와 일치하는가?"를 통계적으로 증명합니다.

### 3. `phase2_training.py` (실제 AI 모듈 학습 및 검증)
* 제안하는 얇은 인공지능 모듈인 `SimplifiedSurvivalGate`가 포함되어 있습니다.
* 명령어(--mode)에 따라 Baseline 대조군 실험부터 실제 제안 기법(Proposed Gate)까지 모델을 훈련하고, 훈련 직후 6대 왜곡 테스트를 통해 Robustness(강건성) 지표를 도출합니다.

### 4. `.bat` 파일들 (실행 리모컨)
* **학습 + 평가 일괄 진행**: `run_all_experiments.bat`
* **평가만 일괄 진행 (저장된 모델 불러오기)**: `test_all_experiments.bat`
* 개별 실험 진행: `run_baseline.bat`, `run_proposed_gradient.bat` 등
* 터미널 환경이 낯설어도 더블클릭만으로 모든 논문용 결과(CSV 및 통계)를 얻어낼 수 있도록 돕는 단축 스크립트입니다.

---
**💡 요약**
이 구조적 분리는 연구의 확장성을 극대화합니다. 향후 완전히 새로운 왜곡 환경을 추가하거나 데이터셋을 교체할 일이 생기더라도 `phase1`이나 `phase2` 코드는 전혀 건드릴 필요 없이, **오직 부품 창고인 `survalign_p.py`만 수정하면 전역에 자동 반영**됩니다.
