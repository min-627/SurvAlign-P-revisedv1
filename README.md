# SurvAlign-P: Feature-Aligned Speech Watermarking with Survival Gate

성균관대학교(SKKU) AAI URP 연구 프로젝트인 **SurvAlign-P**의 공식 코드 저장소입니다.  
본 프로젝트는 기존 딥러닝 기반 블랙박스 오디오 워터마킹(e.g., `AlignMark`, ICME 2026)의 근본적인 한계를 극복하기 위해 고안된 **"사후 가이드형 잔차 최적화(Post-hoc Guided Residual Optimization)"** 네트워크입니다.

---

## 🔬 1. 연구 배경 및 필요성 (Background & Motivation)

### 블랙박스 워터마킹의 한계
기존의 오디오 워터마킹 모델들은 원본 오디오에 비밀 메시지를 숨기기 위해 인코더-디코더(Encoder-Decoder) 구조를 사용합니다. 하지만 이러한 모델들은 학습 시 경험하지 못한 **극한의 실제 환경 왜곡(MP3 압축, 밴드패스 필터링, 잔향 등)**이 발생하면 워터마크 생존율이 급감하는 치명적인 취약점을 가집니다.

### 무분별한 에너지 증폭(Energy Cheating)의 위험성
강건성(Robustness)을 높이기 위한 가장 쉬운 꼼수는 워터마크 신호의 소리(에너지) 자체를 키우는 것입니다. 하지만 이는 원음의 지각적 품질(Fidelity)을 심각하게 훼손합니다. 따라서, **"지각 품질을 전혀 훼손하지 않으면서도 왜곡에 강건하게 살아남도록 정교한 스펙트럼 에너지를 분배하는 기술"**이 본 연구의 가장 핵심적인 필요성입니다.

---

## 💡 2. 핵심 인사이트: Survival Map vs Decoder Gradient Map

기존 딥러닝 워터마크 모델들은 강건성을 높이기 위해 '왜곡 시뮬레이터'와 '디코더'를 한 번에 엮어 거대한 역전파(Backpropagation)를 수행했습니다. 하지만 이 방식은 학습이 매우 불안정하고 새로운 왜곡에 취약합니다. 

이에 SurvAlign-P는 다음과 같은 근본적인 질문을 던집니다:
> **"워터마크 신호가 물리적으로 살아남기 좋은 시간-주파수(T-F) 천연 요새가, 과연 디코더 인공지능이 정답을 맞히기 위해 필사적으로 의존하는 핵심 급소와 일치할까?"**

이를 규명하기 위해 다음 두 가지 맵(Map)을 정의하고 겹쳐봅니다:

| 맵(Map) | 성격 | 의미 | 계산 방식 | 시점 |
|:---|:---|:---|:---|:---|
| **Survival Map** | 물리적 / 직관적 | **물리적 생존율** | 7가지 혹독한 왜곡(FACodec, MP3 등)을 통과시킨 뒤, 각 T-F 픽셀별 잔차 보존율(SIR) 산출 | 왜곡 **후** |
| **Gradient Map** | 딥러닝 / 역전파 | **수학적 민감도** | 입력 오디오 픽셀이 미세하게 변할 때 디코더의 에러율(CE Loss)이 치솟는 오차 역전파 크기 | 왜곡 **전** |

* **상관관계가 갖는 결정적 의의**: 
  만약 이 두 맵의 상관성이 높다면(Phase 1 증명), 우리는 **더 이상 디코더의 무거운 역전파에 의존할 필요가 없습니다.** 단순히 가볍고 직관적인 "물리적 생존율(Survival Map)"을 나침반(Prior) 삼아 워터마크 에너지를 조율(Phase 2)하기만 해도, 디코더가 이를 자연스럽게 찰떡같이 읽어낸다는 강력한 수학적·이론적 근거가 완성되기 때문입니다.

---

## 🧪 3. 2-Phase 실험 설계 및 연구 흐름

본 연구는 통계적 원인 분석(Phase 1)과 실제 인공지능 최적화(Phase 2)를 엄격하게 분리하여 진행합니다.

### Phase 1 (Attribution Analysis)
- **목표**: Survival Map과 Gradient Map 간의 샘플 단위 상관관계(Pearson, Spearman 평균 및 신뢰구간) 및 영역 교집합(Top-20% IoU)을 정량적으로 측정합니다.
- **인과 검증**: 맵의 상위 20% 영역만을 남기는 **절제 연구(Ablation)**를 수행하되, 아티팩트 방지를 위해 **Soft Masking 및 Local Energy Noise Filling**을 적용합니다. 마스킹 전/후의 BER 변화량에 대해 **Paired t-test (p-value < 0.05)**를 수행하여, 해당 영역이 실제 에러율 방어에 결정적인 인과적 기여를 하는지 통계적으로 증명합니다.

### Phase 2 (Survival Gate Training)
- **목표**: Phase 1의 통계적 발견을 바탕으로, 실제 잔차 에너지를 최적화하는 가벼운 AI 모듈(`Survival Gate`)을 학습합니다.
- **연속 점수 조율 (Soft Weighting)**: Phase 1의 거친 이진 마스크 검증과 달리, 본학습에서는 맵의 **연속적인 점수(Continuous Score)**를 가이드(Prior)로 입력받습니다. 3계층 CNN 구조를 통해 `[0.8, 1.2]` 범위의 정교한 연속 가중치를 출력하여, 주파수-시간 픽셀 단위로 워터마크의 강약을 섬세하게 조율합니다.

---

## 📐 4. 수학적·구조적 원리 (Mathematical & Structural Principles)

연구의 학술적 무결성(Integrity)과 공정한 대조군 비교를 위해 완벽한 수학적 제약을 가합니다.

1. **엄격한 L2 Waveform Projection (Hard Constraint)**
   - Gate를 통해 수정된 잔차($r_{gated}$)가 원본 잔차($r_0$)의 전체 에너지를 절대 초과하지 못하도록 **수학적 투영(Projection)** 제약을 강제로 부여합니다. 
   - $ \tilde{r} = r_{gated} \times \min\left(1, \frac{||r_0||_2}{||r_{gated}||_2}\right) $
   - 이를 통해 단순히 신호 에너지를 증폭시켜 강건성을 달성했다는 비판(Energy Cheating)을 구조적/수학적으로 원천 차단합니다.

2. **미분 가능한 디코더 역전파 (Differentiable Decoding)**
   - 원본 AlignMark 모델이 지닌 Vector Quantization(VQ) 병목을 우회하여, 입력 파형부터 최종 CE Loss까지 Autograd Gradient가 단절 없이 완벽하게 흐르도록 아키텍처를 재설계했습니다.

3. **철저한 다중 데이터셋 및 화자 격리 (Speaker Disjoint)**
   - `LibriSpeech` (다화자), `VCTK` (다화자), `LJSpeech` (단일화자) 등 음향 특성이 확연히 다른 3가지 대규모 데이터셋을 동시 지원합니다.
   - 훈련(Train)과 평가(Test) 시 **동일한 화자의 목소리가 겹치지 않도록 철저히 격리(Disjoint)** 분할하여 모델의 과적합(Overfitting)을 방지합니다.

---

## 🚀 5. 시작하기 (Setup Guide)

### 1. 가상환경 구축 및 패키지 설치
Python 3.10+ 환경에서 실행을 권장합니다.

```bash
# 가상환경 생성 및 활성화 (선택)
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # macOS/Linux

# 필수 의존성 패키지 설치
pip install -r AlignMark/requirements.txt
pip install pesq pystoi scikit-learn soundfile torchaudio
```

### 2. 사전 학습된 가중치(Pretrained Weights) 다운로드
대용량 파일 제한 정책으로 인해 아래 가중치 파일들은 GitHub에 포함되어 있지 않습니다. 아래 파일을 직접 다운로드하여 명시된 경로에 배치해 주세요:

1. **AlignMark 모델 가중치** (`weight.pth`)
   - **배치 경로**: `AlignMark/weight.pth`
2. **SpeechTokenizer 모델 가중치** (`SpeechTokenizer.pt`)
   - **배치 경로**: `AlignMark/speechtokenizer/pretrained_model/SpeechTokenizer.pt`

### 3. 데이터셋 일괄 자동 지원
- 코드 실행 시 자동으로 지정된 데이터셋을 감지하여 다운로드 및 로딩을 수행합니다.
- LibriSpeech(16kHz), VCTK(48kHz→16kHz), LJSpeech(22050Hz→16kHz) 모두 일괄된 샘플레이트와 텐서 차원 `(1, 32000)`으로 변환되어 모델에 제공됩니다.

---

## 🏃 6. 전체 실험 진행 순서 (Step-by-Step Guide)

본 프로젝트는 통계적 검증(Phase 1)부터 대규모 AI 본학습 및 평가(Phase 2)까지 논문 작성을 위한 완벽한 파이프라인을 제공합니다. 다음 순서대로 실행하시면 전체 실험 결과를 손쉽게 확보할 수 있습니다.

### Step 1. 가중치 및 환경 세팅 확인
가장 먼저 `AlignMark/weight.pth`와 `SpeechTokenizer.pt` 파일이 제 위치에 있는지 확인합니다. (`5. 시작하기` 항목 참조)

### Step 2. Phase 1: 가설 검증 및 인과성 분석 (필수 통과 관문)
본학습(Phase 2)에 돌입하기 전, 우리의 핵심 가설이 맞는지 데이터셋 단위로 통계적 검증을 수행합니다.

```bash
# LibriSpeech에 대해 Phase 1 분석 실행
python phase1_attribution.py --dataset_type librispeech --batch_size 4
```
* **결과 확인**: 터미널 출력 및 `results/phase1_summary_librispeech.txt`에서 상관계수(r)와 t-test 결과를 확인합니다.
* **분기 조건**: 스크립트가 다단계 로직(r 값 및 p-value 분석)을 거쳐 **"Proceed to Phase 2"**라는 결론을 내리면 다음 스텝으로 넘어갑니다. (시각화된 맵은 `results/phase1_map_comparison.png`에서 확인 가능합니다.)

### Step 3. Phase 2: 대규모 본학습 및 18개 시나리오 평가 (One-Click)
가설이 증명되었다면, 논문에 실을 **최종 성능 테이블(Excel)**을 추출하기 위해 본학습과 평가를 돌립니다. 3개 데이터셋 × 6개 모드를 아우르는 총 18번의 방대한 실험이 자동화되어 있습니다.

```bash
# 18개 실험 일괄 학습 및 평가 (가장 권장)
run_all_experiments.bat
```
* 이 스크립트는 아무 조작을 가하지 않은 `baseline`부터 대조군인 `energy_gate`, 최종 제안 모델인 `proposed_gate`까지 순차적으로 훈련(`Epoch 5`)하고 최고 성능의 모델 체크포인트를 `checkpoints/` 폴더에 자동 저장합니다.
* 각 실험이 끝날 때마다 4가지 오디오 품질 지표(PESQ, STOI 등)와 8가지 왜곡 방어 지표(Clean, FACodec 포함)가 측정됩니다.

### Step 4. 최종 결과표(CSV) 수집 및 논문 작성
모든 실험이 완료되면 `results/phase2_results.csv` 파일이 생성됩니다.
이 CSV 파일에는 데이터셋별, 모드별 성능이 엑셀 표 형태로 예쁘게 누적되어 있습니다. 이 결과를 그대로 복사하여 논문의 Table에 붙여넣고 분석을 진행하시면 됩니다.

---
**💡 추가 평가 트랙 (옵션)**
논문 심사(Defense) 과정에서 리뷰어가 "다른 데이터셋에서는 안 통하는 것 아니냐?" 또는 "원 논문(AlignMark)과 완전히 똑같은 조건에서 비교해봐라" 라고 공격할 경우를 대비하여 다음 트랙을 추가로 구동할 수 있습니다.
* **OOD 일반화 방어 (`--load_weight`)**: LibriSpeech에서 학습된 체크포인트를 VCTK에서 테스트합니다.
  `python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --test_only --load_weight ./checkpoints/best_gate_librispeech_proposed_gate_survival.pth`
* **SOTA 1:1 비교 방어 (`--dataset_type combined`)**: 원 논문과 똑같이 3개 데이터셋을 모두 섞어서 거대 학습을 진행합니다.
  `python phase2_training.py --mode proposed_gate --map_type survival --dataset_type combined`

