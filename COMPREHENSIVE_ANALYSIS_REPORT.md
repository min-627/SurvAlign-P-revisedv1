# SurvAlign-P Comprehensive Analysis Report

본 리포트는 SurvAlign-P 프로젝트가 원본 논문(AlignMark)의 어떤 한계를 극복하고자 제안되었는지, 수학적·논리적 기반은 무엇인지, 그리고 구체적인 실험(Phase 1, Phase 2)이 어떤 결과를 도출하는지 총체적으로 정리한 문서입니다.

---

## 1. 문제 제기: 원본 논문(AlignMark)의 한계와 BER의 함정

원본 AlignMark는 평균적인 비트 오류율(Bit Error Rate, BER)을 최소화하도록 End-to-End로 학습되었습니다. 하지만 이 접근법은 강한 압축/복원 모델(예: 최신 뉴럴 코덱, 진짜 MP3) 앞에서 치명적인 한계를 드러냅니다.

### "평균 93.7% 정확도"의 함정 (Bit Accuracy Illusion)과 학술적 근거
- 기존 연구들은 Bit Error Rate (BER) 최소화에 집중합니다. 메시지가 16비트일 때 15비트를 맞추면 수치상으로는 93.7%의 높은 Bit Accuracy를 기록합니다. 
- **학술적/수학적 치명성**: 하지만 딥페이크 출처 추적과 같은 포렌식 환경(Forensic Application)에서는 단 1비트의 오류(Hamming distance = 1)만 허용해도 오인식 확률이 폭증합니다. 16비트의 총 경우의 수는 65,536개이며 완벽 일치 시 무작위 오인식 확률(Random False Accept)은 약 0.0015%입니다. 그러나 딱 1비트의 에러를 허용하는 순간, 인정되는 해밍 구(Hamming Sphere)의 범위가 17개로 늘어나 오인식 확률이 **약 17배(0.026%) 폭증**합니다.
- 특히 다수의 유저(후보군)가 존재하는 환경에서는 이 오류 확률의 팽창이 치명적인 동점(Tie)이나 무고한 오인식(False Attribution)을 유발하므로, 포렌식 학계에서는 False Accept Rate (FAR)를 극도로 엄격하게 통제할 것을 요구합니다.
- 즉, 평균 비트 정확도는 높으나 16비트를 완벽하게 맞춘 **Exact-message Accuracy**는 강한 공격 시 절반 이하로 곤두박질치는 현상이 발생하며, 이는 단순한 성능 저하가 아닌 '식별 시스템으로서의 실패'를 의미합니다.

### 학습 환경(Proxy)과 실제 환경(Unseen Codec)의 괴리
- AlignMark는 학습 시 미분이 가능하도록 부드러운 Proxy 공격이나 **Identity STE**(그냥 통과시키는 척하는 수학적 속임수)를 사용했습니다.
- 실제 강한 코덱(FACodec, EnCodec 등)은 사람이 듣지 못하는 주파수 대역을 무자비하게 0으로 날려버립니다(Hard Quantization).
- 골고루 흩뿌려진 AlignMark의 워터마크 에너지는 이 과정에서 통째로 유실되어 복호 실패로 직결됩니다.

---

## 2. 해결 가설 및 최종 문제 정의

이러한 한계를 극복하기 위해 제안된 SurvAlign-P의 문제 정의와 가설은 다음과 같습니다.

> **가설 (Hypothesis)**
> 원본 모델(Embedder/Decoder)을 재학습하지 않고(Post-hoc), 워터마크 잔차(Residual)의 **총 에너지 예산(L2 Norm)을 고정**한 상태에서, 코덱 공격 후에도 살아남는 시간-주파수(T-F) 대역으로 에너지를 지능적으로 재분배(Redistribution)하면 지각 품질(Perceptual Quality)의 저하 없이 Exact-message 복호 실패를 회복할 수 있을 것이다.

---

## 3. 실험 설계와 데이터셋 활용 (데이터 누수 방지)

### 데이터셋 및 누수 방지 (Speaker-Disjoint)
- **사용 데이터셋**: LibriSpeech (train-clean-100), VCTK 등 (`UnifiedSpeechDataset`으로 통합 관리).
- **데이터 격리 원칙**: 원본 논문들이 흔히 범하는 데이터 누수를 막기 위해, 학습에 등장한 화자는 테스트에 절대 등장하지 않도록 **화자 단위 격리(Speaker-Disjoint)**를 기본으로 적용했습니다 (`--combined_protocol speaker_disjoint`).

---

## 4. Phase 1과 Phase 2의 구체적 역할과 결과

연구의 엄밀성을 위해 '학습'과 '평가' 파이프라인을 완전히 분리했습니다.

### Phase 1: 진단 (Controlled Attribution Diagnostics)
- **목표**: "어떤 주파수/시간 대역(T-F)을 우선적으로 살려야 복호가 잘 되는가?"를 학습 없이 순수 수학적으로 검증.
- **실험 방식 (Equal Energy Masking)**: 동일한 에너지 제약 하에서 물리적 생존율(Survival Map), 디코더 민감도(Gradient Saliency), 단순 에너지 크기(Residual Energy) 등 여러 기준(Prior)으로 마스킹을 씌운 뒤 실제 공격을 통과시켜 복원력을 비교.
- **도출 결과**: 통계적 검증(Wilcoxon, Permutation)을 통해 **"물리적인 코덱 생존 대역(Survival Map)을 남기는 것이 디코더의 기울기나 단순 에너지를 남기는 것보다 통계적으로 유의미하게 우월하다"**는 이론적 명분을 증명합니다.

### Phase 2: 학습 및 Paired 평가 (Gate Training & Evaluation)
- **목표**: Phase 1에서 입증된 Survival Map을 무기로 삼아, 실제로 실패했던 워터마크를 살려내는 CNN Gate를 학습.
- **Identity STE의 극복**: 코덱은 미분이 불가능하므로 Gate는 역전파만으로 코덱의 특성을 배울 수 없습니다. 이 모순을 극복하기 위해 물리적 생존 지표인 **Survival Map을 Forward Pass의 Input으로 제공**하여, "디코더가 좋아하는 대역"과 "코덱이 살려두는 대역"을 모두 고려하도록 설계했습니다.
- **도출 결과 (핵심 수치)**:
  1. **Failure Recovery Rate**: 원본(Baseline)이 1~2비트 틀려서 실패했던 샘플들을 우리 방식이 완벽한 16비트로 얼마나 구제(Rescue)했는지 증명.
  2. **Regression Rate**: 원본이 성공했던 것을 망가뜨린 부작용 비율(0에 가까워야 함).
  3. **Ablation 비교**: 순수 생존맵 이동(`analytic_survival`)과 디코더 기반 학습(`proposed_gate`)의 점수를 비교하여 "학습"이 주는 추가적인 가치를 증명.
  4. **Perceptual Quality**: 동일한 L2 에너지 하에서 PESQ, STOI, SI-SDR 점수가 하락하지 않았음을 수치로 보장.

---

## 5. 새로이 분리/도입된 Distortion (Attack) 환경

평가의 신뢰도를 높이기 위해 학습용(Proxy)과 평가용(Real) 공격을 완전히 격리(`--strict_heldout`)했습니다.

1. **학습 및 Map 생성용 (Differentiable or Proxy)**
   - `noise`, `lowpass`, `resample`: 기본적인 신호 왜곡.
   - `reconstruct_nq6`: 미분 가능하도록 Identity STE를 씌운 SpeechTokenizer 공격.
   - `spectral_proxy`: 주파수 대역 압축 프록시.
2. **최종 평가용 (Strict Held-out, Non-differentiable)**
   - **`ffmpeg_mp3`**: 진짜 FFmpeg를 호출하여 물리적 MP3 압축 수행.
   - **`clearervoice`, `facodec`, `encodec`**: 학습에 전혀 쓰이지 않은 완전히 새로운 외부 뉴럴 코덱 구조.

---

## 6. 결론

본 프로젝트는 단순히 "성능을 약간 올리는" 것이 아닙니다. 
원본 모델의 동결, 동일한 잔차 에너지, 완벽히 분리된 화자, 처음 보는 진짜 외부 코덱 공격이라는 **극도로 척박하고 엄격한 통제 환경**에서, 수학적으로 계산된 물리적 생존 지표(Survival Map)를 주입하여 기존 방법론이 구제하지 못했던 Exact-message를 살려내는 **강력한 Post-hoc 파이프라인**입니다.
