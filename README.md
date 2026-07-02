# SurvAlign-P: Feature-Aligned Speech Watermarking with Survival Gate

성균관대학교(SKKU) AAI URP 연구 프로젝트인 **SurvAlign-P**의 공식 코드 저장소입니다.  
본 프로젝트는 기존 `AlignMark` (ICME 2026) 오디오 워터마킹 백본의 한계를 극복하고, 지각 품질(Fidelity) 저하 없이 다양한 실제 환경 채널 왜곡 하에서의 강건성(Robustness)을 개선하기 위해 설계되었습니다.

---

## ⚙️ 주요 아키텍처 특징

1. **6채널 Survival Gate**:
   - 원본 음성 스펙트로그램, AlignMark 초기 잔차, 다중 왜곡 생존율 맵, 지각 마스킹 프록시, STFT 기반 Soft VAD 및 오리지널 잔차 프라이어의 총 6가지 시간-주파수 특징을 활용해 최적의 스펙트럼 에너지 게이팅 가중치를 예측합니다.
2. **미분 가능한 디코더 직접 역전파 (Bypassing VQ Bottleneck)**:
   - 디코딩 경로 상에 Vector Quantization(VQ)이 배제되어, 실제 디코더(frozen)를 통과하여 입력 파형까지 완벽하게 autograd gradient가 전달됩니다.
3. **L2 Waveform Projection 제약**:
   - 수정 잔차의 L2 에너지가 원래 AlignMark 잔차 에너지 한도를 넘지 않도록 제한하여 단순히 에너지를 올려 성능을 향상시키는 꼼수를 차단합니다.
4. **Presence Head & Calibration**:
   - 디코딩 에비던스 특징 벡터(Entropy, Margin, 확률)에 기반한 Presence Head를 제공하며, Calibration 화자 데이터로 오탐률 1% 미만의 판단 임계값을 보정합니다.

---

## 🚀 시작하기 (Setup Guide)

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

*(연구실 공유용 구글 드라이브 혹은 드롭박스 링크를 여기에 추가하세요)*

### 3. 디렉토리 구조 검증
정상 실행을 위해 클론한 프로젝트의 경로 구성이 아래와 같은지 확인해 주세요:
```text
├── .gitignore
├── README.md
├── survalign_p.py                    # SurvAlign-P 엔진 코드
├── survalign_p_technical_report.md   # 최종 기술 명세서
└── AlignMark/
    ├── weight.pth                    # [다운로드 파일]
    ├── speechtokenizer/
    │   └── pretrained_model/
    │       ├── SpeechTokenizer.pt    # [다운로드 파일]
    │       └── speechtokenizer_hubert_avg_config.json
    ...
```

---

## 🏃 실행 및 평가 (Running & Evaluation)

### 데이터셋
학습용 LibriSpeech 데이터셋은 **코드 실행 시 자동으로 웹에서 다운로드 및 압축 해제가 진행**됩니다.
- **dev-clean**: 약 338MB (기본값)
- **train-clean-100**: 약 6.3GB

코드 내부에서 Speaker ID를 격리하여 자동으로 학습(80%), 보정(10%), 최종 테스트(10%) 화자 풀로 분할됩니다.

### 실행 명령 및 옵션
`survalign_p.py`는 다음과 같은 다양한 명령줄 실행 인자를 지원합니다:

```bash
# 기본 설정으로 학습 진행 (n_gate_steps=1000, n_presence_steps=500, dev-clean 사용)
python survalign_p.py

# 3분 이내로 전체 코드 동작을 점검하기 위한 초고속 Sanity Check 실행
python survalign_p.py --sanity_check

# 대용량 train-clean-100 데이터셋으로 대규모 학습 진행
python survalign_p.py --dataset_name train-clean-100 --n_gate_steps 2000 --n_presence_steps 1000

# 하이퍼파라미터 조절 예시
python survalign_p.py --batch_size 8 --lr_gate 5e-5 --lr_presence 1e-4
```

### 실행 인자 상세 설명
- `--dataset_name`: 다운로드 및 학습에 사용할 LibriSpeech 데이터셋 이름 (`dev-clean` 또는 `train-clean-100`).
- `--n_gate_steps`: Survival Gate 학습용 최대 스텝 수 (기본값: 1000).
- `--n_presence_steps`: Presence Head 학습용 최대 스텝 수 (기본값: 500).
- `--n_eval_steps`: 최종 평가 시 평가할 오디오 샘플 수 (기본값: 100).
- `--sanity_check`: 활성화 시 즉시 학습 스텝을 초소형화(5스텝/5스텝/2샘플)하여 테스트 작동.
- `--batch_size`: 데이터 로더 배치 크기 (기본값: 4).
- `--lr_gate`: Survival Gate 옵티마이저 학습률 (기본값: 1e-4).
- `--lr_presence`: Presence Head 옵티마이저 학습률 (기본값: 5e-4).

### 모델 저장 및 체크포인트
학습이 정상 완료되면 현재 작업 디렉토리에 다음 파일들이 자동으로 생성 및 저장됩니다:
- `gate_checkpoint.pth`: 학습이 완료된 Survival Gate 가중치.
- `presence_checkpoint.pth`: Presence Head 가중치 및 명목 FPR <= 1% 기준의 탐지 임계값(`tau_p`) 정보.

### 출력 리포트
코드가 완료되면 다음과 같은 종합 지표가 **평균 및 95% 신뢰구간 (±95% CI)**을 포함하여 표 형태로 출력됩니다:
- **BER (Bit Error Rate)**: AWGN, Lowpass, Bandpass, Resample, RVQ Reconstruct, MP3 Proxy 조건 하에서 기본형 대비 개선도(pp 단위) 리포트.
- **Audio Quality**: PESQ WB, STOI, SI-SDR의 1:1 오디오 쌍 비교 평가.
- **Open-set Detection**: Calibrated FPR 임계치에서의 Test TPR 및 탐지 AUROC.
