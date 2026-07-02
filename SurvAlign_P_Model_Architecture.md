# SurvAlign-P 모델 세부 아키텍처 (Model Architectures)

> [!NOTE]
> 본 문서는 두 가지 제안 기법(후보군)인 **Survival Map 기반 아키텍처**와 **Gradient Map 기반 아키텍처**의 내부 데이터 흐름(Data Flow)을 도식화한 것입니다. 논문의 모델 구조도(Figure)를 그리실 때 이 다이어그램을 참고하시면 매우 유용합니다.

---

## 1. Candidate A: Survival Map 기반 아키텍처
**핵심 철학**: "물리적으로 시끄러운 환경(왜곡)을 거친 뒤에도 씻겨 내려가지 않는 주파수 대역을 칠하자."

```mermaid
graph TD
    %% 입력 계층
    InAudio([원본 오디오 파형]) --> AM_Embed
    InMsg([비밀 메시지]) --> AM_Embed
    
    %% AlignMark 블랙박스
    subgraph AlignMark [오리지널 AlignMark 엔진 / Frozen]
        AM_Embed[AlignMark Embedder]
    end
    
    %% 베이스라인 출력
    AM_Embed -->|1차 출력물| WavWM[워터마크 오디오]
    AM_Embed -->|잉크| Res[초기 잔차 Residual]
    
    %% Survival Map 계산부 (물리적 필터)
    WavWM -.-> DistSim[6대 왜곡 시뮬레이터]
    DistSim -.-> SurvCalc[신호 대 간섭비 SIR 계산]
    SurvCalc -.-> SMap(Survival Map)
    
    %% 게이트 네트워크 (우리의 핵심 AI)
    subgraph SurvAlign_Gate [SurvAlign-P Gate Network / Trainable]
        FeaturePack[Feature 묶기: 원본 + 잔차 + Map]
        CNN[3-Layer CNN & GroupNorm]
        FeaturePack --> CNN
        CNN --> GScale(Gate Scale 1.0 ± 0.2)
    end
    
    %% 흐름 제어
    InAudio --> FeaturePack
    Res --> FeaturePack
    SMap --> FeaturePack
    
    %% 최종 병합
    Res --> Multiply{X 곱하기}
    GScale --> Multiply
    Multiply --> GatedRes[증폭된 최종 잔차]
    
    InAudio --> Add{+ 더하기}
    GatedRes --> Add
    Add --> FinalWav([최종 SurvAlign-P 오디오])
    
    %% 최종 디코딩 테스트
    FinalWav -->|실제 배포 환경| FinalDist[각종 왜곡 발생]
    FinalDist --> AM_Dec[AlignMark Decoder]
    AM_Dec --> OutMsg([해독된 메시지])
```

---

## 2. Candidate B: Gradient Map 기반 아키텍처
**핵심 철학**: "디코더 인공지능이 메시지를 읽을 때 역전파(Backprop) 수치가 크게 요동치는, 즉 수학적으로 가장 민감한 취약점 대역을 칠하자."

```mermaid
graph TD
    %% 입력 계층
    InAudio([원본 오디오 파형]) --> AM_Embed
    InMsg([비밀 메시지]) --> AM_Embed
    
    %% AlignMark 블랙박스
    subgraph AlignMark [오리지널 AlignMark 엔진 / Frozen]
        AM_Embed[AlignMark Embedder]
        AM_Dec_Probe[AlignMark Decoder / 취약점 탐색용]
    end
    
    %% 베이스라인 출력
    AM_Embed -->|1차 출력물| WavWM[워터마크 오디오]
    AM_Embed -->|잉크| Res[초기 잔차 Residual]
    
    %% Gradient Map 계산부 (수학적 취약점)
    WavWM -.-> AM_Dec_Probe
    InMsg -.-> Loss[Cross-Entropy Loss 계산]
    AM_Dec_Probe -.-> Loss
    Loss -.->|오차 역전파| Backprop[Backpropagation]
    Backprop -.-> GMap(Gradient Map)
    
    %% 게이트 네트워크 (우리의 핵심 AI)
    subgraph SurvAlign_Gate [SurvAlign-P Gate Network / Trainable]
        FeaturePack[Feature 묶기: 원본 + 잔차 + Map]
        CNN[3-Layer CNN & GroupNorm]
        FeaturePack --> CNN
        CNN --> GScale(Gate Scale 1.0 ± 0.2)
    end
    
    %% 흐름 제어
    InAudio --> FeaturePack
    Res --> FeaturePack
    GMap --> FeaturePack
    
    %% 최종 병합
    Res --> Multiply{X 곱하기}
    GScale --> Multiply
    Multiply --> GatedRes[증폭된 최종 잔차]
    
    InAudio --> Add{+ 더하기}
    GatedRes --> Add
    Add --> FinalWav([최종 SurvAlign-P 오디오])
    
    %% 최종 디코딩 테스트
    FinalWav -->|실제 배포 환경| FinalDist[각종 왜곡 발생]
    FinalDist --> AM_Dec[AlignMark Decoder]
    AM_Dec --> OutMsg([해독된 메시지])
```

---

### 💡 두 아키텍처의 결정적 차이점 비교
위 흐름도를 보시면 두 모델의 차이는 오직 **"Feature Pack(특징 묶음)에 세 번째 재료로 무엇을 넣어주는가?"**에 있습니다.

* **Candidate A (Survival)**: 왜곡 시뮬레이터(`DistSim`)를 통과시켜 살아남는 소리의 크기를 계산한 물리적인 지도(`Survival Map`)를 넣습니다. 디코더를 미리 열어보지 않고 순수하게 물리 법칙에 의존합니다.
* **Candidate B (Gradient)**: 디코더(`AM_Dec_Probe`)에 오디오를 한 번 넣어본 뒤, 디코더가 아파하는 수학적 미분값(`Gradient Map`)을 역으로 추적하여 넣습니다. 인공지능의 내부 취약점 공략에 의존합니다.

나머지 부분, 즉 **AlignMark의 출력을 가로채서 3-Layer CNN을 통해 곱해준 뒤(X) 더해서(+) 내보낸다**는 핵심 매커니즘은 두 아키텍처 모두 완벽하게 동일합니다!
