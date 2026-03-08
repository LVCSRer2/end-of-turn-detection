# Semantic Turn Detector 비교 분석

NAMO Turn Detector, LiveKit Turn Detector, Turnsense 세 프로젝트를 비교 분석한 문서입니다.

---

## 1. 기본 정보

| 항목 | NAMO (VideoSDK) | LiveKit Turn Detector | Turnsense (latishab) |
|------|-----------------|----------------------|---------------------|
| **개발** | VideoSDK | LiveKit | Latisha Besariani (개인) |
| **기반 모델** | mmBERT (다국어) / DistilBERT (단일언어) | Qwen2.5-0.5B (다국어) / SmolLM2-135M (영어) | SmolLM2-135M |
| **모델 크기** | ~135MB (단일) / ~295MB (다국어) | ~66MB (영어) / ~281MB (다국어) | ~135M 파라미터 |
| **라이선스** | Apache 2.0 | Open-weights (커스텀) | Apache 2.0 |
| **GitHub** | [videosdk-live/NAMO-Turn-Detector-v1](https://github.com/videosdk-live/NAMO-Turn-Detector-v1) | [livekit/agents](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-turn-detector) | [latishab/turnsense](https://github.com/latishab/turnsense) |

---

## 2. 성능 비교

| 항목 | NAMO | LiveKit | Turnsense |
|------|------|---------|-----------|
| **추론 속도** | <19ms (단일) / <29ms (다국어) | 15~45ms (영어) / 50~160ms (다국어) | 명시 없음 |
| **정확도** | 최대 97.3% (단일) / 평균 90.25% (다국어) | TP 98.8%, TN 87.5% (영어) | 97.5% (표준) / 93.75% (양자화) |
| **평가 데이터** | 25,000+ 다국어 발화 | 자체 벤치마크 | TURNS-2K (2,000 샘플) |
| **RAM 요구량** | 명시 없음 | <500MB | 경량 (Raspberry Pi 구동 가능) |

---

## 3. 언어 지원

| 항목 | NAMO | LiveKit | Turnsense |
|------|------|---------|-----------|
| **지원 언어 수** | 23개 | 14개 | 영어만 |
| **한국어** | ✅ | ✅ | ❌ |
| **일본어** | ✅ | ✅ | ❌ |
| **중국어** | ✅ | ✅ | ❌ |

### NAMO 지원 언어 (23개)

Turkish, Korean, Japanese, German, Hindi, Dutch, Norwegian, Chinese, Finnish, English, Polish, Indonesian, Italian, Danish, Portuguese, Spanish, Marathi, Ukrainian, Russian, Vietnamese, French, Arabic, Bengali

### LiveKit 지원 언어 (14개)

Chinese, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Russian, Spanish, Turkish

---

## 4. 모델 아키텍처 및 크기 비교

### NAMO Turn Detector v1

| 구분 | 다국어 모델 | 언어별 특화 모델 |
|------|-----------|----------------|
| **베이스 모델** | mmBERT (Multilingual BERT) | DistilBERT (distilbert-base-multilingual-cased) |
| **모델 유형** | 인코더 (Encoder-only) | 인코더 (Encoder-only) |
| **파라미터 수** | ~178M | ~66M |
| **ONNX 파일 크기** | ~295MB (양자화) | ~135MB (비양자화) |
| **최적화** | ONNX 양자화 (2.19x 속도 향상, 정확도 손실 <0.2%) | ONNX (양자화 버전 결함 있음) |
| **추론 속도** | <29ms (양자화 전 61.3ms → 28.0ms) | <19ms (양자화 전 38ms → 14.9ms) |
| **출력 방식** | 이진 분류 (0: CONT, 1: EOT) + softmax 확률 | 동일 |
| **지원 언어** | 23개 | 언어별 1개 |

### LiveKit Turn Detector

| 구분 | 다국어 모델 (v0.4.1-intl) | 영어 전용 (deprecated) |
|------|-------------------------|----------------------|
| **베이스 모델** | Qwen2.5-0.5B-Instruct | SmolLM2-135M |
| **교사 모델** | Qwen2.5-7B-Instruct (Knowledge Distillation) | - |
| **모델 유형** | 디코더 (Decoder-only, LLM) | 디코더 (Decoder-only) |
| **파라미터 수** | ~500M | ~135M |
| **ONNX 파일 크기** | ~281MB (INT8 양자화) | ~66MB |
| **추론 속도** | 50-160ms | 15-45ms |
| **출력 방식** | `<\|im_end\|>` 토큰 출현 확률 + 언어별 임계값 비교 | 동일 |
| **지원 언어** | 14개 | 영어만 |

- Knowledge Distillation: 7B 교사 → 0.5B 학생 모델로 증류, ~1,500 스텝에서 수렴
- VAD와 결합하여 사용 (대화 맥락 4턴 슬라이딩 윈도우)
- v0.3 → v0.4.1에서 false-positive 중단 39% 감소 달성
- 전화번호, 주소, 이메일 등 구조화 데이터 처리 강화

### Turnsense

| 구분 | 값 |
|------|---|
| **베이스 모델** | SmolLM2-135M (HuggingFace) |
| **모델 유형** | 디코더 (Decoder-only) |
| **파라미터 수** | ~135M |
| **학습 방식** | LoRA Fine-tuning |
| **학습 데이터** | TURNS-2K (2,000 샘플, 영어) |
| **출력 방식** | 이진 분류 (EOT / CONT) |
| **지원 언어** | 영어만 (한국어 불가 — 베이스 모델에 한국어 학습 데이터 없음) |

- 엣지 디바이스(Raspberry Pi) 최적화가 핵심 목표
- SmolLM2-135M은 영어 + 유럽 5개 언어만 지원 (한국어 미포함)

### 내부 동작 상세: NAMO (인코더)

```
입력: "오늘 저녁에 뭐 할"
                │
                ▼
┌─────────────────────────────────┐
│  1. 토크나이저 (WordPiece)       │
│  [CLS] 오늘 저녁 ##에 뭐 할 [SEP] │
│  → input_ids, attention_mask    │
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  2. BERT 인코더 (양방향 Attention) │
│                                  │
│  모든 토큰이 서로를 동시에 참조    │
│  "할" ←→ "뭐" ←→ "저녁"          │
│  ←→ "오늘" ←→ [CLS]              │
│                                  │
│  12 layers × 12 heads (mmBERT)   │
│  6 layers × 12 heads (DistilBERT)│
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  3. [CLS] 토큰의 hidden state    │
│  → 문장 전체의 의미 요약 벡터      │
│  shape: (768,)                   │
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  4. 분류 헤드 (Linear + Softmax)  │
│  (768,) → Linear(768, 2) → (2,) │
│  → softmax → [0.99, 0.01]       │
│               CONT   EOT        │
└─────────────────────────────────┘
                │
                ▼
        argmax → CONT (99%)
```

- **양방향(Bidirectional) Attention**: "할"이라는 단어가 앞의 "뭐", "저녁에"와 뒤의 관계를 동시에 참조
- **[CLS] 토큰**: 문장 전체 의미를 하나의 벡터(768차원)로 압축. 이 벡터 하나만으로 분류
- **한 번의 Forward Pass**: 입력 전체를 한꺼번에 처리 (토큰 생성 없음)
- **출력 차원 2**: CONT/EOT 2개 클래스에 대한 확률만 계산 → 빠름

### 내부 동작 상세: LiveKit (디코더)

```
입력: "오늘 저녁에 뭐 할"
                │
                ▼
┌──────────────────────────────────────┐
│  1. 텍스트 정규화                      │
│  NFKC 정규화 → 소문자 → 구두점 제거    │
│  "오늘 저녁에 뭐 할" → "오늘 저녁에 뭐 할" │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  2. Qwen 채팅 템플릿 포맷              │
│                                      │
│  <|im_start|>user                    │
│  오늘 저녁에 뭐 할                     │
│                                      │
│  ↑ 여기서 <|im_end|> 를 일부러 제거!   │
│  → 모델에게 "이 턴이 끝났는가?" 를 질문  │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  3. 토크나이저 (BPE, 좌측 절삭)        │
│  max_length=128, truncation_side=left│
│  → input_ids (int64)                 │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  4. Qwen2.5-0.5B 디코더 (단방향)      │
│                                      │
│  각 토큰은 자신보다 앞의 토큰만 참조    │
│  "오늘" → "저녁" → "에" → "뭐" → "할"  │
│     ↓        ↓       ↓      ↓     ↓   │
│  Causal Mask (삼각형 어텐션)           │
│                                      │
│  24 layers × 14 heads               │
│  hidden_size: 896                    │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  5. 마지막 토큰의 logits 추출          │
│                                      │
│  output shape: (1, seq_len, 151936)  │
│  → 마지막 위치 logits: (151936,)      │
│     = 전체 어휘에 대한 다음 토큰 확률    │
│                                      │
│  여기서 <|im_end|> 토큰의 확률만 추출   │
│  → eou_probability = 0.0024          │
└──────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  6. 언어별 임계값 비교                  │
│                                      │
│  한국어 임계값: 0.0156                  │
│  0.0024 < 0.0156 → CONT (발화 중)     │
└──────────────────────────────────────┘
```

- **단방향(Causal) Attention**: 각 토큰은 자신보다 앞의 토큰만 참조 (뒤를 못 봄)
- **다음 토큰 예측**: "할" 다음에 `<|im_end|>`가 올 확률이 높으면 EOT
- **채팅 템플릿 트릭**: 마지막 `<|im_end|>`를 제거해서 모델이 "여기서 턴이 끝나는가?"를 자연스럽게 예측
- **대화 맥락 포함**: 이전 턴(최대 4턴)을 함께 넣어 문맥 파악 가능
- **출력 차원 151,936**: Qwen 전체 어휘 확률을 계산하므로 상대적으로 느림

### 인코더 vs 디코더 핵심 차이 요약

| | NAMO (인코더) | LiveKit (디코더) |
|--|--------------|-----------------|
| **Attention** | 양방향 (모든 토큰 ↔ 모든 토큰) | 단방향 (앞 → 뒤만) |
| **무엇을 계산?** | [CLS] 벡터 → 2-class 분류 | 전체 어휘 확률 중 `<\|im_end\|>` 하나만 추출 |
| **출력 차원** | 2 (CONT, EOT) | 151,936 (Qwen 전체 어휘) 중 1개 |
| **대화 맥락** | 현재 발화만 | 최대 4턴 이전 대화 포함 |
| **판별 기준** | softmax argmax (0 vs 1) | 확률 > 언어별 임계값 |
| **속도** | 빠름 (분류 전용, 출력 2차원) | 느림 (LLM Forward, 출력 15만 차원) |

---

## 5. 학습 데이터

| 프로젝트 | 학습 데이터 | 규모 | 공개 여부 |
|----------|------------|------|----------|
| **NAMO** | 자체 수집 다국어 발화 데이터 | 25,000+ | 비공개 추정 |
| **LiveKit** | 자체 수집 + 다양한 STT 출력 포맷 | 명시 없음 | 비공개 |
| **Turnsense** | [TURNS-2K](https://huggingface.co/datasets/latishab/turns-2k) | 2,000 샘플 | 공개 |

### Turnsense TURNS-2K 데이터셋 특징

- 백채널(backchannel) 포함
- 자가 수정(self-correction) 포함
- 코드 스위칭(code-switching) 포함
- 다양한 STT 출력 포맷 포함

---

## 6. LiveKit 다국어 모델 버전별 개선 (v0.3.0 → v0.4.1)

TP rate 99.3% 기준 false-positive rate 비교:

| 언어 | v0.3.0 | v0.4.1 | 개선율 |
|------|--------|--------|--------|
| English | 16.60% | 13.00% | 21.69% |
| Dutch | 26.10% | 11.90% | 54.33% |
| German | 23.40% | 12.20% | 47.86% |
| Spanish | 21.50% | 14.00% | 33.88% |
| French | 16.80% | 11.10% | 33.93% |
| Japanese | 19.70% | 11.20% | 43.15% |

---

## 7. 통합 및 생태계

| 항목 | NAMO | LiveKit | Turnsense |
|------|------|---------|-----------|
| **SDK 통합** | VideoSDK Agents SDK, LiveKit 플러그인 존재 | LiveKit Agents 프레임워크 내장 | 독립형 (직접 통합 필요) |
| **배포 대상** | 클라우드/서버 | 클라우드/서버 (CPU 최적화) | 엣지 디바이스 (Raspberry Pi) |
| **프로덕션 성숙도** | 높음 | 높음 | 실험적/초기 단계 |
| **PyPI 패키지** | `livekit-plugins-namo-turn-detector` | `livekit-plugins-turn-detector` | 없음 |
| **Hugging Face** | [Namo-Turn-Detector-v1-Multilingual](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual) | [livekit/turn-detector](https://huggingface.co/livekit/turn-detector) | [latishab/turnsense](https://huggingface.co/latishab/turnsense) |

---

## 8. 총평

| 프로젝트 | 강점 | 약점 |
|----------|------|------|
| **NAMO** | 가장 빠른 추론 속도(<19ms), 23개 언어 지원, ONNX 최적화 | 자체 SDK 종속성, 평가 데이터셋 비공개 |
| **LiveKit** | Knowledge Distillation으로 높은 품질, 가장 성숙한 생태계, 지속적 개선 | 다국어 모델 추론이 상대적으로 느림(50~160ms) |
| **Turnsense** | 초경량 엣지 배포 가능, 완전 오픈소스, Apache 2.0, 공개 데이터셋 | 영어만 지원, 학습 데이터 2,000개로 적음, 벤치마크 검증 부족 |

### 용도별 추천

- **프로덕션 다국어 환경 (속도 우선)**: NAMO
- **프로덕션 다국어 환경 (생태계/품질 우선)**: LiveKit
- **엣지 디바이스 / 경량 실험**: Turnsense

---

## 9. NAMO 한국어 특화 모델 실측 테스트 결과

### 모델 정보

- **Repo**: `videosdk-live/Namo-Turn-Detector-v1-Korean`
- **아키텍처**: DistilBERT (distilbert-base-multilingual-cased)
- **공식 정확도**: 97.30% (F1: 97.32%)
- **max_length**: 512 토큰

### 양자화 모델 문제 발견

`model_quant.onnx` (양자화 버전)는 **한국어 특화 모델에서 완전히 망가져 있음**.

| 입력 | logits | 결과 |
|------|--------|------|
| "오늘 저녁에 뭐 할 거야?" | [+0.20, -0.21] | CONT 60.1% |
| "그건 좀 생각을 해봐야 할 것 같아." | [+0.20, -0.20] | CONT 59.8% |
| "교남동은 종로구 내에서 상대적으로 보수세가 강한 지역으로 분류된다." | [+0.20, -0.21] | CONT 60.0% |

모든 입력에 대해 logits가 `[~0.2, ~-0.2]`로 거의 동일하게 출력되어 판별 능력이 없음.
**다국어 모델의 `model_quant.onnx`는 정상 작동.**

### 비양자화 모델 (model.onnx) 실측 결과

`model.onnx`로 전환 시 정상 작동. 추론 속도 18~49ms.

#### 스트리밍 시뮬레이션 (단어 단위 누적 입력)

**"오늘 저녁에 뭐 할 거야?"**

| 단계 | 입력 | 판별 | 확신도 |
|------|------|------|--------|
| 1 | 오늘 | CONT | 98.1% |
| 2 | 오늘 저녁에 | CONT | 100.0% |
| 3 | 오늘 저녁에 뭐 | CONT | 100.0% |
| 4 | 오늘 저녁에 뭐 할 | CONT | 100.0% |
| 5 | 오늘 저녁에 뭐 할 거야? | **EOT** | **96.1%** |

**"그건 좀 생각을 해봐야 할 것 같아."**

| 단계 | 입력 | 판별 | 확신도 |
|------|------|------|--------|
| 1 | 그건 | CONT | 100.0% |
| 2 | 그건 좀 | CONT | 100.0% |
| 3 | 그건 좀 생각을 | CONT | 100.0% |
| 4 | 그건 좀 생각을 해봐야 | CONT | 100.0% |
| 5 | 그건 좀 생각을 해봐야 할 | CONT | 100.0% |
| 6 | 그건 좀 생각을 해봐야 할 것 | CONT | 100.0% |
| 7 | 그건 좀 생각을 해봐야 할 것 같아. | CONT | 100.0% |

#### 배치 판별 결과

| 입력 | 판별 | 확신도 | 정답 여부 |
|------|------|--------|----------|
| 오늘 저녁에 뭐 할 거야? | EOT | 96.1% | ✅ |
| 오늘 저녁에 | CONT | 100.0% | ✅ |
| 그건 좀 생각을 해봐야 할 것 같아. | CONT | 100.0% | ❌ (완성 문장) |
| 그건 좀 | CONT | 100.0% | ✅ |
| 교남동은 종로구 내에서 상대적으로 보수세가 강한 지역으로 분류된다. | EOT | 99.4% | ✅ |
| 나는 어제 서울에 갔다. | EOT | 90.4% | ✅ |
| 나는 어제 | CONT | 100.0% | ✅ |
| 혹시 내일 시간 되세요? | EOT | 99.8% | ✅ |
| 혹시 내일 시간 | CONT | 100.0% | ✅ |
| I think the next logical step is to | CONT | 99.9% | ✅ |
| What are you doing tonight? | EOT | 73.2% | ✅ |

### 관찰된 한계

1. **양자화 ONNX 손상**: 한국어 특화 모델의 `model_quant.onnx`는 판별 능력이 없음. 반드시 `model.onnx` 사용 필요
2. **구어체 종결 표현 오판**: "~것 같아.", "~해봐야 할 것 같아." 등 구어체 종결 표현을 미완성으로 오판
3. **영어 문장 낮은 확신도**: 한국어 특화 모델로 영어 입력 시 EOT 확신도가 낮음 (73.2%) — 예상된 동작

---

## 참고 자료

- [NAMO Turn Detector v1 - GitHub](https://github.com/videosdk-live/NAMO-Turn-Detector-v1)
- [NAMO Blog - VideoSDK](https://www.videosdk.live/blog/namo-turn-detection-v1-semantic-turn-detection-for-ai-voice-agents)
- [LiveKit Turn Detector Plugin Docs](https://docs.livekit.io/agents/build/turns/turn-detector/)
- [LiveKit Improved End-of-Turn Model Blog](https://blog.livekit.io/improved-end-of-turn-model-cuts-voice-ai-interruptions-39/)
- [Turnsense - GitHub](https://github.com/latishab/turnsense)
- [Turnsense - Latisha Besariani](https://latishab.com/turnsense)
- [TURNS-2K Dataset - Hugging Face](https://huggingface.co/datasets/latishab/turns-2k)
