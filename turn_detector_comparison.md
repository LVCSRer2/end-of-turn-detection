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

## 4. 아키텍처 접근 방식

### NAMO

- BERT 계열 인코더 모델 기반
- ONNX 양자화로 최적화 (2.19x 속도 향상, 정확도 손실 <0.2%)
- 다국어 모델(mmBERT)과 언어별 특화 모델(DistilBERT) 이원화 전략
- 양자화 전후: 61.3ms → 28.0ms (다국어), 38ms → 14.9ms (단일)

### LiveKit

- Knowledge Distillation 활용
- Qwen2.5-7B 교사 모델 → Qwen2.5-0.5B 학생 모델로 증류
- VAD와 결합하여 사용 (대화 맥락 4턴 슬라이딩 윈도우)
- v0.3 → v0.4.1에서 false-positive 중단 39% 감소 달성
- 전화번호, 주소, 이메일 등 구조화 데이터 처리 강화

### Turnsense

- SmolLM2-135M에 LoRA 파인튜닝
- 엣지 디바이스(Raspberry Pi) 최적화가 핵심 목표
- 가장 경량한 아키텍처

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
