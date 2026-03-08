# End-of-Turn Detection

음성 AI 에이전트를 위한 **Semantic Turn Detection** 모델 비교 및 데모 프로젝트입니다.

사용자가 발화를 마쳤는지(End-of-Turn) 아직 말하는 중인지(Continuation)를 텍스트 기반으로 판별합니다.
기존 VAD(Voice Activity Detection)의 한계를 넘어, 자연어 이해(NLU) 기반으로 턴 경계를 예측합니다.

## 비교 대상 모델

| 모델 | 아키텍처 | 크기 | 추론 속도 | 언어 |
|------|---------|------|----------|------|
| **NAMO Turn Detector v1** | mmBERT / DistilBERT + ONNX | ~135-295MB | <19-29ms | 23개 |
| **LiveKit Turn Detector** | Qwen2.5-0.5B + ONNX INT8 | ~66-281MB | 15-160ms | 14개 |
| **Turnsense** | SmolLM2-135M + LoRA | ~135M params | - | 영어 |

상세 비교 분석은 [turn_detector_comparison.md](turn_detector_comparison.md) 참조.

## 설치

```bash
pip install onnxruntime transformers huggingface_hub numpy
```

모델은 첫 실행 시 Hugging Face에서 자동 다운로드됩니다.

## 데모 스크립트

### 1. NAMO 전용 데모 (`streaming_turn_detection_demo.py`)

NAMO Turn Detector v1으로 스트리밍 텍스트 입력의 turn detection을 수행합니다.

```bash
# 스트리밍 시뮬레이션 (단어 단위 누적 입력)
python streaming_turn_detection_demo.py

# 배치 판별 (미완성/완성 문장 비교)
python streaming_turn_detection_demo.py --batch

# 대화형 모드 (직접 텍스트 입력)
python streaming_turn_detection_demo.py --interactive

# 한국어 특화 모델 사용
python streaming_turn_detection_demo.py --lang ko
```

지원 언어: `en`, `ko`, `ja`, `zh`, `de`, `fr`, `es`, `ru`, `tr`, `hi`

> **참고**: 한국어 특화 모델의 양자화 ONNX(`model_quant.onnx`)에 결함이 있어 비양자화 모델(`model.onnx`)을 자동으로 사용합니다.

### 2. LiveKit 전용 데모 (`livekit_turn_detection_demo.py`)

LiveKit Turn Detector (Qwen2.5-0.5B 기반)로 turn detection을 수행합니다.
`<|im_end|>` 토큰 출현 확률과 언어별 임계값을 기반으로 EOU를 판별합니다.

```bash
# 스트리밍 시뮬레이션
python livekit_turn_detection_demo.py

# 배치 판별
python livekit_turn_detection_demo.py --batch

# 대화형 모드 (history on/off로 대화 컨텍스트 토글)
python livekit_turn_detection_demo.py --interactive

# 한국어 임계값 적용
python livekit_turn_detection_demo.py --lang ko
```

대화형 모드에서 `history on`을 입력하면 이전 대화 컨텍스트를 포함하여 판별합니다.

지원 언어: `en`, `ko`, `ja`, `zh`, `de`, `fr`, `es`, `ru`, `tr`, `hi`, `id`, `nl`, `pt`, `it`

### 3. NAMO vs LiveKit 비교 데모 (`compare_turn_detectors.py`)

동일한 입력에 대해 두 모델의 판별 결과를 나란히 비교합니다.

```bash
# 스트리밍 비교
python compare_turn_detectors.py

# 배치 비교 (정확도 및 일치율 표시)
python compare_turn_detectors.py --batch

# 대화형 비교
python compare_turn_detectors.py --interactive

# 한국어 설정
python compare_turn_detectors.py --lang ko
```

## 실측 결과 요약 (한국어)

### NAMO (한국어 특화 모델, `model.onnx`)

| 입력 | 판별 | 확신도 | 속도 |
|------|------|--------|------|
| 오늘 저녁에 | CONT | 100.0% | 23ms |
| 오늘 저녁에 뭐 할 거야? | **EOT** | 96.1% | 31ms |
| 그건 좀 생각을 해봐야 할 것 같아. | CONT | 100.0% | 43ms |
| 혹시 내일 시간 되세요? | **EOT** | 99.8% | 41ms |

### LiveKit (한국어 임계값 0.0156)

| 입력 | 판별 | EOU 확률 | 속도 |
|------|------|---------|------|
| 오늘 저녁에 | CONT | 0.0006 | 75ms |
| 오늘 저녁에 뭐 할 거야? | **EOT** | 0.9715 | 112ms |
| 그건 좀 생각을 해봐야 할 것 같아. | **EOT** | 0.0640 | 99ms |
| 혹시 내일 시간 되세요? | **EOT** | 0.8572 | 88ms |

### 모델 간 비교

- 배치 15개 샘플 기준 **80% 일치율**
- NAMO가 약 2-3배 빠름 (20-45ms vs 55-135ms)
- LiveKit이 "~것 같아." 같은 구어체 종결 표현에서 더 정확
- NAMO가 "나는 어제 서울에 갔다." 같은 평서문 종결에서 더 정확

## 파일 구조

```
.
├── README.md                          # 프로젝트 설명
├── requirements.txt                   # 의존성
├── streaming_turn_detection_demo.py   # NAMO 전용 데모
├── livekit_turn_detection_demo.py     # LiveKit 전용 데모
├── compare_turn_detectors.py          # NAMO vs LiveKit 비교 데모
└── turn_detector_comparison.md        # 모델 비교 분석 문서
```

## 참고 자료

- [NAMO Turn Detector v1 - GitHub](https://github.com/videosdk-live/NAMO-Turn-Detector-v1)
- [NAMO Turn Detector v1 - Hugging Face](https://huggingface.co/videosdk-live/Namo-Turn-Detector-v1-Multilingual)
- [LiveKit Turn Detector Plugin](https://docs.livekit.io/agents/build/turns/turn-detector/)
- [LiveKit Turn Detector - Hugging Face](https://huggingface.co/livekit/turn-detector)
- [Turnsense - GitHub](https://github.com/latishab/turnsense)
