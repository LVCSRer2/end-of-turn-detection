"""
NAMO Turn Detector v1 - Streaming Text Turn Detection Demo

STT 출력을 시뮬레이션하여 단어가 하나씩 들어올 때마다
실시간으로 End-of-Turn 여부를 판별하는 데모입니다.

사용법:
  python streaming_turn_detection_demo.py          # 기본 데모 (샘플 문장)
  python streaming_turn_detection_demo.py --interactive  # 대화형 모드
  python streaming_turn_detection_demo.py --lang ko      # 한국어 특화 모델
"""

import argparse
import sys
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


# ──────────────────────────────────────────────
# NAMO Turn Detector wrapper
# ──────────────────────────────────────────────
class NamoTurnDetector:
    """NAMO Turn Detector v1 ONNX 추론 래퍼."""

    MULTILINGUAL_REPO = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"

    # 언어별 특화 모델 매핑
    LANG_REPOS = {
        "en": "videosdk-live/Namo-Turn-Detector-v1-English",
        "ko": "videosdk-live/Namo-Turn-Detector-v1-Korean",
        "ja": "videosdk-live/Namo-Turn-Detector-v1-Japanese",
        "zh": "videosdk-live/Namo-Turn-Detector-v1-Chinese",
        "de": "videosdk-live/Namo-Turn-Detector-v1-German",
        "fr": "videosdk-live/Namo-Turn-Detector-v1-French",
        "es": "videosdk-live/Namo-Turn-Detector-v1-Spanish",
        "ru": "videosdk-live/Namo-Turn-Detector-v1-Russian",
        "tr": "videosdk-live/Namo-Turn-Detector-v1-Turkish",
        "hi": "videosdk-live/Namo-Turn-Detector-v1-Hindi",
    }

    def __init__(self, language: str | None = None):
        if language and language in self.LANG_REPOS:
            repo_id = self.LANG_REPOS[language]
            print(f"[모델 로드] 언어 특화 모델: {repo_id}")
        else:
            repo_id = self.MULTILINGUAL_REPO
            print(f"[모델 로드] 다국어 모델: {repo_id}")

        # 언어별 특화 모델은 양자화 ONNX에 문제가 있으므로 비양자화 모델 사용
        if language and language in self.LANG_REPOS:
            onnx_file = "model.onnx"
            self.max_length = 512  # DistilBERT 기반 특화 모델
        else:
            onnx_file = "model_quant.onnx"
            self.max_length = 8192  # mmBERT 기반 다국어 모델

        model_path = hf_hub_download(repo_id=repo_id, filename=onnx_file)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.session = ort.InferenceSession(model_path)
        print(f"[모델 로드] 완료 ({onnx_file})\n")

    def predict(self, text: str) -> dict:
        """텍스트를 입력받아 turn 판별 결과를 반환."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        t0 = time.perf_counter_ns()
        outputs = self.session.run(None, feed)
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        logits = outputs[0][0]
        probs = _softmax(logits)
        label = int(np.argmax(probs))
        confidence = float(np.max(probs))

        return {
            "label": label,  # 0: Not EoT, 1: EoT
            "is_end_of_turn": label == 1,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ──────────────────────────────────────────────
# 시각화 헬퍼
# ──────────────────────────────────────────────
LABEL_MAP = {True: "EOT  (발화 완료)", False: "CONT (발화 중..)"}
COLOR = {True: "\033[92m", False: "\033[93m"}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def print_header():
    print("=" * 70)
    print(f"{BOLD}  NAMO Turn Detector v1 — Streaming Turn Detection Demo{RESET}")
    print("=" * 70)


def print_prediction(step: int, text: str, result: dict):
    is_eot = result["is_end_of_turn"]
    conf = result["confidence"]
    ms = result["latency_ms"]
    tag = f"{COLOR[is_eot]}{LABEL_MAP[is_eot]}{RESET}"
    bar = _confidence_bar(conf, is_eot)

    print(f"  [{step:2d}] {DIM}{text}{RESET}")
    print(f"       {tag}  {bar} {conf:.1%}  {DIM}({ms:.1f}ms){RESET}")
    print()


def _confidence_bar(conf: float, is_eot: bool, width: int = 20) -> str:
    filled = int(conf * width)
    color = COLOR[is_eot]
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


# ──────────────────────────────────────────────
# 데모 모드 1: 스트리밍 시뮬레이션
# ──────────────────────────────────────────────
SAMPLE_SENTENCES = [
    # (문장, 언어 태그)
    ("I think the next logical step is to", "en"),
    ("I think the next logical step is to consider all our options.", "en"),
    ("What are you doing tonight?", "en"),
    ("오늘 저녁에", "ko"),
    ("오늘 저녁에 뭐 할 거야?", "ko"),
    ("그건 좀 생각을 해봐야", "ko"),
    ("그건 좀 생각을 해봐야 할 것 같아.", "ko"),
    ("The Revenue Act of 1862 adopted rates that increased with", "en"),
    ("The Revenue Act of 1862 adopted rates that increased with income.", "en"),
    ("今日の天気は", "ja"),
    ("今日の天気はどうですか？", "ja"),
]


def run_streaming_demo(detector: NamoTurnDetector):
    """단어를 하나씩 추가하며 실시간 판별하는 스트리밍 시뮬레이션."""
    print_header()
    print(f"\n{BOLD}[모드] 스트리밍 시뮬레이션{RESET}")
    print(f"{DIM}단어가 하나씩 추가될 때마다 turn detection 수행{RESET}\n")

    sentences = [
        "I think the next logical step is to consider all our options.",
        "What are you doing tonight?",
        "오늘 저녁에 뭐 할 거야?",
        "그건 좀 생각을 해봐야 할 것 같아.",
    ]

    for sentence in sentences:
        print(f"{'─' * 60}")
        print(f"{BOLD}  전체 문장: {sentence}{RESET}\n")

        words = sentence.split()
        for i in range(1, len(words) + 1):
            partial = " ".join(words[:i])
            result = detector.predict(partial)
            print_prediction(i, partial, result)
            time.sleep(0.15)  # 스트리밍 효과

        print()


# ──────────────────────────────────────────────
# 데모 모드 2: 배치 판별
# ──────────────────────────────────────────────
def run_batch_demo(detector: NamoTurnDetector):
    """미리 준비된 샘플 문장들을 일괄 판별."""
    print_header()
    print(f"\n{BOLD}[모드] 배치 판별{RESET}")
    print(f"{DIM}미완성 문장 vs 완성 문장 비교{RESET}\n")
    print(f"{'─' * 60}")

    for i, (text, lang) in enumerate(SAMPLE_SENTENCES, 1):
        result = detector.predict(text)
        print_prediction(i, f"[{lang}] {text}", result)


# ──────────────────────────────────────────────
# 데모 모드 3: 대화형 입력
# ──────────────────────────────────────────────
def run_interactive_demo(detector: NamoTurnDetector):
    """사용자가 직접 텍스트를 입력하여 판별."""
    print_header()
    print(f"\n{BOLD}[모드] 대화형 입력{RESET}")
    print(f"{DIM}텍스트를 입력하면 turn detection 결과를 표시합니다.{RESET}")
    print(f"{DIM}종료: Ctrl+C 또는 'quit' 입력{RESET}\n")

    step = 0
    while True:
        try:
            text = input(f"{BOLD}입력> {RESET}").strip()
            if not text or text.lower() in ("quit", "exit", "q"):
                print("\n종료합니다.")
                break
            step += 1
            result = detector.predict(text)
            print_prediction(step, text, result)
        except (KeyboardInterrupt, EOFError):
            print("\n\n종료합니다.")
            break


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NAMO Turn Detector v1 — Streaming Turn Detection Demo"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드로 실행",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="배치 판별 모드로 실행",
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default=None,
        help="언어 특화 모델 사용 (en, ko, ja, zh, de, fr, es, ru, tr, hi)",
    )
    args = parser.parse_args()

    detector = NamoTurnDetector(language=args.lang)

    if args.interactive:
        run_interactive_demo(detector)
    elif args.batch:
        run_batch_demo(detector)
    else:
        run_streaming_demo(detector)


if __name__ == "__main__":
    main()
