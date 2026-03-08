"""
LiveKit Turn Detector - Streaming Text Turn Detection Demo

STT 출력을 시뮬레이션하여 단어가 하나씩 들어올 때마다
실시간으로 End-of-Turn 여부를 판별하는 데모입니다.

모델: Qwen2.5-0.5B 기반, Knowledge Distillation, ONNX INT8 양자화
방식: 대화 컨텍스트를 Qwen 채팅 템플릿으로 포맷 후 <|im_end|> 토큰 출현 확률로 EOU 판별

사용법:
  python livekit_turn_detection_demo.py                # 기본 데모 (샘플 문장)
  python livekit_turn_detection_demo.py --interactive  # 대화형 모드
  python livekit_turn_detection_demo.py --batch        # 배치 판별
  python livekit_turn_detection_demo.py --lang ko      # 한국어 임계값 적용
"""

import argparse
import re
import time
import unicodedata

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


# ──────────────────────────────────────────────
# LiveKit Turn Detector wrapper
# ──────────────────────────────────────────────
class LiveKitTurnDetector:
    """LiveKit Turn Detector (Qwen2.5-0.5B 기반) ONNX 추론 래퍼."""

    HG_MODEL = "livekit/turn-detector"
    REVISION = "v0.4.1-intl"
    MAX_TOKENS = 128

    # 언어별 EOU 임계값 (languages.json 기반)
    THRESHOLDS = {
        "zh": 0.0066, "de": 0.0062, "nl": 0.0077, "en": 0.011,
        "pt": 0.0069, "es": 0.0058, "fr": 0.0078, "it": 0.0037,
        "ja": 0.0096, "ko": 0.0156, "ru": 0.0032, "tr": 0.0045,
        "id": 0.0132, "hi": 0.0398,
    }

    LANG_NAMES = {
        "zh": "Chinese", "de": "German", "nl": "Dutch", "en": "English",
        "pt": "Portuguese", "es": "Spanish", "fr": "French", "it": "Italian",
        "ja": "Japanese", "ko": "Korean", "ru": "Russian", "tr": "Turkish",
        "id": "Indonesian", "hi": "Hindi",
    }

    def __init__(self, language: str | None = None):
        self.language = language or "en"
        self.threshold = self.THRESHOLDS.get(self.language, 0.011)
        lang_name = self.LANG_NAMES.get(self.language, self.language)

        print(f"[모델 로드] LiveKit Turn Detector ({self.HG_MODEL} @ {self.REVISION})")
        print(f"[설정] 언어: {lang_name}, 임계값: {self.threshold}")

        onnx_path = hf_hub_download(
            self.HG_MODEL, "model_q8.onnx",
            subfolder="onnx", revision=self.REVISION,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HG_MODEL, revision=self.REVISION, truncation_side="left",
        )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"], sess_options=sess_opts,
        )
        print("[모델 로드] 완료\n")

    def _normalize_text(self, text: str) -> str:
        """LiveKit 다국어 모델의 텍스트 정규화 (NFKC + 소문자 + 구두점 제거)."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text.lower())
        text = "".join(
            ch for ch in text
            if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
        )
        return re.sub(r"\s+", " ", text).strip()

    def _format_chat_ctx(self, messages: list[dict]) -> str:
        """대화 컨텍스트를 Qwen 채팅 템플릿으로 포맷."""
        merged = []
        last = None
        for msg in messages:
            if not msg["content"]:
                continue
            content = self._normalize_text(msg["content"])
            if last and last["role"] == msg["role"]:
                last["content"] += f" {content}"
            else:
                new_msg = {"role": msg["role"], "content": content}
                merged.append(new_msg)
                last = new_msg

        text = self.tokenizer.apply_chat_template(
            merged, add_generation_prompt=False,
            add_special_tokens=False, tokenize=False,
        )
        # 마지막 <|im_end|> 제거 — 모델이 이 토큰의 출현 확률을 예측
        ix = text.rfind("<|im_end|>")
        return text[:ix]

    def predict(self, text: str, chat_history: list[dict] | None = None) -> dict:
        """
        텍스트를 입력받아 EOU 확률을 반환.
        chat_history: 이전 대화 턴 리스트 (선택사항)
        """
        if chat_history is None:
            messages = [{"role": "user", "content": text}]
        else:
            messages = chat_history + [{"role": "user", "content": text}]

        formatted = self._format_chat_ctx(messages)
        inputs = self.tokenizer(
            formatted, add_special_tokens=False, return_tensors="np",
            max_length=self.MAX_TOKENS, truncation=True,
        )

        t0 = time.perf_counter_ns()
        outputs = self.session.run(
            None, {"input_ids": inputs["input_ids"].astype(np.int64)}
        )
        latency_ms = (time.perf_counter_ns() - t0) / 1e6

        eou_prob = float(outputs[0].flatten()[-1])
        is_eot = eou_prob > self.threshold

        return {
            "is_end_of_turn": is_eot,
            "eou_probability": eou_prob,
            "threshold": self.threshold,
            "latency_ms": latency_ms,
        }


# ──────────────────────────────────────────────
# 시각화 헬퍼
# ──────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"

LABEL_MAP = {True: "EOT  (발화 완료)", False: "CONT (발화 중..)"}
COLOR = {True: GREEN, False: YELLOW}


def print_header(lang: str, threshold: float):
    print("=" * 70)
    print(f"{BOLD}  LiveKit Turn Detector — Streaming Turn Detection Demo{RESET}")
    print(f"  언어: {lang}  |  임계값: {threshold}  |  모델: Qwen2.5-0.5B (INT8)")
    print("=" * 70)


def _eou_bar(prob: float, threshold: float, width: int = 25) -> str:
    """EOU 확률을 바 형태로 시각화. 임계값 위치도 표시."""
    is_eot = prob > threshold
    filled = max(1, int(prob * width)) if prob > 0.005 else 0
    thr_pos = max(0, min(width - 1, int(threshold * width)))
    color = GREEN if is_eot else YELLOW

    bar_chars = []
    for i in range(width):
        if i < filled:
            bar_chars.append(f"{color}█{RESET}")
        elif i == thr_pos:
            bar_chars.append(f"\033[91m│{RESET}")  # 임계값 표시 (빨간색)
        else:
            bar_chars.append(f"{DIM}░{RESET}")

    return "".join(bar_chars)


def print_prediction(step: int, text: str, result: dict):
    is_eot = result["is_end_of_turn"]
    prob = result["eou_probability"]
    thr = result["threshold"]
    ms = result["latency_ms"]

    tag = f"{COLOR[is_eot]}{LABEL_MAP[is_eot]}{RESET}"
    bar = _eou_bar(prob, thr)

    print(f"  [{step:2d}] {DIM}{text}{RESET}")
    print(
        f"       {tag}  {bar}  "
        f"prob={BOLD}{prob:.4f}{RESET}  "
        f"{DIM}thr={thr:.4f}  ({ms:.1f}ms){RESET}"
    )
    print()


# ──────────────────────────────────────────────
# 데모 모드 1: 스트리밍 시뮬레이션
# ──────────────────────────────────────────────
def run_streaming_demo(detector: LiveKitTurnDetector):
    print_header(detector.language, detector.threshold)
    print(f"\n{BOLD}[모드] 스트리밍 시뮬레이션{RESET}")
    print(f"{DIM}단어가 하나씩 추가될 때마다 turn detection 수행{RESET}\n")

    sentences = [
        "I think the next logical step is to consider all our options.",
        "What are you doing tonight?",
        "오늘 저녁에 뭐 할 거야?",
        "그건 좀 생각을 해봐야 할 것 같아.",
    ]

    for sentence in sentences:
        print(f"{'─' * 70}")
        print(f"{BOLD}  전체 문장: {sentence}{RESET}\n")

        words = sentence.split()
        for i in range(1, len(words) + 1):
            partial = " ".join(words[:i])
            result = detector.predict(partial)
            print_prediction(i, partial, result)
            time.sleep(0.1)

        print()


# ──────────────────────────────────────────────
# 데모 모드 2: 배치 판별
# ──────────────────────────────────────────────
SAMPLE_SENTENCES = [
    ("I think the next logical step is to", "미완성 (영어)"),
    ("I think the next logical step is to consider all our options.", "완성 (영어)"),
    ("What are you doing tonight?", "완성 질문 (영어)"),
    ("오늘 저녁에", "미완성 (한국어)"),
    ("오늘 저녁에 뭐 할 거야?", "완성 질문 (한국어)"),
    ("그건 좀 생각을", "미완성 (한국어)"),
    ("그건 좀 생각을 해봐야 할 것 같아.", "완성 (한국어)"),
    ("혹시 내일 시간", "미완성 (한국어)"),
    ("혹시 내일 시간 되세요?", "완성 질문 (한국어)"),
    ("나는 어제", "미완성 (한국어)"),
    ("나는 어제 서울에 갔다.", "완성 (한국어)"),
    ("The Revenue Act of 1862 adopted rates that increased with", "미완성 (영어)"),
    ("The Revenue Act of 1862 adopted rates that increased with income.", "완성 (영어)"),
    ("今日の天気は", "미완성 (일본어)"),
    ("今日の天気はどうですか？", "완성 질문 (일본어)"),
]


def run_batch_demo(detector: LiveKitTurnDetector):
    print_header(detector.language, detector.threshold)
    print(f"\n{BOLD}[모드] 배치 판별{RESET}")
    print(f"{DIM}미완성 문장 vs 완성 문장 비교{RESET}\n")

    # 요약 테이블
    print(f"{'─' * 70}")
    correct = 0
    total = len(SAMPLE_SENTENCES)

    for text, desc in SAMPLE_SENTENCES:
        expected_eot = "완성" in desc
        result = detector.predict(text)
        is_eot = result["is_end_of_turn"]
        prob = result["eou_probability"]
        ms = result["latency_ms"]

        tag = f"{GREEN}EOT{RESET}" if is_eot else f"{YELLOW}CONT{RESET}"
        exp = "EOT" if expected_eot else "CONT"

        match = is_eot == expected_eot
        if match:
            correct += 1
        mark = f"{GREEN}✓{RESET}" if match else f"\033[91m✗{RESET}"

        display = text if len(text) <= 40 else text[:37] + "..."
        print(
            f"  {mark} {tag} {prob:>7.4f}  "
            f"{DIM}{ms:>5.1f}ms{RESET}  "
            f"{display}"
        )

    print(f"{'─' * 70}")
    print(f"  정확도: {correct}/{total} ({correct / total:.0%})")
    print()

    # 상세 결과
    print(f"{BOLD}[상세 결과]{RESET}\n")
    for i, (text, desc) in enumerate(SAMPLE_SENTENCES, 1):
        result = detector.predict(text)
        print(f"  {DIM}{desc}{RESET}")
        print_prediction(i, text, result)


# ──────────────────────────────────────────────
# 데모 모드 3: 대화형 입력
# ──────────────────────────────────────────────
def run_interactive_demo(detector: LiveKitTurnDetector):
    print_header(detector.language, detector.threshold)
    print(f"\n{BOLD}[모드] 대화형 입력{RESET}")
    print(f"{DIM}텍스트를 입력하면 turn detection 결과를 표시합니다.{RESET}")
    print(f"{DIM}'history on/off' — 대화 히스토리 모드 토글{RESET}")
    print(f"{DIM}종료: Ctrl+C 또는 'quit' 입력{RESET}\n")

    step = 0
    history_mode = False
    chat_history: list[dict] = []

    while True:
        try:
            prompt_prefix = f"{CYAN}[history]{RESET} " if history_mode else ""
            text = input(f"{prompt_prefix}{BOLD}입력> {RESET}").strip()

            if not text or text.lower() in ("quit", "exit", "q"):
                print("\n종료합니다.")
                break

            if text.lower() == "history on":
                history_mode = True
                chat_history = []
                print(f"  {GREEN}대화 히스토리 모드 활성화{RESET} — 이전 턴이 컨텍스트로 포함됩니다.\n")
                continue
            elif text.lower() == "history off":
                history_mode = False
                chat_history = []
                print(f"  {YELLOW}대화 히스토리 모드 비활성화{RESET}\n")
                continue
            elif text.lower() == "history clear":
                chat_history = []
                print(f"  히스토리 초기화 완료.\n")
                continue

            step += 1
            if history_mode:
                result = detector.predict(text, chat_history=chat_history)
                # 히스토리에 추가 (EOT일 때만 턴 완료로 기록)
                if result["is_end_of_turn"]:
                    chat_history.append({"role": "user", "content": text})
                    chat_history.append(
                        {"role": "assistant", "content": "(assistant response)"}
                    )
                    # 최근 6턴만 유지
                    if len(chat_history) > 12:
                        chat_history = chat_history[-12:]
            else:
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
        description="LiveKit Turn Detector — Streaming Turn Detection Demo"
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
        help="언어 설정 — 임계값 자동 적용 (en, ko, ja, zh, de, fr, es, ru, tr, hi, id, nl, pt, it)",
    )
    args = parser.parse_args()

    detector = LiveKitTurnDetector(language=args.lang)

    if args.interactive:
        run_interactive_demo(detector)
    elif args.batch:
        run_batch_demo(detector)
    else:
        run_streaming_demo(detector)


if __name__ == "__main__":
    main()
