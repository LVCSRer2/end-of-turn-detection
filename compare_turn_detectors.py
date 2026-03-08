"""
NAMO vs LiveKit Turn Detector 비교 데모

동일한 입력에 대해 두 모델의 판별 결과를 나란히 비교합니다.

사용법:
  python compare_turn_detectors.py                # 스트리밍 시뮬레이션
  python compare_turn_detectors.py --batch        # 배치 비교
  python compare_turn_detectors.py --interactive  # 대화형 모드
  python compare_turn_detectors.py --lang ko      # 한국어 특화 (NAMO만 적용)
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
# NAMO Turn Detector
# ──────────────────────────────────────────────
class NamoDetector:
    """NAMO Turn Detector v1 ONNX 추론."""

    MULTILINGUAL_REPO = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
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
            onnx_file = "model.onnx"  # 양자화 모델 손상 이슈로 비양자화 사용
            self.max_length = 512
        else:
            repo_id = self.MULTILINGUAL_REPO
            onnx_file = "model_quant.onnx"
            self.max_length = 8192

        model_path = hf_hub_download(repo_id=repo_id, filename=onnx_file)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.session = ort.InferenceSession(model_path)
        self.name = f"NAMO ({language or 'multi'})"

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="np"
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

        return {
            "is_eot": label == 1,
            "confidence": float(probs[1]),  # EOT 확률
            "latency_ms": latency_ms,
        }


# ──────────────────────────────────────────────
# LiveKit Turn Detector
# ──────────────────────────────────────────────
class LiveKitDetector:
    """LiveKit Turn Detector (Qwen2.5-0.5B 기반) ONNX 추론."""

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

    def __init__(self, language: str | None = None):
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

        self.language = language or "en"
        self.threshold = self.THRESHOLDS.get(self.language, 0.011)
        self.name = f"LiveKit ({self.language})"

    def _normalize_text(self, text: str) -> str:
        """LiveKit 다국어 모델의 텍스트 정규화."""
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
        """텍스트를 입력받아 EOU 확률을 반환."""
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

        return {
            "is_eot": eou_prob > self.threshold,
            "confidence": eou_prob,
            "threshold": self.threshold,
            "latency_ms": latency_ms,
        }


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def print_header():
    print("=" * 74)
    print(f"{BOLD}  NAMO vs LiveKit Turn Detector — 비교 데모{RESET}")
    print("=" * 74)


def _bar(value: float, is_eot: bool, width: int = 15) -> str:
    filled = int(value * width)
    color = GREEN if is_eot else YELLOW
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


def print_comparison(step: int, text: str, namo: dict, livekit: dict):
    """두 모델의 결과를 나란히 출력."""
    n_tag = f"{GREEN}EOT{RESET}" if namo["is_eot"] else f"{YELLOW}CONT{RESET}"
    l_tag = f"{GREEN}EOT{RESET}" if livekit["is_eot"] else f"{YELLOW}CONT{RESET}"

    n_conf = namo["confidence"]
    l_conf = livekit["confidence"]
    l_thr = livekit["threshold"]

    agree = namo["is_eot"] == livekit["is_eot"]
    agree_mark = f"{GREEN}O{RESET}" if agree else f"\033[91mX{RESET}"

    print(f"  {DIM}[{step:2d}]{RESET} {text}")
    print(
        f"       {CYAN}NAMO{RESET}    {n_tag}  "
        f"{_bar(n_conf, namo['is_eot'])} "
        f"{n_conf:.1%}  {DIM}({namo['latency_ms']:.1f}ms){RESET}"
    )
    print(
        f"       {MAGENTA}LiveKit{RESET} {l_tag}  "
        f"{_bar(l_conf, livekit['is_eot'])} "
        f"{l_conf:.4f} (thr:{l_thr:.4f})  {DIM}({livekit['latency_ms']:.1f}ms){RESET}"
    )
    print(f"       일치: {agree_mark}")
    print()


# ──────────────────────────────────────────────
# 모드 1: 스트리밍 시뮬레이션
# ──────────────────────────────────────────────
def run_streaming(namo: NamoDetector, livekit: LiveKitDetector):
    print_header()
    print(f"\n{BOLD}[모드] 스트리밍 시뮬레이션{RESET}")
    print(f"{DIM}단어가 하나씩 추가될 때마다 두 모델 동시 판별{RESET}\n")

    sentences = [
        "I think the next logical step is to consider all our options.",
        "What are you doing tonight?",
        "오늘 저녁에 뭐 할 거야?",
        "그건 좀 생각을 해봐야 할 것 같아.",
    ]

    for sentence in sentences:
        print(f"{'─' * 74}")
        print(f"{BOLD}  전체 문장: {sentence}{RESET}\n")

        words = sentence.split()
        for i in range(1, len(words) + 1):
            partial = " ".join(words[:i])
            n_result = namo.predict(partial)
            l_result = livekit.predict(partial)
            print_comparison(i, partial, n_result, l_result)
            time.sleep(0.1)

        print()


# ──────────────────────────────────────────────
# 모드 2: 배치 비교
# ──────────────────────────────────────────────
BATCH_SAMPLES = [
    # (텍스트, 설명)
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


def run_batch(namo: NamoDetector, livekit: LiveKitDetector):
    print_header()
    print(f"\n{BOLD}[모드] 배치 비교{RESET}")
    print(f"{DIM}미완성/완성 문장 쌍을 두 모델로 동시 판별{RESET}\n")

    # 요약 테이블 헤더
    print(f"{'─' * 74}")
    print(
        f"  {'텍스트':<35} {'기대':^6} "
        f"{CYAN}{'NAMO':^12}{RESET} "
        f"{MAGENTA}{'LiveKit':^12}{RESET} {'일치':^4}"
    )
    print(f"{'─' * 74}")

    agree_count = 0
    total = len(BATCH_SAMPLES)

    for text, desc in BATCH_SAMPLES:
        expected_eot = "완성" in desc
        n = namo.predict(text)
        l = livekit.predict(text)

        n_tag = f"{GREEN}EOT{RESET}" if n["is_eot"] else f"{YELLOW}CONT{RESET}"
        l_tag = f"{GREEN}EOT{RESET}" if l["is_eot"] else f"{YELLOW}CONT{RESET}"
        exp_tag = "EOT" if expected_eot else "CONT"

        agree = n["is_eot"] == l["is_eot"]
        if agree:
            agree_count += 1
        agree_mark = f"{GREEN}O{RESET}" if agree else f"\033[91mX{RESET}"

        # 텍스트가 너무 길면 자르기
        display_text = text if len(text) <= 33 else text[:30] + "..."

        print(
            f"  {display_text:<35} {exp_tag:^6} "
            f"{n_tag} {n['confidence']:>5.1%} {DIM}{n['latency_ms']:>4.0f}ms{RESET}  "
            f"{l_tag} {l['confidence']:>7.4f} {DIM}{l['latency_ms']:>4.0f}ms{RESET}  "
            f"{agree_mark}"
        )

    print(f"{'─' * 74}")
    print(f"  모델 간 일치율: {agree_count}/{total} ({agree_count / total:.0%})")
    print()

    # 상세 결과
    print(f"\n{BOLD}[상세 결과]{RESET}\n")
    for text, desc in BATCH_SAMPLES:
        n = namo.predict(text)
        l = livekit.predict(text)
        print(f"  {DIM}{desc}{RESET}")
        print_comparison(0, text, n, l)


# ──────────────────────────────────────────────
# 모드 3: 대화형
# ──────────────────────────────────────────────
def run_interactive(namo: NamoDetector, livekit: LiveKitDetector):
    print_header()
    print(f"\n{BOLD}[모드] 대화형 입력{RESET}")
    print(f"{DIM}텍스트를 입력하면 두 모델의 결과를 비교합니다.{RESET}")
    print(f"{DIM}종료: Ctrl+C 또는 'quit' 입력{RESET}\n")

    step = 0
    while True:
        try:
            text = input(f"{BOLD}입력> {RESET}").strip()
            if not text or text.lower() in ("quit", "exit", "q"):
                print("\n종료합니다.")
                break
            step += 1
            n = namo.predict(text)
            l = livekit.predict(text)
            print_comparison(step, text, n, l)
        except (KeyboardInterrupt, EOFError):
            print("\n\n종료합니다.")
            break


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NAMO vs LiveKit Turn Detector 비교 데모"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드")
    parser.add_argument("--batch", "-b", action="store_true", help="배치 비교 모드")
    parser.add_argument("--lang", "-l", type=str, default=None,
                        help="언어 코드 (en, ko, ja, zh, de, fr, es, ru, tr, hi)")
    args = parser.parse_args()

    print(f"\n{DIM}모델 로딩 중...{RESET}")
    namo = NamoDetector(language=args.lang)
    print(f"  {GREEN}✓{RESET} {namo.name} 로드 완료")

    livekit = LiveKitDetector(language=args.lang)
    print(f"  {GREEN}✓{RESET} {livekit.name} 로드 완료")
    print()

    if args.interactive:
        run_interactive(namo, livekit)
    elif args.batch:
        run_batch(namo, livekit)
    else:
        run_streaming(namo, livekit)


if __name__ == "__main__":
    main()
