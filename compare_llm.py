# benchmark_tps.py
import time
from typing import Dict, Any, List
import requests
from dataclasses import dataclass

from transformers import AutoTokenizer

# ===== 你要改的地方 =====
MODEL_NAME_OR_PATH = "openai/gpt-oss-20b" # 或者本地路徑: "/path/to/gptoss"

BACKENDS = [
    {
        "name": "vllm-gptoss",
        "base_url": "http://127.0.0.1:8000",  # vLLM OpenAI server
        "model": "openai/gpt-oss-20b",
    },
    {
        "name": "trt-gptoss",
        "base_url": "http://127.0.0.1:9000",  # TRT-LLM OpenAI 相容 endpoint
        "model": "openai/gpt-oss-20b",
    },
    # {
    #     "name": "hf-gptoss",
    #     "base_url": "http://127.0.0.1:7000",  # 你自己包的 HF service
    #     "model": "gptoss",
    # },
]
# =========================

# 測試用的 prompt，可以依你的 workload 調整 / 增加
TEST_PROMPTS = [
    "請用繁體中文詳細說明 Transformer 架構的原理，大約 300 字。",
    "解釋什麼是知識蒸餾及其在大型語言模型中的應用，限制在 500 字以內。",
    "請比較 RNN、LSTM、Transformer 在長序列建模上的差異。",
]

MAX_NEW_TOKENS = 512   # 每次生成的上限，對速度很重要
WARMUP_ROUNDS = 2      # 每個 backend 先跑幾次熱機，不記錄
REPEAT_PER_PROMPT = 3  # 每個 prompt 實測幾次

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)


def count_tokens(text: str) -> int:
    # 簡單 token counting，用於沒有回傳 usage 的情況
    return len(tokenizer.encode(text, add_special_tokens=False))


@dataclass
class RunResult:
    backend: str
    prompt_len: int
    output_len: int
    elapsed: float
    gen_tokens_per_s: float
    total_tokens_per_s: float


class OpenAICompatBackend:
    def __init__(self, name: str, base_url: str, model: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """
        非 streaming，單純算「整個 completion 使用時間 / token 數」
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            # 若有 auth 自行加：
            # "Authorization": f"Bearer {API_KEY}",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]

        # 優先用服務端回傳的 usage
        usage = data.get("usage", None)
        if usage is not None:
            in_tokens = usage.get("prompt_tokens", 0)
            out_tokens = usage.get("completion_tokens", 0)
        else:
            # 若沒有 usage，就自己用 tokenizer 算
            in_tokens = count_tokens(prompt)
            out_tokens = count_tokens(text)

        return {
            "text": text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "elapsed": elapsed,
        }


def benchmark_one_backend(backend_cfg: Dict[str, str]) -> List[RunResult]:
    backend = OpenAICompatBackend(
        name=backend_cfg["name"],
        base_url=backend_cfg["base_url"],
        model=backend_cfg["model"],
    )
    results: List[RunResult] = []

    # 熱機（不記錄結果，避免 cold start 影響）
    for _ in range(WARMUP_ROUNDS):
        _ = backend.generate(TEST_PROMPTS[0], MAX_NEW_TOKENS)

    # 正式測試
    for prompt in TEST_PROMPTS:
        for _ in range(REPEAT_PER_PROMPT):
            out = backend.generate(prompt, MAX_NEW_TOKENS)
            gen_tps = out["output_tokens"] / out["elapsed"] if out["elapsed"] > 0 else 0.0
            total_tps = (out["input_tokens"] + out["output_tokens"]) / out["elapsed"] if out["elapsed"] > 0 else 0.0

            result = RunResult(
                backend=backend.name,
                prompt_len=out["input_tokens"],
                output_len=out["output_tokens"],
                elapsed=out["elapsed"],
                gen_tokens_per_s=gen_tps,
                total_tokens_per_s=total_tps,
            )
            results.append(result)
            print(
                f"[{backend.name}] "
                f"prompt_tokens={result.prompt_len}, "
                f"output_tokens={result.output_len}, "
                f"time={result.elapsed:.3f}s, "
                f"gen_tps={result.gen_tokens_per_s:.2f}, "
                f"total_tps={result.total_tokens_per_s:.2f}"
            )

    return results


def summarize(all_results: List[RunResult]):
    by_backend: Dict[str, List[RunResult]] = {}
    for r in all_results:
        by_backend.setdefault(r.backend, []).append(r)

    print("\n===== Summary (average) =====")
    for backend, rs in by_backend.items():
        n = len(rs)
        avg_gen_tps = sum(r.gen_tokens_per_s for r in rs) / n
        avg_total_tps = sum(r.total_tokens_per_s for r in rs) / n
        avg_out_len = sum(r.output_len for r in rs) / n
        print(
            f"{backend}: "
            f"gen_tokens/s={avg_gen_tps:.2f}, "
            f"total_tokens/s={avg_total_tps:.2f}, "
            f"avg_output_tokens={avg_out_len:.1f} (n={n})"
        )


if __name__ == "__main__":
    all_results: List[RunResult] = []
    for cfg in BACKENDS:
        print(f"\n===== Benchmark {cfg['name']} =====")
        res = benchmark_one_backend(cfg)
        all_results.extend(res)

    summarize(all_results)
