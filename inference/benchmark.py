import time
import json
import os
import ollama
from datetime import datetime
from config import MODEL


BENCHMARK_PROMPTS = [
    "What is machine learning? Answer in 2 sentences.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is a neural network? Give a simple analogy.",
    "What is RAG in AI? Keep it brief.",
    "What are embeddings and why are they useful?"
]

REPORTS_DIR = "inference/reports"


def benchmark_ollama(
    prompts: list = None,
    model: str = MODEL
) -> dict:
    """
    Benchmark the Ollama model (llama3.2).
    Measures: tokens per second, latency, response quality.
    """
    prompts = prompts or BENCHMARK_PROMPTS
    print(f"\n{'='*55}")
    print(f"  BENCHMARKING OLLAMA: {model}")
    print(f"{'='*55}\n")

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:50]}...")

        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.3}
        )
        elapsed_ms = round((time.time() - start) * 1000, 2)
        text = response["message"]["content"]
        word_count = len(text.split())
        estimated_tokens = word_count * 1.3  # rough estimate
        tps = round(estimated_tokens / (elapsed_ms / 1000), 1)

        result = {
            "prompt": prompt[:60],
            "response": text[:200],
            "time_ms": elapsed_ms,
            "word_count": word_count,
            "estimated_tps": tps
        }
        results.append(result)
        print(f"   Time: {elapsed_ms}ms | Words: {word_count} | ~{tps} tok/s")

    avg_time = sum(r["time_ms"] for r in results) / len(results)
    avg_tps = sum(r["estimated_tps"] for r in results) / len(results)

    summary = {
        "engine": "ollama",
        "model": model,
        "total_prompts": len(results),
        "avg_time_ms": round(avg_time, 2),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "timestamp": datetime.now().isoformat()
    }

    print(f"\n  Average time:   {avg_time:.0f}ms")
    print(f"  Average tok/s:  {avg_tps:.1f}")

    return {"summary": summary, "results": results}


def benchmark_gguf(
    model_path: str,
    prompts: list = None,
    max_tokens: int = 100
) -> dict:
    """
    Benchmark a GGUF quantized model.
    """
    from inference.quantized_model import QuantizedModel

    prompts = prompts or BENCHMARK_PROMPTS
    print(f"\n{'='*55}")
    print(f"  BENCHMARKING GGUF: {os.path.basename(model_path)}")
    print(f"{'='*55}\n")

    model = QuantizedModel(model_path=model_path)
    info = model.get_info()
    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:50]}...")

        result = model.chat(
            message=prompt,
            max_tokens=max_tokens,
            temperature=0.3
        )

        record = {
            "prompt": prompt[:60],
            "response": result["text"][:200],
            "time_ms": result["time_ms"],
            "tokens_generated": result["tokens_generated"],
            "tokens_per_second": result["tokens_per_second"]
        }
        results.append(record)
        print(f"   Time: {result['time_ms']}ms | "
              f"Tokens: {result['tokens_generated']} | "
              f"TPS: {result['tokens_per_second']}")

    avg_time = sum(r["time_ms"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)

    summary = {
        "engine": "llama-cpp",
        "model": os.path.basename(model_path),
        "model_size_mb": info["size_mb"],
        "total_prompts": len(results),
        "avg_time_ms": round(avg_time, 2),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "timestamp": datetime.now().isoformat()
    }

    print(f"\n  Average time:   {avg_time:.0f}ms")
    print(f"  Average tok/s:  {avg_tps:.1f}")

    return {"summary": summary, "results": results}


def compare_and_report(
    ollama_results: dict,
    gguf_results: dict,
    save: bool = True
) -> dict:
    """
    Compare Ollama vs GGUF benchmark results side by side.
    """
    o = ollama_results["summary"]
    g = gguf_results["summary"]

    print(f"\n{'='*60}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<25} {'Ollama':>15} {'GGUF':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Model':<25} {o['model']:>15} {g['model'][:15]:>15}")
    print(f"  {'Avg Latency (ms)':<25} {o['avg_time_ms']:>15.0f} {g['avg_time_ms']:>15.0f}")
    print(f"  {'Avg Tokens/sec':<25} {o['avg_tokens_per_sec']:>15.1f} {g['avg_tokens_per_sec']:>15.1f}")

    if "model_size_mb" in g:
        print(f"  {'Model Size (MB)':<25} {'~2000+':>15} {g['model_size_mb']:>15.0f}")

    # Determine winner
    if o["avg_tokens_per_sec"] > g["avg_tokens_per_sec"]:
        print(f"\n  🏆 Faster: Ollama ({o['avg_tokens_per_sec']:.1f} vs "
              f"{g['avg_tokens_per_sec']:.1f} tok/s)")
    else:
        print(f"\n  🏆 Faster: GGUF ({g['avg_tokens_per_sec']:.1f} vs "
              f"{o['avg_tokens_per_sec']:.1f} tok/s)")

    print(f"{'='*60}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "ollama": o,
        "gguf": g,
        "winner_speed": "ollama" if o["avg_tokens_per_sec"] > g["avg_tokens_per_sec"] else "gguf"
    }

    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        filename = f"{REPORTS_DIR}/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {filename}")

    return report