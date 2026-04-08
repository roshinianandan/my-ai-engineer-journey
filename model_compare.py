import time
import ollama
from hf_models import load_pipeline, generate
from config import MODEL

TEST_PROMPTS = [
    "What is machine learning? Answer in 2 sentences.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is a neural network? Give a simple analogy.",
    "What is RAG in the context of AI? Keep it brief.",
    "What are embeddings and why are they useful?"
]


def query_ollama(prompt: str) -> dict:
    """Query the Ollama model and measure response time."""
    start = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    elapsed = round(time.time() - start, 2)
    return {
        "text": response["message"]["content"],
        "time_seconds": elapsed,
        "model": MODEL
    }


def query_hf(pipe, prompt: str) -> dict:
    """Query the HuggingFace model and measure response time."""
    return generate(pipe, prompt, max_new_tokens=150)


def compare(prompts: list = None):
    """
    Run every prompt through both models and display results side by side.
    Measures response quality and speed.
    """
    if prompts is None:
        prompts = TEST_PROMPTS

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON: Ollama vs HuggingFace")
    print("=" * 65)
    print(f"  Ollama model:      {MODEL}")
    print(f"  HuggingFace model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("=" * 65)

    print("\nLoading HuggingFace model...")
    pipe = load_pipeline()

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─'*65}")
        print(f"PROMPT {i}: {prompt}")
        print(f"{'─'*65}")

        # Ollama response
        ollama_result = query_ollama(prompt)
        print(f"\n🦙 Ollama ({ollama_result['time_seconds']}s):")
        print(f"   {ollama_result['text'][:300]}")

        # HuggingFace response
        hf_result = query_hf(pipe, prompt)
        print(f"\n🤗 HuggingFace ({hf_result['time_seconds']}s):")
        print(f"   {hf_result['text'][:300]}")

        results.append({
            "prompt": prompt,
            "ollama_time": ollama_result["time_seconds"],
            "hf_time": hf_result["time_seconds"]
        })

    # Summary
    print(f"\n{'='*65}")
    print("  SPEED SUMMARY")
    print(f"{'='*65}")
    avg_ollama = sum(r["ollama_time"] for r in results) / len(results)
    avg_hf = sum(r["hf_time"] for r in results) / len(results)
    print(f"  Avg Ollama response time:      {avg_ollama:.2f}s")
    print(f"  Avg HuggingFace response time: {avg_hf:.2f}s")
    faster = "Ollama" if avg_ollama < avg_hf else "HuggingFace"
    print(f"  Faster model: {faster}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Ollama vs HuggingFace")
    parser.add_argument("--prompt", type=str, help="Test a single custom prompt")
    args = parser.parse_args()

    if args.prompt:
        print("\nLoading HuggingFace model...")
        pipe = load_pipeline()
        print(f"\n{'─'*55}")
        print(f"PROMPT: {args.prompt}")
        print(f"{'─'*55}")
        ollama_r = query_ollama(args.prompt)
        hf_r = query_hf(pipe, args.prompt)
        print(f"\n🦙 Ollama ({ollama_r['time_seconds']}s): {ollama_r['text'][:300]}")
        print(f"\n🤗 HuggingFace ({hf_r['time_seconds']}s): {hf_r['text'][:300]}")
    else:
        compare()