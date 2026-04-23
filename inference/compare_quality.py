import ollama
import math
from config import MODEL


QUALITY_PROMPTS = [
    {
        "id": "q1",
        "prompt": "What is machine learning? Answer in exactly 2 sentences.",
        "expected_keywords": ["data", "learn", "algorithm", "pattern", "AI"]
    },
    {
        "id": "q2",
        "prompt": "What is the difference between RAM and ROM?",
        "expected_keywords": ["memory", "store", "read", "write", "volatile"]
    },
    {
        "id": "q3",
        "prompt": "Explain what a neural network is in simple terms.",
        "expected_keywords": ["neuron", "layer", "input", "output", "brain"]
    },
    {
        "id": "q4",
        "prompt": "What programming language is best for data science?",
        "expected_keywords": ["Python", "library", "pandas", "numpy", "data"]
    },
    {
        "id": "q5",
        "prompt": "What is gradient descent?",
        "expected_keywords": ["optimize", "loss", "minimum", "weight", "train"]
    }
]


def get_embedding(text: str) -> list:
    """Get embedding for semantic similarity scoring."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def score_response(
    response: str,
    expected_keywords: list,
    reference_response: str = None
) -> dict:
    """
    Score a model response for quality.

    Metrics:
    1. Keyword coverage — does it contain expected keywords?
    2. Length appropriateness — not too short, not too verbose
    3. Semantic similarity — if reference provided
    """
    response_lower = response.lower()

    # Keyword score
    found_keywords = [
        kw for kw in expected_keywords
        if kw.lower() in response_lower
    ]
    keyword_score = len(found_keywords) / len(expected_keywords)

    # Length score (ideal: 30-200 words)
    word_count = len(response.split())
    if 30 <= word_count <= 200:
        length_score = 1.0
    elif word_count < 10:
        length_score = 0.2
    elif word_count < 30:
        length_score = 0.6
    else:
        length_score = max(0.5, 1.0 - (word_count - 200) / 500)

    # Semantic similarity (if reference provided)
    semantic_score = None
    if reference_response:
        try:
            emb1 = get_embedding(response)
            emb2 = get_embedding(reference_response)
            semantic_score = round(cosine_similarity(emb1, emb2), 4)
        except Exception:
            semantic_score = None

    # Combined score
    if semantic_score is not None:
        combined = (keyword_score * 0.3 + length_score * 0.2 +
                    semantic_score * 0.5)
    else:
        combined = keyword_score * 0.6 + length_score * 0.4

    return {
        "keyword_score": round(keyword_score, 3),
        "keywords_found": found_keywords,
        "length_score": round(length_score, 3),
        "word_count": word_count,
        "semantic_score": semantic_score,
        "combined_score": round(combined, 3)
    }


def compare_quality_ollama_vs_gguf(
    gguf_model_path: str = None,
    prompts: list = None
) -> dict:
    """
    Compare response quality between Ollama and GGUF model.
    """
    prompts = prompts or QUALITY_PROMPTS

    print(f"\n{'='*60}")
    print("  QUALITY COMPARISON: Ollama vs GGUF")
    print(f"{'='*60}\n")

    all_results = []

    for item in prompts:
        prompt = item["prompt"]
        keywords = item["expected_keywords"]
        print(f"\nPrompt: {prompt[:60]}...")

        # Get Ollama response
        ollama_response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.3}
        )["message"]["content"]

        ollama_scores = score_response(ollama_response, keywords)
        print(f"  Ollama score: {ollama_scores['combined_score']:.3f} | "
              f"Keywords: {ollama_scores['keywords_found']}")

        result = {
            "id": item["id"],
            "prompt": prompt[:60],
            "ollama": {
                "response": ollama_response[:200],
                "scores": ollama_scores
            }
        }

        # Get GGUF response if model provided
        if gguf_model_path:
            try:
                from inference.quantized_model import QuantizedModel
                gguf = QuantizedModel(model_path=gguf_model_path)
                gguf_result = gguf.chat(
                    message=prompt,
                    max_tokens=150,
                    temperature=0.3
                )
                gguf_response = gguf_result["text"]

                gguf_scores = score_response(
                    gguf_response,
                    keywords,
                    reference_response=ollama_response
                )
                print(f"  GGUF score:   {gguf_scores['combined_score']:.3f} | "
                      f"Keywords: {gguf_scores['keywords_found']}")

                result["gguf"] = {
                    "response": gguf_response[:200],
                    "scores": gguf_scores
                }

            except Exception as e:
                print(f"  GGUF error: {e}")
                result["gguf"] = {"error": str(e)}

        all_results.append(result)

    # Summary
    ollama_avg = sum(
        r["ollama"]["scores"]["combined_score"]
        for r in all_results
    ) / len(all_results)

    print(f"\n{'='*60}")
    print("  QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"  Ollama avg quality score: {ollama_avg:.3f}")

    if gguf_model_path and "gguf" in all_results[0]:
        gguf_avg = sum(
            r["gguf"]["scores"]["combined_score"]
            for r in all_results
            if "scores" in r.get("gguf", {})
        ) / len(all_results)
        print(f"  GGUF avg quality score:   {gguf_avg:.3f}")
        winner = "Ollama" if ollama_avg > gguf_avg else "GGUF"
        print(f"\n  🏆 Better quality: {winner}")

    print(f"{'='*60}\n")

    return {"results": all_results, "ollama_avg_score": ollama_avg}


if __name__ == "__main__":
    import argparse
    from inference.gguf_loader import (
        list_available_models,
        list_downloaded_models,
        download_model,
        get_model_path
    )

    parser = argparse.ArgumentParser(description="Model Comparison Tools")
    parser.add_argument("--list",      action="store_true",
                        help="List available GGUF models")
    parser.add_argument("--downloaded",action="store_true",
                        help="List downloaded models")
    parser.add_argument("--download",  type=str,
                        help="Download a GGUF model by key")
    parser.add_argument("--benchmark-ollama", action="store_true",
                        help="Benchmark Ollama model")
    parser.add_argument("--benchmark-gguf",   type=str,
                        help="Benchmark a GGUF model (path or key)")
    parser.add_argument("--compare",   type=str,
                        help="Compare quality: Ollama vs GGUF model path")
    parser.add_argument("--chat",      type=str,
                        help="Chat with GGUF model (path or key)")
    parser.add_argument("--full",      action="store_true",
                        help="Run full benchmark and comparison")
    args = parser.parse_args()

    if args.list:
        list_available_models()

    elif args.downloaded:
        list_downloaded_models()

    elif args.download:
        download_model(args.download)

    elif args.benchmark_ollama:
        from inference.benchmark import benchmark_ollama
        benchmark_ollama()

    elif args.benchmark_gguf:
        from inference.benchmark import benchmark_gguf
        path = get_model_path(
            model_key=args.benchmark_gguf
            if args.benchmark_gguf in ["tinyllama-q4", "tinyllama-q8", "phi2-q4"]
            else None,
            model_path=args.benchmark_gguf
            if os.path.exists(args.benchmark_gguf) else None
        )
        benchmark_gguf(path)

    elif args.compare:
        compare_quality_ollama_vs_gguf(gguf_model_path=args.compare)

    elif args.chat:
        from inference.quantized_model import QuantizedModel
        path = get_model_path(
            model_key=args.chat
            if args.chat in ["tinyllama-q4", "tinyllama-q8", "phi2-q4"]
            else None,
            model_path=args.chat if os.path.exists(args.chat) else None
        )
        model = QuantizedModel(model_path=path)
        print(f"\n💬 Chat with {os.path.basename(path)}")
        print("   Type 'quit' to exit\n")
        while True:
            try:
                msg = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if msg.lower() == "quit":
                break
            result = model.chat(msg)
            print(f"\n🤖 {result['text']}")
            print(f"   [{result['tokens_per_second']} tok/s]\n")

    elif args.full:
        import os
        from inference.benchmark import benchmark_ollama, benchmark_gguf, compare_and_report
        from inference.gguf_loader import get_model_path

        print("Running full benchmark suite...")
        ollama_results = benchmark_ollama()

        downloaded = list_downloaded_models()
        if downloaded:
            gguf_path = str(downloaded[0])
            gguf_results = benchmark_gguf(gguf_path)
            compare_and_report(ollama_results, gguf_results)

            quality = compare_quality_ollama_vs_gguf(gguf_model_path=gguf_path)
        else:
            print("\nNo GGUF model downloaded. Run --download tinyllama-q4 first.")

    else:
        parser.print_help()