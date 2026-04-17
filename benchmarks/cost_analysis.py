import time
import json
import os
from datetime import datetime
from cache.cache_manager import CachedLLMClient
from cache.prompt_compressor import count_tokens_estimate


# Test queries — mix of unique and repeated/similar questions
TEST_QUERIES = [
    # Round 1 — First time asks
    "What is machine learning?",
    "Explain neural networks simply.",
    "What is RAG in AI?",
    "What are embeddings?",
    "How does gradient descent work?",

    # Round 2 — Exact same questions (should be exact cache hits)
    "What is machine learning?",
    "What are embeddings?",

    # Round 3 — Semantically similar (should be semantic cache hits)
    "Can you explain what ML is?",
    "What are word embeddings in NLP?",
    "Tell me about RAG systems.",
    "How do neural networks function?",
]


def run_cost_analysis(
    similarity_threshold: float = 0.92,
    save_report: bool = True
) -> dict:
    """
    Run a full cost analysis comparing cached vs uncached performance.
    Tests exact cache hits, semantic cache hits, and cache misses.
    """
    print(f"\n{'='*60}")
    print(f"  COST AND PERFORMANCE ANALYSIS")
    print(f"  Similarity threshold: {similarity_threshold}")
    print(f"  Test queries: {len(TEST_QUERIES)}")
    print(f"{'='*60}\n")

    client = CachedLLMClient(
        similarity_threshold=similarity_threshold,
        enable_compression=False
    )

    results = []
    total_start = time.time()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query}")

        start = time.time()
        result = client.chat(
            message=query,
            system_prompt="You are a helpful AI assistant. Be concise.",
            use_cache=True
        )
        elapsed = round(time.time() - start, 3)

        status = "CACHED" if result["from_cache"] else "LLM CALL"
        cache_type = result.get("cache_type", "miss")

        print(f"  Status: {status} ({cache_type}) | Time: {elapsed}s")
        print(f"  Answer: {result['answer'][:80]}...")

        results.append({
            "query": query,
            "from_cache": result["from_cache"],
            "cache_type": cache_type,
            "time_seconds": elapsed,
            "tokens_used": result.get("tokens_used", 0),
            "answer_length": len(result["answer"])
        })

    total_time = round(time.time() - total_start, 2)

    # Calculate savings
    cached_results = [r for r in results if r["from_cache"]]
    uncached_results = [r for r in results if not r["from_cache"]]

    avg_llm_time = sum(r["time_seconds"] for r in uncached_results) / len(uncached_results) if uncached_results else 0
    avg_cache_time = sum(r["time_seconds"] for r in cached_results) / len(cached_results) if cached_results else 0
    time_saved = sum(avg_llm_time - r["time_seconds"] for r in cached_results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "cache_hits": len(cached_results),
        "llm_calls": len(uncached_results),
        "hit_rate_percent": round(len(cached_results) / len(results) * 100, 1),
        "total_time_seconds": total_time,
        "avg_llm_time": round(avg_llm_time, 3),
        "avg_cache_time": round(avg_cache_time, 4),
        "estimated_time_saved": round(time_saved, 2),
        "speedup_factor": round(avg_llm_time / avg_cache_time, 1) if avg_cache_time > 0 else 0,
        "similarity_threshold": similarity_threshold
    }

    print(f"\n{'='*60}")
    print(f"  ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total queries:        {summary['total_queries']}")
    print(f"  Cache hits:           {summary['cache_hits']} ({summary['hit_rate_percent']}%)")
    print(f"  LLM calls made:       {summary['llm_calls']}")
    print(f"  Avg LLM response:     {summary['avg_llm_time']}s")
    print(f"  Avg cached response:  {summary['avg_cache_time']}s")
    print(f"  Time saved:           ~{summary['estimated_time_saved']}s")
    print(f"  Cache speedup:        ~{summary['speedup_factor']}x faster")
    print(f"  Total time:           {total_time}s")
    print(f"{'='*60}\n")

    client.show_metrics()
    client.cache.show_stats()

    if save_report:
        os.makedirs("benchmarks/reports", exist_ok=True)
        filename = f"benchmarks/reports/cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"Report saved: {filename}")

    return {"summary": summary, "results": results}


def compare_thresholds():
    """
    Compare cache performance at different similarity thresholds.
    Higher threshold = more precise but fewer hits.
    Lower threshold = more hits but some wrong answers.
    """
    thresholds = [0.98, 0.95, 0.92, 0.88]

    print(f"\n{'='*60}")
    print("  THRESHOLD COMPARISON")
    print(f"{'='*60}\n")

    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        from cache.semantic_cache import SemanticCache
        cache = SemanticCache(similarity_threshold=threshold)
        cache.clear()

        import ollama
        from config import MODEL

        test_pairs = [
            ("What is machine learning?", "Machine learning is a subset of AI."),
            ("What are embeddings?", "Embeddings are vector representations of text."),
        ]

        for query, answer in test_pairs:
            cache.set(query, answer)

        similar_queries = [
            "Can you explain machine learning?",
            "What is ML?",
            "Define word embeddings.",
        ]

        hits = 0
        for q in similar_queries:
            result = cache.get(q)
            if result:
                hits += 1
                print(f"  HIT  (sim={result.get('similarity', 'exact')}) '{q}'")
            else:
                print(f"  MISS '{q}'")

        print(f"  Hit rate: {hits}/{len(similar_queries)} = {round(hits/len(similar_queries)*100)}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cache Cost Analysis")
    parser.add_argument("--run",        action="store_true", help="Run full analysis")
    parser.add_argument("--thresholds", action="store_true", help="Compare thresholds")
    parser.add_argument("--threshold",  type=float, default=0.92, help="Similarity threshold")
    args = parser.parse_args()

    if args.thresholds:
        compare_thresholds()
    elif args.run:
        run_cost_analysis(similarity_threshold=args.threshold)
    else:
        run_cost_analysis(similarity_threshold=args.threshold)