import time
import ollama
from cache.semantic_cache import SemanticCache
from cache.prompt_compressor import compress_prompt, count_tokens_estimate
from config import MODEL, TEMPERATURE


class CachedLLMClient:
    """
    A drop-in wrapper around Ollama that adds:
    1. Semantic caching — skip LLM call if similar query cached
    2. Prompt compression — reduce tokens before sending
    3. Cost tracking — measure every request
    4. Performance metrics — compare cached vs uncached speed
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        enable_compression: bool = False,
        max_prompt_tokens: int = 800
    ):
        self.cache = SemanticCache(similarity_threshold=similarity_threshold)
        self.enable_compression = enable_compression
        self.max_prompt_tokens = max_prompt_tokens

        self.metrics = {
            "total_calls": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "total_time": 0.0,
            "llm_time": 0.0,
            "cached_time": 0.0,
            "estimated_tokens_used": 0,
            "estimated_tokens_saved": 0
        }

    def chat(
        self,
        message: str,
        system_prompt: str = "",
        use_cache: bool = True,
        stream: bool = False
    ) -> dict:
        """
        Send a message with caching and compression.
        Returns response dict with answer and metadata.
        """
        self.metrics["total_calls"] += 1
        start_time = time.time()

        # Check cache first
        if use_cache:
            cached = self.cache.get(message)
            if cached:
                self.metrics["cache_hits"] += 1
                elapsed = round(time.time() - start_time, 4)
                self.metrics["cached_time"] += elapsed
                return {
                    "answer": cached["answer"],
                    "from_cache": True,
                    "cache_type": cached.get("cache_hit", "unknown"),
                    "time_seconds": elapsed,
                    "tokens_used": 0
                }

        # Compress prompt if enabled
        final_message = message
        compression_stats = None

        if self.enable_compression:
            result = compress_prompt(
                message,
                system_prompt,
                max_total_tokens=self.max_prompt_tokens
            )
            final_message = result["compressed_prompt"]
            compression_stats = result

        # Call LLM
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": final_message})

        if stream:
            print("\n🤖 Answer: ", end="", flush=True)
            full_reply = ""
            for chunk in ollama.chat(
                model=MODEL,
                messages=messages,
                stream=True,
                options={"temperature": TEMPERATURE}
            ):
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_reply += token
            print()
        else:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                stream=False,
                options={"temperature": TEMPERATURE}
            )
            full_reply = response["message"]["content"]

        elapsed = round(time.time() - start_time, 2)
        tokens_used = count_tokens_estimate(final_message + full_reply)

        self.metrics["llm_calls"] += 1
        self.metrics["llm_time"] += elapsed
        self.metrics["total_time"] += elapsed
        self.metrics["estimated_tokens_used"] += tokens_used

        # Store in cache
        if use_cache:
            self.cache.set(message, full_reply)

        return {
            "answer": full_reply,
            "from_cache": False,
            "time_seconds": elapsed,
            "tokens_used": tokens_used,
            "compression": compression_stats
        }

    def show_metrics(self):
        """Display performance and cost metrics."""
        m = self.metrics
        total = m["total_calls"]
        hits = m["cache_hits"]
        hit_rate = round(hits / total * 100, 1) if total > 0 else 0
        avg_llm_time = round(m["llm_time"] / m["llm_calls"], 2) if m["llm_calls"] > 0 else 0
        avg_cache_time = round(m["cached_time"] / hits, 4) if hits > 0 else 0
        speedup = round(avg_llm_time / avg_cache_time) if avg_cache_time > 0 else 0

        print(f"\n{'='*55}")
        print(f"  PERFORMANCE METRICS")
        print(f"{'='*55}")
        print(f"  Total calls:        {total}")
        print(f"  Cache hits:         {hits} ({hit_rate}%)")
        print(f"  LLM calls:          {m['llm_calls']}")
        print(f"  Avg LLM time:       {avg_llm_time}s")
        print(f"  Avg cache time:     {avg_cache_time}s")
        print(f"  Cache speedup:      ~{speedup}x faster")
        print(f"  Tokens used:        ~{m['estimated_tokens_used']}")
        print(f"  Tokens saved:       ~{m['estimated_tokens_saved']}")
        print(f"{'='*55}\n")

    def clear_cache(self):
        """Clear all cached entries."""
        self.cache.clear()
        print("[Cache cleared]")