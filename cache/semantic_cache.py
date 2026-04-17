import json
import math
import time
import hashlib
import ollama
from datetime import datetime
from vector_store import get_or_create_collection

CACHE_COLLECTION = "semantic_cache"


def get_embedding(text: str) -> list:
    """Get embedding for cache key."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def exact_hash(text: str) -> str:
    """Create a hash for exact match caching."""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


class SemanticCache:
    """
    A two-level cache system:

    Level 1 — Exact cache: instant lookup using MD5 hash.
    If the exact same question was asked before, return instantly.

    Level 2 — Semantic cache: find similar questions using
    cosine similarity. If a question meaning >0.95 similar
    to a cached question, return that cached answer.

    This saves API calls when users ask the same thing
    in slightly different ways.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_cache_size: int = 1000,
        ttl_hours: int = 24
    ):
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_hours = ttl_hours
        self.collection = get_or_create_collection(CACHE_COLLECTION)

        # In-memory exact cache for instant lookups
        self.exact_cache: dict = {}

        # Stats tracking
        self.stats = {
            "total_requests": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "tokens_saved": 0
        }

    def get(self, query: str) -> dict | None:
        """
        Try to find a cached answer for the query.
        Returns cached result dict or None if not found.
        """
        self.stats["total_requests"] += 1

        # Level 1: Exact match check
        query_hash = exact_hash(query)
        if query_hash in self.exact_cache:
            self.stats["exact_hits"] += 1
            cached = self.exact_cache[query_hash]
            cached["cache_hit"] = "exact"
            cached["cache_level"] = 1
            print(f"  [Cache HIT - Exact] '{query[:50]}...'")
            return cached

        # Level 2: Semantic similarity check
        if self.collection.count() == 0:
            self.stats["misses"] += 1
            return None

        try:
            query_embedding = get_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "distances", "metadatas"]
            )

            if results["documents"][0]:
                distance = results["distances"][0][0]
                similarity = round(1 - distance, 4)

                if similarity >= self.similarity_threshold:
                    meta = results["metadatas"][0][0]
                    cached_answer = meta.get("answer", "")
                    cached_query = results["documents"][0][0]

                    self.stats["semantic_hits"] += 1
                    estimated_tokens = len(query.split()) + len(cached_answer.split())
                    self.stats["tokens_saved"] += estimated_tokens

                    print(f"  [Cache HIT - Semantic] similarity={similarity}")
                    print(f"  Original query: '{cached_query[:60]}...'")

                    return {
                        "answer": cached_answer,
                        "cache_hit": "semantic",
                        "cache_level": 2,
                        "similarity": similarity,
                        "original_query": cached_query
                    }

        except Exception as e:
            print(f"  [Cache lookup error: {e}]")

        self.stats["misses"] += 1
        return None

    def set(self, query: str, answer: str, metadata: dict = None):
        """
        Store a query-answer pair in the cache.
        Stores in both exact cache and semantic cache.
        """
        # Store in exact cache
        query_hash = exact_hash(query)
        self.exact_cache[query_hash] = {
            "answer": answer,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }

        # Store in semantic cache
        try:
            query_embedding = get_embedding(query)
            cache_id = f"cache_{query_hash}"

            self.collection.upsert(
                documents=[query],
                embeddings=[query_embedding],
                ids=[cache_id],
                metadatas=[{
                    "answer": answer[:2000],  # ChromaDB metadata limit
                    "timestamp": datetime.now().isoformat(),
                    "query_hash": query_hash,
                    **(metadata or {})
                }]
            )
            print(f"  [Cached] '{query[:50]}...'")

        except Exception as e:
            print(f"  [Cache store error: {e}]")

    def invalidate(self, query: str):
        """Remove a specific query from cache."""
        query_hash = exact_hash(query)

        if query_hash in self.exact_cache:
            del self.exact_cache[query_hash]

        try:
            self.collection.delete(ids=[f"cache_{query_hash}"])
            print(f"  [Cache invalidated] '{query[:50]}'")
        except Exception:
            pass

    def clear(self):
        """Clear all cached entries."""
        self.exact_cache.clear()
        try:
            all_items = self.collection.get()
            if all_items["ids"]:
                self.collection.delete(ids=all_items["ids"])
        except Exception:
            pass
        print("[Cache cleared]")

    def show_stats(self):
        """Display cache performance statistics."""
        s = self.stats
        total = s["total_requests"]
        hits = s["exact_hits"] + s["semantic_hits"]
        hit_rate = round((hits / total * 100), 1) if total > 0 else 0

        print(f"\n{'='*50}")
        print(f"  CACHE STATISTICS")
        print(f"{'='*50}")
        print(f"  Total requests:   {total}")
        print(f"  Exact hits:       {s['exact_hits']}")
        print(f"  Semantic hits:    {s['semantic_hits']}")
        print(f"  Cache misses:     {s['misses']}")
        print(f"  Hit rate:         {hit_rate}%")
        print(f"  Tokens saved:     ~{s['tokens_saved']}")
        print(f"  Cache size:       {self.collection.count()} entries")
        print(f"{'='*50}\n")

    def list_cached(self, limit: int = 10) -> list:
        """List recently cached queries."""
        try:
            all_items = self.collection.get(include=["documents", "metadatas"])
            entries = []
            for doc, meta in zip(
                all_items["documents"][:limit],
                all_items["metadatas"][:limit]
            ):
                entries.append({
                    "query": doc[:80],
                    "timestamp": meta.get("timestamp", ""),
                    "answer_preview": meta.get("answer", "")[:60]
                })
            return entries
        except Exception:
            return []