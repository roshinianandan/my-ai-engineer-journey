import math
import ollama
from rank_bm25 import BM25Okapi
from vector_store import get_or_create_collection

KB_COLLECTION = "knowledge_base"


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def cosine_similarity(vec1: list, vec2: list) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class HybridRetriever:
    """
    Combines BM25 keyword search with semantic vector search.

    Why hybrid beats either alone:
    - Semantic search finds meaning but misses exact keywords
    - BM25 finds exact keywords but misses meaning
    - Hybrid finds both — better recall, better precision

    The final score = alpha * semantic_score + (1 - alpha) * bm25_score
    alpha=1.0 = pure semantic, alpha=0.0 = pure BM25, alpha=0.5 = balanced
    """

    def __init__(self, alpha: float = 0.6):
        """
        alpha: weight for semantic search (0.0 to 1.0)
        1 - alpha: weight for BM25 keyword search
        """
        self.alpha = alpha
        self.collection = get_or_create_collection(KB_COLLECTION)
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        Build BM25 index from all documents in ChromaDB.
        BM25 works on tokenized text — splits each document into words.
        """
        try:
            all_docs = self.collection.get(include=["documents", "metadatas"])

            if not all_docs["documents"]:
                print("[HybridRetriever] No documents found in collection")
                return

            self.corpus = all_docs["documents"]
            self.corpus_ids = all_docs["ids"]
            self.corpus_metadatas = all_docs.get("metadatas", [{}] * len(self.corpus))

            # Tokenize for BM25 — simple whitespace tokenization
            tokenized = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized)

            print(f"[HybridRetriever] BM25 index built: {len(self.corpus)} documents")

        except Exception as e:
            print(f"[HybridRetriever] Error building index: {e}")

    def _semantic_search(self, query: str, top_k: int = 10) -> list:
        """Run semantic search using ChromaDB."""
        if self.collection.count() == 0:
            return []

        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "distances", "metadatas"]
        )

        semantic_results = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            semantic_results.append({
                "text": doc,
                "semantic_score": round(1 - dist, 4),
                "source": meta.get("source", "unknown"),
                "metadata": meta
            })

        return semantic_results

    def _bm25_search(self, query: str, top_k: int = 10) -> list:
        """Run BM25 keyword search."""
        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        normalized = [s / max_score for s in scores]

        # Get top-k results
        top_indices = sorted(
            range(len(normalized)),
            key=lambda i: normalized[i],
            reverse=True
        )[:top_k]

        bm25_results = []
        for idx in top_indices:
            if normalized[idx] > 0:
                meta = self.corpus_metadatas[idx] if idx < len(self.corpus_metadatas) else {}
                bm25_results.append({
                    "text": self.corpus[idx],
                    "bm25_score": round(normalized[idx], 4),
                    "source": meta.get("source", "unknown"),
                    "metadata": meta
                })

        return bm25_results

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str = None
    ) -> list:
        """
        Hybrid search: combine semantic and BM25 results.

        Algorithm:
        1. Run both searches with 2x top_k candidates
        2. Merge results, combining scores for documents found by both
        3. Apply hybrid score = alpha * semantic + (1-alpha) * bm25
        4. Sort by hybrid score and return top_k
        """
        candidates_k = top_k * 2

        semantic_results = self._semantic_search(query, top_k=candidates_k)
        bm25_results = self._bm25_search(query, top_k=candidates_k)

        # Merge into unified score dict
        merged = {}

        for r in semantic_results:
            key = r["text"][:100]  # use text prefix as key
            merged[key] = {
                "text": r["text"],
                "semantic_score": r["semantic_score"],
                "bm25_score": 0.0,
                "source": r["source"],
                "metadata": r.get("metadata", {})
            }

        for r in bm25_results:
            key = r["text"][:100]
            if key in merged:
                merged[key]["bm25_score"] = r["bm25_score"]
            else:
                merged[key] = {
                    "text": r["text"],
                    "semantic_score": 0.0,
                    "bm25_score": r["bm25_score"],
                    "source": r["source"],
                    "metadata": r.get("metadata", {})
                }

        # Calculate hybrid score
        results = []
        for item in merged.values():
            hybrid_score = (
                self.alpha * item["semantic_score"] +
                (1 - self.alpha) * item["bm25_score"]
            )
            item["hybrid_score"] = round(hybrid_score, 4)

            # Apply source filter
            if source_filter and item["source"] != source_filter:
                continue

            results.append(item)

        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]

    def compare_search_modes(self, query: str, top_k: int = 3) -> dict:
        """
        Run all three search modes and compare results.
        Useful for understanding when hybrid outperforms individual methods.
        """
        print(f"\nComparing search modes for: '{query}'\n")

        # Save original alpha
        original_alpha = self.alpha

        # Pure semantic
        self.alpha = 1.0
        semantic_only = self.search(query, top_k=top_k)

        # Pure BM25
        self.alpha = 0.0
        bm25_only = self.search(query, top_k=top_k)

        # Hybrid
        self.alpha = 0.6
        hybrid = self.search(query, top_k=top_k)

        # Restore
        self.alpha = original_alpha

        print("Semantic Only:")
        for r in semantic_only:
            print(f"  [{r['semantic_score']:.3f}] {r['text'][:80]}...")

        print("\nBM25 Only:")
        for r in bm25_only:
            print(f"  [{r['bm25_score']:.3f}] {r['text'][:80]}...")

        print("\nHybrid (alpha=0.6):")
        for r in hybrid:
            print(f"  [{r['hybrid_score']:.3f}] {r['text'][:80]}...")

        return {
            "query": query,
            "semantic": semantic_only,
            "bm25": bm25_only,
            "hybrid": hybrid
        }