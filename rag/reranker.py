import ollama
from config import MODEL


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using an LLM as a cross-encoder.

    The difference between retrieval and reranking:
    - Retrieval (bi-encoder): embed query once, embed docs once, compare vectors
      Fast but approximate — good for finding candidates
    - Reranking (cross-encoder): score each (query, doc) pair together
      Slower but more accurate — good for final ranking of candidates

    In production this uses a dedicated cross-encoder model.
    Here we use the LLM as a judge for relevance scoring.
    """

    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size

    def score_chunk(self, query: str, chunk: str) -> float:
        """
        Score the relevance of a single chunk to a query.
        Returns a score from 0.0 (irrelevant) to 1.0 (perfectly relevant).
        """
        prompt = f"""Score how relevant this passage is to the question.
Return ONLY a number from 0 to 10. Nothing else.

Question: {query}

Passage: {chunk[:400]}

Relevance score (0-10):"""

        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.0}
            )
            raw = response["message"]["content"].strip()
            # Extract first number found
            import re
            match = re.search(r"\d+(?:\.\d+)?", raw)
            if match:
                score = float(match.group())
                return min(1.0, max(0.0, score / 10.0))
            return 0.5
        except Exception:
            return 0.5

    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int = 3,
        verbose: bool = True
    ) -> list:
        """
        Rerank a list of chunks by relevance to the query.

        chunks: list of dicts with 'text', 'source', 'hybrid_score'
        Returns top_k chunks sorted by reranker score.
        """
        if not chunks:
            return []

        if verbose:
            print(f"\n[Reranker] Scoring {len(chunks)} chunks for query: '{query[:50]}...'")

        scored = []
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"  [{i+1}/{len(chunks)}] Scoring: {chunk['text'][:60]}...")

            rerank_score = self.score_chunk(query, chunk["text"])

            # Combine reranker score with retrieval score
            retrieval_score = chunk.get("hybrid_score",
                                        chunk.get("semantic_score", 0.5))
            final_score = round(0.7 * rerank_score + 0.3 * retrieval_score, 4)

            scored.append({
                **chunk,
                "rerank_score": rerank_score,
                "final_score": final_score
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        if verbose:
            print(f"\n[Reranker] Top {top_k} after reranking:")
            for r in scored[:top_k]:
                print(f"  final={r['final_score']:.3f} | "
                      f"rerank={r['rerank_score']:.3f} | "
                      f"retrieval={r.get('hybrid_score', 0):.3f} | "
                      f"{r['text'][:60]}...")

        return scored[:top_k]