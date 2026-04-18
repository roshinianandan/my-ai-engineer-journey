import ollama
from config import MODEL


class HyDERetriever:
    """
    HyDE — Hypothetical Document Embeddings

    The problem with query embeddings:
    A query like 'What is RAG?' is short and sparse.
    The ideal document answering it is long and dense.
    These live in different regions of the embedding space
    so similarity search is less effective.

    HyDE solution:
    1. Generate a hypothetical answer to the query using the LLM
    2. Embed the hypothetical answer (not the query)
    3. Search for real documents similar to the hypothetical answer
    4. The hypothetical answer is in the same embedding space as real docs

    This dramatically improves retrieval for complex questions.
    """

    def __init__(self, num_hypotheses: int = 1):
        self.num_hypotheses = num_hypotheses

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical answer to the query.
        This answer may be partially wrong — that is fine.
        Its purpose is to be embedded, not to be shown to the user.
        """
        prompt = f"""Write a short passage (3-5 sentences) that directly answers this question.
Write it as if it were from an authoritative document on the topic.
Be specific and use relevant technical terminology.
Do not say 'this passage' or 'according to' — just write the answer directly.

Question: {query}

Passage:"""

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.7}
        )

        return response["message"]["content"]

    def generate_multiple_hypotheses(self, query: str) -> list:
        """
        Generate multiple hypothetical documents for better coverage.
        Averaging their embeddings produces a more robust search vector.
        """
        hypotheses = []
        for i in range(self.num_hypotheses):
            hyp = self.generate_hypothetical_document(query)
            hypotheses.append(hyp)
        return hypotheses

    def get_search_embedding(self, query: str) -> tuple:
        """
        Get the embedding to use for search.
        Returns (embedding, hypothetical_document) tuple.
        """
        import ollama as ol

        hypothetical_doc = self.generate_hypothetical_document(query)

        response = ol.embeddings(
            model="nomic-embed-text",
            prompt=hypothetical_doc
        )
        embedding = response["embedding"]

        return embedding, hypothetical_doc

    def search(
        self,
        query: str,
        top_k: int = 5,
        show_hypothesis: bool = True
    ) -> list:
        """
        HyDE search: generate hypothesis, embed it, search with it.
        """
        from vector_store import get_or_create_collection
        collection = get_or_create_collection("knowledge_base")

        if collection.count() == 0:
            print("[HyDE] No documents indexed.")
            return []

        print(f"\n[HyDE] Generating hypothetical document for: '{query}'")
        embedding, hypothesis = self.get_search_embedding(query)

        if show_hypothesis:
            print(f"\n[HyDE] Hypothetical document:\n{hypothesis}\n")

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "distances", "metadatas"]
        )

        chunks = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            chunks.append({
                "text": doc,
                "score": round(1 - dist, 4),
                "source": meta.get("source", "unknown"),
                "retrieval_method": "hyde"
            })

        return chunks