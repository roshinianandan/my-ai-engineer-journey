import ollama
import argparse
from config import MODEL
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.hyde import HyDERetriever


def generate_answer(query: str, chunks: list, stream: bool = True) -> str:
    """Generate a grounded answer from retrieved chunks."""
    if not chunks:
        return "No relevant information found."

    context = "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'unknown')} | "
        f"Score: {c.get('final_score', c.get('score', c.get('hybrid_score', 0))):.3f}]\n"
        f"{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present, say you do not have that information.
Always cite which source you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    if stream:
        print("\n🤖 Answer: ", end="", flush=True)
        full_reply = ""
        for chunk in ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.3}
        ):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_reply += token
        print()
        return full_reply
    else:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.3}
        )
        return response["message"]["content"]


def advanced_rag(
    query: str,
    use_reranker: bool = True,
    use_hyde: bool = False,
    top_k: int = 5,
    rerank_top_k: int = 3,
    alpha: float = 0.6,
    verbose: bool = True
) -> dict:
    """
    Full Advanced RAG pipeline:

    Standard RAG:  Query → Semantic Search → Generate
    Advanced RAG:  Query → HyDE (optional) → Hybrid Search
                        → Reranker → Generate

    Each step improves retrieval quality at the cost of latency.
    """
    print(f"\n{'='*60}")
    print(f"  ADVANCED RAG PIPELINE")
    print(f"  Query: {query}")
    print(f"  HyDE: {use_hyde} | Reranker: {use_reranker} | Alpha: {alpha}")
    print(f"{'='*60}\n")

    # Step 1: Retrieval
    if use_hyde:
        print("[Step 1] HyDE Retrieval...")
        hyde = HyDERetriever()
        chunks = hyde.search(query, top_k=top_k, show_hypothesis=verbose)
    else:
        print("[Step 1] Hybrid Retrieval (BM25 + Semantic)...")
        retriever = HybridRetriever(alpha=alpha)
        chunks = retriever.search(query, top_k=top_k)

    if not chunks:
        return {"query": query, "answer": "No documents found.", "chunks": []}

    if verbose:
        print(f"\nRetrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            score_key = "score" if "score" in c else "hybrid_score"
            print(f"  [{i}] {c.get(score_key, 0):.3f} | "
                  f"{c.get('source', 'unknown')} | "
                  f"{c['text'][:60]}...")

    # Step 2: Reranking (optional)
    if use_reranker:
        print(f"\n[Step 2] Reranking {len(chunks)} chunks...")
        reranker = CrossEncoderReranker()
        chunks = reranker.rerank(query, chunks, top_k=rerank_top_k, verbose=verbose)
    else:
        chunks = chunks[:rerank_top_k]

    # Step 3: Generation
    print(f"\n[Step 3] Generating answer from {len(chunks)} chunks...")
    answer = generate_answer(query, chunks, stream=True)

    return {
        "query": query,
        "answer": answer,
        "chunks_used": len(chunks),
        "pipeline": {
            "hyde": use_hyde,
            "reranker": use_reranker,
            "alpha": alpha
        }
    }


def compare_pipelines(query: str) -> dict:
    """
    Compare naive RAG vs hybrid RAG vs hybrid+reranker.
    Shows the quality difference at each stage.
    """
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPARISON")
    print(f"  Query: {query}")
    print(f"{'='*60}")

    results = {}

    # Pipeline 1: Naive (semantic only, no reranker)
    print("\n--- Pipeline 1: Naive RAG (semantic only) ---")
    results["naive"] = advanced_rag(
        query, use_reranker=False, use_hyde=False, alpha=1.0, verbose=False
    )

    # Pipeline 2: Hybrid (no reranker)
    print("\n--- Pipeline 2: Hybrid RAG (BM25 + semantic) ---")
    results["hybrid"] = advanced_rag(
        query, use_reranker=False, use_hyde=False, alpha=0.6, verbose=False
    )

    # Pipeline 3: Hybrid + Reranker
    print("\n--- Pipeline 3: Hybrid + Reranker ---")
    results["hybrid_reranker"] = advanced_rag(
        query, use_reranker=True, use_hyde=False, alpha=0.6, verbose=True
    )

    return results


def interactive_advanced_rag():
    """Interactive advanced RAG session."""
    print(f"\n📚 Advanced RAG Assistant")
    print("   Commands: 'hyde on/off', 'rerank on/off', 'compare', 'quit'\n")
    print("-" * 55)

    use_hyde = False
    use_reranker = True

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "hyde on":
            use_hyde = True
            print("HyDE: ON")
            continue
        if query.lower() == "hyde off":
            use_hyde = False
            print("HyDE: OFF")
            continue
        if query.lower() == "rerank on":
            use_reranker = True
            print("Reranker: ON")
            continue
        if query.lower() == "rerank off":
            use_reranker = False
            print("Reranker: OFF")
            continue
        if query.lower().startswith("compare"):
            compare_pipelines(query.replace("compare", "").strip() or query)
            continue

        advanced_rag(
            query,
            use_reranker=use_reranker,
            use_hyde=use_hyde
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline")
    parser.add_argument("--query",   type=str, help="Single query")
    parser.add_argument("--hyde",    action="store_true", help="Enable HyDE")
    parser.add_argument("--rerank",  action="store_true", help="Enable reranker")
    parser.add_argument("--compare", type=str, help="Compare all pipelines on a query")
    parser.add_argument("--chat",    action="store_true", help="Interactive session")
    parser.add_argument("--alpha",   type=float, default=0.6, help="Semantic weight")
    args = parser.parse_args()

    if args.compare:
        compare_pipelines(args.compare)
    elif args.query:
        advanced_rag(
            args.query,
            use_reranker=args.rerank,
            use_hyde=args.hyde,
            alpha=args.alpha
        )
    elif args.chat:
        interactive_advanced_rag()
    else:
        interactive_advanced_rag()