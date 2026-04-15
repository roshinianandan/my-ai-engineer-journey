import time
import ollama
from config import MODEL


def stream_rag_answer(
    question: str,
    top_k: int = 3,
    source_filter: str = None,
    level_filter: str = None,
    show_sources: bool = True
) -> str:
    """
    Full RAG pipeline with streaming generation.

    Steps:
    1. Retrieve relevant chunks (instant — already indexed)
    2. Show sources to user immediately
    3. Stream the generated answer token by token

    The user sees sources appear first, then the answer streams in.
    This feels much faster than waiting for the full response.
    """
    from rag.knowledge_base import search

    print(f"\n🔍 Retrieving relevant chunks for: '{question}'")
    start = time.time()

    chunks = search(
        query=question,
        top_k=top_k,
        source_filter=source_filter,
        level_filter=level_filter
    )

    retrieval_time = round(time.time() - start, 2)

    if not chunks:
        print("No relevant chunks found in the knowledge base.")
        return ""

    print(f"   Found {len(chunks)} chunks in {retrieval_time}s\n")

    if show_sources:
        print("📚 Sources:")
        for i, chunk in enumerate(chunks, 1):
            print(f"   [{i}] {chunk['source']} | Score: {chunk['score']}")
            print(f"       {chunk['text'][:80]}...\n")

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Score: {c['score']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present, say you do not have that information.
Always mention which source you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    print("🤖 Answer: ", end="", flush=True)

    full_reply = ""
    gen_start = time.time()
    token_count = 0

    for chunk in ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"temperature": 0.3}
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_reply += token
        token_count += 1

    gen_time = round(time.time() - gen_start, 2)
    total_time = round(retrieval_time + gen_time, 2)
    tps = round(token_count / gen_time, 1) if gen_time > 0 else 0

    print(f"\n\n   [Retrieval: {retrieval_time}s | Generation: {gen_time}s | "
          f"Total: {total_time}s | {tps} tok/s]")

    return full_reply


def interactive_streaming_rag():
    """Interactive RAG session with streaming answers."""
    print(f"\n📖 Streaming RAG Assistant")
    print(f"   Model: {MODEL}")
    print("   Ask questions about your knowledge base.")
    print("   Commands: 'sources on/off', 'quit'\n")
    print("-" * 55)

    show_sources = True

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break

        if question.lower() == "sources on":
            show_sources = True
            print("Sources: ON")
            continue

        if question.lower() == "sources off":
            show_sources = False
            print("Sources: OFF")
            continue

        stream_rag_answer(
            question=question,
            show_sources=show_sources
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Streaming RAG")
    parser.add_argument("--question", type=str, help="Single question")
    parser.add_argument("--chat",     action="store_true",
                        help="Interactive streaming RAG session")
    parser.add_argument("--nosrc",    action="store_true",
                        help="Hide source attribution")
    args = parser.parse_args()

    if args.question:
        stream_rag_answer(args.question, show_sources=not args.nosrc)
    elif args.chat:
        interactive_streaming_rag()
    else:
        interactive_streaming_rag()