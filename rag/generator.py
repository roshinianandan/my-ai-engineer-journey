import ollama
from config import MODEL


def build_context(chunks: list) -> str:
    """Format retrieved chunks into a readable context block."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']} | Relevance: {chunk['score']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, chunks: list, stream: bool = True) -> str:
    """
    Generate a grounded answer using retrieved chunks as context.
    This is the 'G' in RAG — Retrieval Augmented GENERATION.
    """
    if not chunks:
        return "I could not find relevant information to answer your question."

    context = build_context(chunks)

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Do NOT make up information. Always cite which source you used.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (based only on the context above):"""

    messages = [{"role": "user", "content": prompt}]

    if stream:
        print("\n🤖 Answer: ", end="", flush=True)
        full_reply = ""
        for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_reply += token
        print("\n")
        return full_reply
    else:
        response = ollama.chat(model=MODEL, messages=messages, stream=False)
        return response["message"]["content"]