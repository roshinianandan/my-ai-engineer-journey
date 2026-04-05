import ollama
from vector_store import get_or_create_collection
from rag.chunker import load_documents, chunk_text

KB_COLLECTION = "knowledge_base"
DOCS_FOLDER = "./data/docs"


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def ingest_folder(
    docs_folder: str = DOCS_FOLDER,
    strategy: str = "sentence",
    force: bool = False
):
    """
    Ingest all documents from a folder into the knowledge base.
    Each document gets automatic metadata tags.
    """
    collection = get_or_create_collection(KB_COLLECTION)
    documents = load_documents(docs_folder, strategy=strategy)

    if not documents:
        print("No documents found.")
        return

    print(f"\nIngesting {len(documents)} document(s) using '{strategy}' chunking...\n")

    total_chunks = 0
    for doc in documents:
        # Check if already indexed
        existing = collection.get(
            where={"source": doc["filename"]}
        )
        if existing["ids"] and not force:
            print(f"Skipping '{doc['filename']}' — already indexed "
                  f"({len(existing['ids'])} chunks). Use force=True to re-index.")
            continue

        # Delete old version if force re-indexing
        if existing["ids"] and force:
            collection.delete(ids=existing["ids"])
            print(f"Removed old version of '{doc['filename']}'")

        texts, embeddings, ids, metadatas = [], [], [], []

        for i, chunk in enumerate(doc["chunks"]):
            chunk_id = f"{doc['name']}_chunk_{i}"
            embedding = get_embedding(chunk)
            texts.append(chunk)
            embeddings.append(embedding)
            ids.append(chunk_id)
            metadatas.append({
                **doc["metadata"],
                "chunk_index": str(i),
                "total_chunks": str(doc["num_chunks"])
            })
            total_chunks += 1
            print(f"  [{doc['filename']}] chunk {i+1}/{doc['num_chunks']}: "
                  f"{chunk[:50]}...")

        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"  Indexed {doc['num_chunks']} chunks from '{doc['filename']}'\n")

    print(f"Done. Total new chunks added: {total_chunks}")
    print(f"Knowledge base now has {collection.count()} total chunks.")


def search(
    query: str,
    top_k: int = 3,
    source_filter: str = None,
    level_filter: str = None
) -> list:
    """
    Search the knowledge base with optional metadata filters.
    Combine semantic similarity with source or level filtering.
    """
    collection = get_or_create_collection(KB_COLLECTION)

    if collection.count() == 0:
        print("Knowledge base is empty. Run ingest_folder() first.")
        return []

    query_embedding = get_embedding(query)

    # Build filter
    where = {}
    if source_filter and level_filter:
        where = {"$and": [
            {"source": source_filter},
            {"level": level_filter}
        ]}
    elif source_filter:
        where = {"source": source_filter}
    elif level_filter:
        where = {"level": level_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where if where else None,
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
            "source": meta.get("source", "unknown"),
            "level": meta.get("level", "general"),
            "chunk_index": meta.get("chunk_index", "0"),
            "score": round(1 - dist, 4)
        })

    return chunks


def list_documents() -> list:
    """List all unique source documents in the knowledge base."""
    collection = get_or_create_collection(KB_COLLECTION)
    if collection.count() == 0:
        print("Knowledge base is empty.")
        return []

    all_items = collection.get(include=["metadatas"])
    sources = {}
    for meta in all_items["metadatas"]:
        src = meta.get("source", "unknown")
        if src not in sources:
            sources[src] = {
                "source": src,
                "level": meta.get("level", "general"),
                "chunks": 0
            }
        sources[src]["chunks"] += 1

    print(f"\n📚 Knowledge Base — {collection.count()} total chunks "
          f"from {len(sources)} document(s):\n")
    for src, info in sources.items():
        print(f"  {src}")
        print(f"    Level: {info['level']} | Chunks: {info['chunks']}\n")

    return list(sources.values())


def delete_document(filename: str):
    """Remove all chunks from a specific document."""
    collection = get_or_create_collection(KB_COLLECTION)
    existing = collection.get(where={"source": filename})

    if not existing["ids"]:
        print(f"Document '{filename}' not found in knowledge base.")
        return

    collection.delete(ids=existing["ids"])
    print(f"Deleted {len(existing['ids'])} chunks from '{filename}'.")


def answer(query: str, top_k: int = 3, source_filter: str = None,
           level_filter: str = None):
    """Full RAG answer with optional filters."""
    import ollama as ol
    from config import MODEL

    chunks = search(query, top_k=top_k,
                    source_filter=source_filter,
                    level_filter=level_filter)

    if not chunks:
        print("No relevant content found.")
        return

    print(f"\n📚 Retrieved {len(chunks)} chunk(s):\n")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] Score: {c['score']} | "
              f"Source: {c['source']} | Level: {c['level']}")
        print(f"      {c['text'][:100]}...\n")

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Score: {c['score']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present, say "I don't have that information."
Always mention which source you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    print("\n🤖 Answer: ", end="", flush=True)
    full_reply = ""
    for chunk in ol.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_reply += token
    print("\n")