import ollama
from vector_store import get_or_create_collection

RAG_COLLECTION = "rag_documents"


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def index_chunks(documents: list):
    """
    Embed all chunks from all documents and store in ChromaDB.
    Each chunk gets a unique ID and metadata about its source file.
    """
    collection = get_or_create_collection(RAG_COLLECTION)

    all_texts = []
    all_embeddings = []
    all_ids = []
    all_metadatas = []

    chunk_counter = 0
    for doc in documents:
        filename = doc["filename"]
        for i, chunk in enumerate(doc["chunks"]):
            chunk_id = f"{filename}_chunk_{i}"
            embedding = get_embedding(chunk)

            all_texts.append(chunk)
            all_embeddings.append(embedding)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": filename,
                "chunk_index": str(i),
                "total_chunks": str(doc["num_chunks"])
            })
            chunk_counter += 1
            print(f"  Indexed chunk {chunk_counter}: {chunk[:50]}...")

    collection.upsert(
        documents=all_texts,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(f"\nTotal chunks indexed: {chunk_counter}")
    return chunk_counter


def retrieve(query: str, top_k: int = 3) -> list:
    """
    Retrieve the most relevant chunks for a query.
    Returns a list of dicts with text, source, and score.
    """
    collection = get_or_create_collection(RAG_COLLECTION)

    if collection.count() == 0:
        print("No documents indexed. Run the pipeline first.")
        return []

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
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
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "score": round(1 - dist, 4)
        })

    return chunks