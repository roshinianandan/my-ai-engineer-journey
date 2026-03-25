import ollama
import json
import os
from vector_store import get_or_create_collection, collection_stats

COLLECTION_NAME = "ai_knowledge_base"

# Expanded knowledge base with metadata tags
DOCUMENTS = [
    {"id": "doc_001", "text": "Machine learning is a subset of artificial intelligence.", "topic": "ml_basics", "level": "beginner"},
    {"id": "doc_002", "text": "Deep learning uses neural networks with many layers.", "topic": "deep_learning", "level": "beginner"},
    {"id": "doc_003", "text": "Python is the most popular language for data science.", "topic": "tools", "level": "beginner"},
    {"id": "doc_004", "text": "Natural language processing helps computers understand text.", "topic": "nlp", "level": "beginner"},
    {"id": "doc_005", "text": "Transformers are the architecture behind modern LLMs.", "topic": "deep_learning", "level": "intermediate"},
    {"id": "doc_006", "text": "Gradient descent is used to train neural networks.", "topic": "ml_basics", "level": "intermediate"},
    {"id": "doc_007", "text": "A dataset is a collection of examples used for training.", "topic": "ml_basics", "level": "beginner"},
    {"id": "doc_008", "text": "Overfitting happens when a model memorizes training data.", "topic": "ml_basics", "level": "intermediate"},
    {"id": "doc_009", "text": "Reinforcement learning trains agents through rewards.", "topic": "rl", "level": "intermediate"},
    {"id": "doc_010", "text": "Computer vision teaches machines to interpret images.", "topic": "cv", "level": "beginner"},
    {"id": "doc_011", "text": "Embeddings represent text as high-dimensional vectors.", "topic": "nlp", "level": "intermediate"},
    {"id": "doc_012", "text": "Cosine similarity measures the angle between two vectors.", "topic": "math", "level": "intermediate"},
    {"id": "doc_013", "text": "RAG stands for Retrieval Augmented Generation.", "topic": "llm", "level": "intermediate"},
    {"id": "doc_014", "text": "An API is a way for programs to communicate.", "topic": "tools", "level": "beginner"},
    {"id": "doc_015", "text": "A neural network is inspired by the human brain.", "topic": "deep_learning", "level": "beginner"},
    {"id": "doc_016", "text": "Supervised learning uses labeled data to train models.", "topic": "ml_basics", "level": "beginner"},
    {"id": "doc_017", "text": "Unsupervised learning finds patterns in unlabeled data.", "topic": "ml_basics", "level": "beginner"},
    {"id": "doc_018", "text": "A token is the smallest unit of text in an LLM.", "topic": "llm", "level": "beginner"},
    {"id": "doc_019", "text": "Fine-tuning adapts a pretrained model to a specific task.", "topic": "llm", "level": "advanced"},
    {"id": "doc_020", "text": "Vector databases store and search embeddings efficiently.", "topic": "tools", "level": "intermediate"},
    {"id": "doc_021", "text": "Attention mechanism allows models to focus on relevant parts.", "topic": "deep_learning", "level": "advanced"},
    {"id": "doc_022", "text": "Batch normalization helps stabilize neural network training.", "topic": "deep_learning", "level": "advanced"},
    {"id": "doc_023", "text": "Dropout is a regularization technique to prevent overfitting.", "topic": "deep_learning", "level": "intermediate"},
    {"id": "doc_024", "text": "A confusion matrix shows classification model performance.", "topic": "ml_basics", "level": "intermediate"},
    {"id": "doc_025", "text": "Transfer learning reuses knowledge from one task to another.", "topic": "ml_basics", "level": "intermediate"},
]


def get_embedding(text: str) -> list:
    """Get embedding vector from Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def index_all(force: bool = False):
    """
    Index all documents into ChromaDB.
    If force=True, re-index even if documents already exist.
    """
    collection = get_or_create_collection(COLLECTION_NAME)
    existing_count = collection.count()

    if existing_count >= len(DOCUMENTS) and not force:
        print(f"Already indexed {existing_count} documents. Use --force to re-index.")
        return

    print(f"Indexing {len(DOCUMENTS)} documents into ChromaDB...")
    print("This embeds each document once and saves to disk permanently.\n")

    texts = []
    embeddings = []
    ids = []
    metadatas = []

    for i, doc in enumerate(DOCUMENTS):
        embedding = get_embedding(doc["text"])
        texts.append(doc["text"])
        embeddings.append(embedding)
        ids.append(doc["id"])
        metadatas.append({"topic": doc["topic"], "level": doc["level"]})
        print(f"  [{i+1}/{len(DOCUMENTS)}] Indexed: {doc['text'][:55]}...")

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"\nDone! {len(DOCUMENTS)} documents saved to disk at ./chroma_db")
    collection_stats(COLLECTION_NAME)


def add_document(doc_id: str, text: str, topic: str = "general", level: str = "beginner"):
    """Add a single new document to the collection."""
    collection = get_or_create_collection(COLLECTION_NAME)
    embedding = get_embedding(text)
    collection.upsert(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[{"topic": topic, "level": level}]
    )
    print(f"Added document '{doc_id}': {text[:60]}")


def remove_document(doc_id: str):
    """Remove a document by its ID."""
    collection = get_or_create_collection(COLLECTION_NAME)
    collection.delete(ids=[doc_id])
    print(f"Removed document '{doc_id}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-index all documents")
    parser.add_argument("--add", nargs=3, metavar=("ID", "TEXT", "TOPIC"), help="Add a document")
    parser.add_argument("--remove", type=str, metavar="ID", help="Remove a document by ID")
    args = parser.parse_args()

    if args.add:
        add_document(args.add[0], args.add[1], args.add[2])
    elif args.remove:
        remove_document(args.remove)
    else:
        index_all(force=args.force)