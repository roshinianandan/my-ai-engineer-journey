import ollama
import argparse
from vector_store import get_or_create_collection, collection_stats, list_collections
from indexer import COLLECTION_NAME

def get_embedding(text: str) -> list:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def search(query: str, top_k: int = 5, topic_filter: str = None, level_filter: str = None):
    """
    Search the ChromaDB collection.
    Optionally filter by topic or level metadata.
    """
    collection = get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        print("No documents indexed yet. Run: python indexer.py")
        return []

    query_embedding = get_embedding(query)

    # Build metadata filter if provided
    where = {}
    if topic_filter and level_filter:
        where = {"$and": [{"topic": topic_filter}, {"level": level_filter}]}
    elif topic_filter:
        where = {"topic": topic_filter}
    elif level_filter:
        where = {"level": level_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where if where else None,
        include=["documents", "distances", "metadatas"]
    )

    print(f"\n🔍 Query: '{query}'")
    if topic_filter:
        print(f"   Filter — topic: {topic_filter}")
    if level_filter:
        print(f"   Filter — level: {level_filter}")
    print(f"   Top {top_k} results from {collection.count()} indexed documents\n")
    print("-" * 60)

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    output = []
    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
        # ChromaDB returns distance not similarity — convert to similarity
        similarity = 1 - dist
        bar = "█" * int(similarity * 20)
        print(f"  {i+1}. Score: {similarity:.4f}  {bar}")
        print(f"     Text: {doc}")
        print(f"     Tags: topic={meta['topic']}  level={meta['level']}\n")
        output.append({"score": similarity, "text": doc, "metadata": meta})

    return output


def interactive_search():
    """Run an interactive search session."""
    print("\n🔍 ChromaDB Semantic Search")
    print("   Commands: 'filter topic=nlp', 'filter level=beginner', 'stats', 'quit'\n")

    topic_filter = None
    level_filter = None

    while True:
        query = input("Search: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            break

        if query.lower() == "stats":
            collection_stats(COLLECTION_NAME)
            print(f"Collections: {list_collections()}")
            continue

        if query.lower().startswith("filter topic="):
            topic_filter = query.split("=")[1].strip()
            print(f"  Topic filter set to: {topic_filter}")
            continue

        if query.lower().startswith("filter level="):
            level_filter = query.split("=")[1].strip()
            print(f"  Level filter set to: {level_filter}")
            continue

        if query.lower() == "clear filters":
            topic_filter = None
            level_filter = None
            print("  Filters cleared.")
            continue

        search(query, top_k=3, topic_filter=topic_filter, level_filter=level_filter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the ChromaDB vector store")
    parser.add_argument("--query",  type=str, help="Search query")
    parser.add_argument("--topk",   type=int, default=5, help="Number of results")
    parser.add_argument("--topic",  type=str, help="Filter by topic")
    parser.add_argument("--level",  type=str, choices=["beginner", "intermediate", "advanced"], help="Filter by level")
    args = parser.parse_args()

    if args.query:
        search(args.query, top_k=args.topk, topic_filter=args.topic, level_filter=args.level)
    else:
        interactive_search()