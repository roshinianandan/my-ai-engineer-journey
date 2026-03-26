import os
import argparse
from rag.chunker import load_documents
from rag.retriever import index_chunks, retrieve
from rag.generator import generate_answer
from vector_store import get_or_create_collection, collection_stats

DOCS_FOLDER = "./data/docs"
RAG_COLLECTION = "rag_documents"


def ingest(docs_folder: str = DOCS_FOLDER):
    """
    Full ingestion pipeline:
    Load documents → chunk → embed → store in ChromaDB
    """
    print("=" * 55)
    print("  RAG INGESTION PIPELINE")
    print("=" * 55)

    documents = load_documents(docs_folder)

    if not documents:
        print("No documents found. Add PDF or TXT files to data/docs/")
        return

    print(f"\nLoaded {len(documents)} document(s). Starting indexing...\n")
    total = index_chunks(documents)
    print(f"\nIngestion complete. {total} chunks ready for search.")
    collection_stats(RAG_COLLECTION)


def ask(query: str, top_k: int = 3, show_sources: bool = True):
    """
    Full RAG query pipeline:
    Embed query → retrieve chunks → generate grounded answer
    """
    print(f"\n{'='*55}")
    print(f"  QUERY: {query}")
    print(f"{'='*55}")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        print("No relevant chunks found.")
        return

    # Step 2: Show sources (optional)
    if show_sources:
        print(f"\n📚 Retrieved {len(chunks)} relevant chunk(s):\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] Score: {chunk['score']} | Source: {chunk['source']}")
            print(f"      {chunk['text'][:100]}...\n")

    # Step 3: Generate grounded answer
    generate_answer(query, chunks)


def interactive(top_k: int = 3):
    """Run an interactive Q&A session over your documents."""
    collection = get_or_create_collection(RAG_COLLECTION)
    count = collection.count()

    if count == 0:
        print("No documents indexed. Run: python -m rag.pipeline --ingest")
        return

    print(f"\n📖 RAG Assistant ready — {count} chunks indexed.")
    print("   Ask anything about your documents. Type 'quit' to exit.\n")
    print("-" * 55)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        ask(query, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--ingest",  action="store_true", help="Ingest documents from data/docs/")
    parser.add_argument("--ask",     type=str, help="Ask a one-shot question")
    parser.add_argument("--chat",    action="store_true", help="Start interactive Q&A session")
    parser.add_argument("--topk",    type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--nosrc",   action="store_true", help="Hide source attribution")
    args = parser.parse_args()

    if args.ingest:
        ingest()
    elif args.ask:
        ask(args.ask, top_k=args.topk, show_sources=not args.nosrc)
    elif args.chat:
        interactive(top_k=args.topk)
    else:
        parser.print_help()
