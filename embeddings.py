import ollama
import argparse
from utils.vector_utils import cosine_similarity, top_k_similar

# A small knowledge base of sentences to search through
KNOWLEDGE_BASE = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Python is the most popular language for data science.",
    "Natural language processing helps computers understand text.",
    "Transformers are the architecture behind modern LLMs.",
    "Gradient descent is used to train neural networks.",
    "A dataset is a collection of examples used for training.",
    "Overfitting happens when a model memorizes training data.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Computer vision teaches machines to interpret images.",
    "Embeddings represent text as high-dimensional vectors.",
    "Cosine similarity measures the angle between two vectors.",
    "RAG stands for Retrieval Augmented Generation.",
    "An API is a way for programs to talk to each other.",
    "A neural network is inspired by the structure of the human brain.",
    "Supervised learning uses labeled data to train models.",
    "Unsupervised learning finds patterns in unlabeled data.",
    "A token is the smallest unit of text processed by an LLM.",
    "Fine-tuning adapts a pretrained model to a specific task.",
    "Vector databases store and search embeddings efficiently.",
]


def get_embedding(text: str) -> list:
    """Get the embedding vector for a piece of text using Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def build_index(sentences: list) -> list:
    """Embed all sentences in the knowledge base."""
    print(f"Building index for {len(sentences)} sentences...")
    vectors = []
    for i, sentence in enumerate(sentences):
        vec = get_embedding(sentence)
        vectors.append(vec)
        print(f"  Indexed {i+1}/{len(sentences)}: {sentence[:50]}...")
    print("Index ready.\n")
    return vectors


def semantic_search(query: str, vectors: list, sentences: list, top_k: int = 3):
    """Find the most semantically similar sentences to a query."""
    print(f"\nQuery: {query}")
    print("-" * 50)

    query_vec = get_embedding(query)
    results = top_k_similar(query_vec, vectors, sentences, k=top_k)

    print(f"Top {top_k} most similar sentences:\n")
    for rank, (score, text) in enumerate(results, 1):
        bar = "█" * int(score * 20)
        print(f"  {rank}. Score: {score:.4f}  {bar}")
        print(f"     {text}\n")

    return results


def compare_two(text1: str, text2: str):
    """Directly compare the similarity of two sentences."""
    print(f"\nComparing two sentences:")
    print(f"  A: {text1}")
    print(f"  B: {text2}")
    print("-" * 50)

    vec1 = get_embedding(text1)
    vec2 = get_embedding(text2)
    score = cosine_similarity(vec1, vec2)

    bar = "█" * int(score * 30)
    print(f"  Similarity score: {score:.4f}")
    print(f"  {bar}")

    if score > 0.9:
        print("  Interpretation: Nearly identical meaning")
    elif score > 0.7:
        print("  Interpretation: Very similar meaning")
    elif score > 0.5:
        print("  Interpretation: Somewhat related")
    else:
        print("  Interpretation: Different topics")


def interactive_search(vectors: list):
    """Let the user search the knowledge base interactively."""
    print("\n🔍 Semantic Search Ready — type a query to search the knowledge base.")
    print("   Type 'compare' to compare two sentences directly.")
    print("   Type 'quit' to exit.\n")

    while True:
        user_input = input("Search: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "compare":
            text1 = input("  Enter sentence A: ").strip()
            text2 = input("  Enter sentence B: ").strip()
            compare_two(text1, text2)
        else:
            semantic_search(user_input, vectors, KNOWLEDGE_BASE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search with Embeddings")
    parser.add_argument("--query",   type=str, help="Search query")
    parser.add_argument("--compare", nargs=2,  metavar=("TEXT1", "TEXT2"), help="Compare two sentences")
    parser.add_argument("--topk",    type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    # Pull the embedding model first time
    print("Loading embedding model...")

    # Build the index
    vectors = build_index(KNOWLEDGE_BASE)

    if args.compare:
        compare_two(args.compare[0], args.compare[1])
    elif args.query:
        semantic_search(args.query, vectors, KNOWLEDGE_BASE, top_k=args.topk)
    else:
        interactive_search(vectors)