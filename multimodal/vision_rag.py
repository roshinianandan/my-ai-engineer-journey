import ollama
import os
import json
from pathlib import Path
from multimodal.image_analyzer import encode_image_to_base64, get_image_info
from vector_store import get_or_create_collection

VISION_MODEL = "llava"
VISION_RAG_COLLECTION = "vision_rag"


def get_embedding(text: str) -> list:
    """Get text embedding from Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def describe_image(image_path: str) -> str:
    """
    Generate a detailed text description of an image using LLaVA.
    This description is what gets embedded and stored in ChromaDB.
    This is the core of Visual RAG — images become searchable text.
    """
    image_b64 = encode_image_to_base64(image_path)

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": """Describe this image in detail for a search system.
Include: all visible objects, people, text, colors, shapes, setting, and context.
Be thorough and specific — this description will be used for semantic search.""",
            "images": [image_b64]
        }]
    )

    return response["message"]["content"]


def index_image(image_path: str, force: bool = False) -> dict:
    """
    Index a single image into the visual RAG system:
    1. Generate text description using LLaVA
    2. Embed the description using nomic-embed-text
    3. Store in ChromaDB with image path as metadata
    """
    collection = get_or_create_collection(VISION_RAG_COLLECTION)
    image_id = Path(image_path).stem

    # Check if already indexed
    existing = collection.get(ids=[image_id])
    if existing["ids"] and not force:
        print(f"[Already indexed: {image_path}]")
        return {"image_path": image_path, "status": "already_indexed"}

    print(f"Describing image: {image_path}")
    description = describe_image(image_path)
    print(f"Description: {description[:100]}...")

    embedding = get_embedding(description)
    info = get_image_info(image_path)

    collection.upsert(
        documents=[description],
        embeddings=[embedding],
        ids=[image_id],
        metadatas=[{
            "image_path": str(image_path),
            "filename": Path(image_path).name,
            "width": str(info.get("width", "unknown")),
            "height": str(info.get("height", "unknown")),
            "format": info.get("format", "unknown")
        }]
    )

    return {
        "image_path": image_path,
        "image_id": image_id,
        "description": description,
        "status": "indexed"
    }


def index_folder(folder: str, force: bool = False) -> list:
    """Index all images in a folder into the visual RAG system."""
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    images = [f for f in folder_path.iterdir()
              if f.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {folder}")
        return []

    print(f"\nIndexing {len(images)} images...")
    results = []

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        result = index_image(str(image_path), force=force)
        results.append(result)

    collection = get_or_create_collection(VISION_RAG_COLLECTION)
    print(f"\nDone. {collection.count()} images indexed in visual RAG.")
    return results


def search_images(query: str, top_k: int = 3) -> list:
    """
    Search indexed images using semantic similarity.
    Returns images whose descriptions best match the query.
    """
    collection = get_or_create_collection(VISION_RAG_COLLECTION)

    if collection.count() == 0:
        print("No images indexed. Run index_folder() first.")
        return []

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"]
    )

    matches = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        matches.append({
            "image_path": meta.get("image_path"),
            "filename": meta.get("filename"),
            "score": round(1 - dist, 4),
            "description": doc[:200] + "..."
        })

    return matches


def visual_qa(query: str, top_k: int = 2) -> dict:
    """
    Full Visual RAG pipeline:
    1. Search for most relevant images
    2. Load those images
    3. Ask LLaVA to answer the query using those images
    """
    print(f"\nQuery: {query}")
    print("Searching visual knowledge base...")

    matches = search_images(query, top_k=top_k)

    if not matches:
        return {"query": query, "answer": "No relevant images found.", "images_used": []}

    print(f"Found {len(matches)} relevant image(s):")
    for m in matches:
        print(f"  {m['filename']} (score: {m['score']})")

    # Load images that exist on disk
    valid_images = [m for m in matches if m["image_path"] and
                    Path(m["image_path"]).exists()]

    if not valid_images:
        # Answer from descriptions only
        context = "\n\n".join(
            f"Image: {m['filename']}\n{m['description']}"
            for m in matches
        )
        response = ollama.chat(
            model="llama3.2",
            messages=[{
                "role": "user",
                "content": f"Based on these image descriptions, answer: {query}\n\nDescriptions:\n{context}"
            }]
        )
        return {
            "query": query,
            "answer": response["message"]["content"],
            "images_used": [m["filename"] for m in matches],
            "mode": "description_only"
        }

    # Answer using actual images
    images_b64 = [encode_image_to_base64(m["image_path"]) for m in valid_images]

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": f"Using the provided images, answer this question: {query}",
            "images": images_b64
        }]
    )

    answer = response["message"]["content"]
    print(f"\nAnswer: {answer}")

    return {
        "query": query,
        "answer": answer,
        "images_used": [m["filename"] for m in valid_images],
        "mode": "visual"
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visual RAG System")
    parser.add_argument("--index",  type=str, help="Index images from a folder")
    parser.add_argument("--search", type=str, help="Search indexed images by text query")
    parser.add_argument("--ask",    type=str, help="Ask a question using visual RAG")
    parser.add_argument("--force",  action="store_true", help="Force re-indexing")
    args = parser.parse_args()

    if args.index:
        index_folder(args.index, force=args.force)
    elif args.search:
        results = search_images(args.search)
        for r in results:
            print(f"\n{r['filename']} (score: {r['score']})")
            print(f"  {r['description']}")
    elif args.ask:
        result = visual_qa(args.ask)
        print(f"\nAnswer: {result['answer']}")
        print(f"Images used: {result['images_used']}")
    else:
        print("Usage:")
        print("  python multimodal/vision_rag.py --index data/images/")
        print("  python multimodal/vision_rag.py --search 'chart showing data'")
        print("  python multimodal/vision_rag.py --ask 'what diagrams show neural networks'")