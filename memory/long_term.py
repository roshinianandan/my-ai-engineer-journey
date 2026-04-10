import json
import ollama
from datetime import datetime
from vector_store import get_or_create_collection

MEMORY_COLLECTION = "long_term_memory"


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


class LongTermMemory:
    """
    Persistent memory that survives across sessions.
    Stores important facts as embeddings in ChromaDB.
    Retrieves relevant memories using semantic search.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.collection = get_or_create_collection(MEMORY_COLLECTION)

    def save(self, fact: str, category: str = "general"):
        """
        Save an important fact to long-term memory.
        Each fact is embedded and stored with metadata.
        """
        memory_id = f"{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        embedding = get_embedding(fact)

        self.collection.upsert(
            documents=[fact],
            embeddings=[embedding],
            ids=[memory_id],
            metadatas=[{
                "user_id": self.user_id,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "fact": fact
            }]
        )
        print(f"[Memory saved: {fact[:60]}...]")

    def recall(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve the most relevant memories for a given query.
        Uses semantic search — finds related memories even without exact word match.
        """
        if self.collection.count() == 0:
            return []

        # Filter to only this user's memories
        try:
            results = self.collection.query(
                query_embeddings=[get_embedding(query)],
                n_results=min(top_k, self.collection.count()),
                where={"user_id": self.user_id},
                include=["documents", "distances", "metadatas"]
            )
        except Exception:
            return []

        memories = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            score = round(1 - dist, 4)
            if score > 0.3:  # only return relevant memories
                memories.append({
                    "fact": doc,
                    "category": meta.get("category", "general"),
                    "timestamp": meta.get("timestamp", ""),
                    "score": score
                })

        return memories

    def list_all(self) -> list:
        """List all memories for this user."""
        try:
            results = self.collection.get(
                where={"user_id": self.user_id},
                include=["documents", "metadatas"]
            )
        except Exception:
            return []

        memories = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            memories.append({
                "fact": doc,
                "category": meta.get("category", "general"),
                "timestamp": meta.get("timestamp", "")
            })
        return memories

    def forget(self, keyword: str) -> int:
        """Delete memories containing a keyword."""
        all_memories = self.list_all()
        deleted = 0

        all_items = self.collection.get(
            where={"user_id": self.user_id},
            include=["documents"]
        )

        ids_to_delete = []
        for doc_id, doc in zip(all_items["ids"], all_items["documents"]):
            if keyword.lower() in doc.lower():
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            deleted = len(ids_to_delete)
            print(f"[Forgotten {deleted} memories containing '{keyword}']")

        return deleted

    def clear_all(self):
        """Delete all memories for this user."""
        all_items = self.collection.get(where={"user_id": self.user_id})
        if all_items["ids"]:
            self.collection.delete(ids=all_items["ids"])
            print(f"[Cleared {len(all_items['ids'])} memories]")

    def stats(self) -> dict:
        all_memories = self.list_all()
        categories = {}
        for m in all_memories:
            cat = m["category"]
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_memories": len(all_memories),
            "categories": categories
        }