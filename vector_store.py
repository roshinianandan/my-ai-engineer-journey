import chromadb
from chromadb.config import Settings

# This is the single source of truth for the ChromaDB client
# All other files import from here

CHROMA_PATH = "./chroma_db"

def get_client():
    """Get a persistent ChromaDB client that saves to disk."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client

def get_or_create_collection(name: str):
    """Get an existing collection or create a new one."""
    client = get_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    return collection

def delete_collection(name: str):
    """Delete a collection entirely."""
    client = get_client()
    try:
        client.delete_collection(name)
        print(f"Collection '{name}' deleted.")
    except Exception as e:
        print(f"Could not delete collection: {e}")

def list_collections():
    """List all collections in the database."""
    client = get_client()
    collections = client.list_collections()
    return [c.name for c in collections]

def collection_stats(name: str):
    """Show how many items are in a collection."""
    collection = get_or_create_collection(name)
    count = collection.count()
    print(f"Collection '{name}' has {count} documents.")
    return count