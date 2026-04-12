from rag.knowledge_base import search as kb_search

# Tool schema
SEARCH_TOOL_SCHEMA = {
    "name": "search_knowledge_base",
    "description": "Search the local knowledge base for information about AI, machine learning, deep learning, Python, and related technical topics. Use this when the user asks about specific AI concepts, algorithms, or tools that might be in the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information in the knowledge base."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return. Default is 3.",
                "default": 3
            }
        },
        "required": ["query"]
    }
}


def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """
    Search the local ChromaDB knowledge base for relevant information.
    Returns the top matching chunks with their source documents.
    """
    try:
        results = kb_search(query, top_k=top_k)

        if not results:
            return {
                "query": query,
                "found": False,
                "message": "No relevant information found in the knowledge base.",
                "results": []
            }

        formatted = []
        for r in results:
            formatted.append({
                "text": r["text"],
                "source": r["source"],
                "relevance_score": r["score"]
            })

        return {
            "query": query,
            "found": True,
            "num_results": len(formatted),
            "results": formatted
        }

    except Exception as e:
        return {
            "query": query,
            "found": False,
            "error": str(e),
            "results": []
        }