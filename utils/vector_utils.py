import math

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Measure how similar two vectors are.
    Returns a score between 0.0 (unrelated) and 1.0 (identical meaning).
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def top_k_similar(query_vec: list, candidates: list, texts: list, k: int = 3) -> list:
    """
    Given a query vector, find the top-k most similar vectors.
    Returns a sorted list of (score, text) tuples.
    """
    scored = []
    for vec, text in zip(candidates, texts):
        score = cosine_similarity(query_vec, vec)
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def normalize(vec: list) -> list:
    """Normalize a vector to unit length."""
    magnitude = math.sqrt(sum(a * a for a in vec))
    if magnitude == 0:
        return vec
    return [a / magnitude for a in vec]