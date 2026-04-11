import math
from rouge_score import rouge_scorer
import ollama


def rouge_scores(generated: str, expected: str) -> dict:
    """
    Calculate ROUGE scores between generated and expected answer.

    ROUGE-1: Overlap of individual words (unigrams)
    ROUGE-2: Overlap of word pairs (bigrams)
    ROUGE-L: Longest common subsequence

    Each score has precision, recall, and F1.
    F1 is the harmonic mean of precision and recall — the main score to watch.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(expected, generated)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def get_embedding(text: str) -> list:
    """Get embedding from Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def semantic_similarity(generated: str, expected: str) -> float:
    """
    Measure semantic similarity between generated and expected answers.
    Uses cosine similarity on embeddings — captures meaning not just words.
    A score of 1.0 means identical meaning, 0.0 means completely unrelated.
    """
    vec1 = get_embedding(generated)
    vec2 = get_embedding(expected)

    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return round(dot / (mag1 * mag2), 4)


def length_ratio(generated: str, expected: str) -> float:
    """
    Compare length of generated vs expected answer.
    A ratio close to 1.0 means similar length.
    Very low means too short. Very high means too verbose.
    """
    gen_words = len(generated.split())
    exp_words = len(expected.split())
    if exp_words == 0:
        return 0.0
    return round(gen_words / exp_words, 2)


def score_answer(generated: str, expected: str) -> dict:
    """
    Run all metrics on a generated answer against the expected answer.
    Returns a combined score report.
    """
    rouge = rouge_scores(generated, expected)
    sem_sim = semantic_similarity(generated, expected)
    length = length_ratio(generated, expected)

    # Combined score — weighted average
    combined = round(
        (rouge["rouge1"] * 0.2) +
        (rouge["rougeL"] * 0.2) +
        (sem_sim * 0.6),
        4
    )

    return {
        "rouge1":    rouge["rouge1"],
        "rouge2":    rouge["rouge2"],
        "rougeL":    rouge["rougeL"],
        "semantic":  sem_sim,
        "length_ratio": length,
        "combined":  combined,
        "grade": grade(combined)
    }


def grade(score: float) -> str:
    """Convert a combined score to a letter grade."""
    if score >= 0.85:
        return "A"
    elif score >= 0.70:
        return "B"
    elif score >= 0.55:
        return "C"
    elif score >= 0.40:
        return "D"
    else:
        return "F"