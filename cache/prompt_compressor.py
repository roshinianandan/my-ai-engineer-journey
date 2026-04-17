import re
import ollama
from config import MODEL


def count_tokens_estimate(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def remove_redundancy(text: str) -> str:
    """
    Remove obvious redundancy from text:
    - Repeated whitespace
    - Repeated punctuation
    - Common filler phrases
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove filler phrases
    fillers = [
        r"\bplease\b\s+",
        r"\bkindly\b\s+",
        r"\bcould you\b\s+",
        r"\bcan you\b\s+",
        r"\bwould you\b\s+",
        r"\bif you don't mind\b\s*",
        r"\bif possible\b\s*",
    ]
    for filler in fillers:
        text = re.sub(filler, "", text, flags=re.IGNORECASE)

    return text.strip()


def compress_context(context: str, max_tokens: int = 500) -> str:
    """
    Compress a long context block to fit within a token budget.
    Uses the LLM to extract only the most relevant information.

    This is used in RAG pipelines where retrieved chunks may be too long.
    """
    current_tokens = count_tokens_estimate(context)

    if current_tokens <= max_tokens:
        return context  # Already short enough

    print(f"  [Compressing context: {current_tokens} → ~{max_tokens} tokens]")

    prompt = f"""Extract only the most important facts from the text below.
Remove redundant or less important information.
Keep the result under {max_tokens * 4} characters.
Preserve key facts, numbers, and specific details.

Text to compress:
{context}

Compressed version:"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.1}
    )

    compressed = response["message"]["content"]
    compressed_tokens = count_tokens_estimate(compressed)

    reduction = round((1 - compressed_tokens / current_tokens) * 100, 1)
    print(f"  [Compression: {current_tokens} → {compressed_tokens} tokens ({reduction}% reduction)]")

    return compressed


def compress_prompt(
    prompt: str,
    system_prompt: str = "",
    max_total_tokens: int = 800
) -> dict:
    """
    Compress a full prompt to fit within a token budget.
    Compresses the longest part first.

    Returns dict with compressed components and stats.
    """
    original_tokens = count_tokens_estimate(prompt + system_prompt)

    # Step 1: Remove redundancy
    prompt_clean = remove_redundancy(prompt)
    system_clean = remove_redundancy(system_prompt)

    # Step 2: If still too long, compress with LLM
    current_tokens = count_tokens_estimate(prompt_clean + system_clean)

    if current_tokens > max_total_tokens and len(prompt_clean) > 200:
        budget = max_total_tokens - count_tokens_estimate(system_clean)
        prompt_clean = compress_context(prompt_clean, max_tokens=budget)

    final_tokens = count_tokens_estimate(prompt_clean + system_clean)
    reduction = round((1 - final_tokens / original_tokens) * 100, 1) if original_tokens > 0 else 0

    return {
        "compressed_prompt": prompt_clean,
        "compressed_system": system_clean,
        "original_tokens": original_tokens,
        "compressed_tokens": final_tokens,
        "reduction_percent": reduction,
        "was_compressed": reduction > 5
    }


def batch_compress(prompts: list, max_tokens: int = 500) -> list:
    """
    Compress a list of prompts.
    Returns compression stats for each.
    """
    results = []
    total_saved = 0

    print(f"\nBatch compressing {len(prompts)} prompts...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Compressing...")
        result = compress_prompt(prompt, max_total_tokens=max_tokens)
        total_saved += result["original_tokens"] - result["compressed_tokens"]
        results.append(result)

    print(f"\nTotal tokens saved: ~{total_saved}")
    return results


if __name__ == "__main__":
    sample_long = """
    I was wondering if you could please help me understand,
    if you don't mind, what exactly machine learning is and how it works,
    and also could you kindly explain the difference between supervised
    and unsupervised learning if possible, and also maybe give me some
    examples of real world applications where machine learning is being
    used today in various industries like healthcare, finance, retail,
    and transportation, and if you could also explain what deep learning
    is and how it relates to machine learning that would be very helpful
    and appreciated.
    """

    print(f"Original: {count_tokens_estimate(sample_long)} tokens")
    print(f"Original text:\n{sample_long[:200]}...\n")

    result = compress_prompt(sample_long)
    print(f"\nCompressed: {result['compressed_tokens']} tokens")
    print(f"Reduction: {result['reduction_percent']}%")
    print(f"Compressed text:\n{result['compressed_prompt']}")