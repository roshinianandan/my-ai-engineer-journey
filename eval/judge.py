import ollama
from config import MODEL


JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an AI-generated answer.

QUESTION: {question}

EXPECTED ANSWER: {expected}

GENERATED ANSWER: {generated}

Evaluate the generated answer on these criteria:
1. Accuracy — Is the information correct and factually accurate?
2. Completeness — Does it cover all key points from the expected answer?
3. Clarity — Is it clear and easy to understand?
4. Conciseness — Is it appropriately concise without unnecessary padding?

Respond in EXACTLY this format:
ACCURACY: [score 1-10]
COMPLETENESS: [score 1-10]
CLARITY: [score 1-10]
CONCISENESS: [score 1-10]
OVERALL: [score 1-10]
FEEDBACK: [one sentence explaining the main strength or weakness]"""


def llm_judge(question: str, expected: str, generated: str) -> dict:
    """
    Use an LLM to judge the quality of a generated answer.
    This is the LLM-as-judge pattern — using AI to evaluate AI.
    Returns structured scores and feedback.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        generated=generated
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.1}
    )

    raw = response["message"]["content"]
    return parse_judge_output(raw)


def parse_judge_output(raw: str) -> dict:
    """Parse the structured output from the LLM judge."""
    result = {
        "accuracy": 0,
        "completeness": 0,
        "clarity": 0,
        "conciseness": 0,
        "overall": 0,
        "feedback": "",
        "raw": raw
    }

    lines = raw.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("ACCURACY:"):
            result["accuracy"] = _extract_score(line)
        elif line.startswith("COMPLETENESS:"):
            result["completeness"] = _extract_score(line)
        elif line.startswith("CLARITY:"):
            result["clarity"] = _extract_score(line)
        elif line.startswith("CONCISENESS:"):
            result["conciseness"] = _extract_score(line)
        elif line.startswith("OVERALL:"):
            result["overall"] = _extract_score(line)
        elif line.startswith("FEEDBACK:"):
            result["feedback"] = line.replace("FEEDBACK:", "").strip()

    return result


def _extract_score(line: str) -> int:
    """Extract integer score from a line like 'ACCURACY: 8'."""
    try:
        parts = line.split(":")
        score_str = parts[-1].strip().split()[0]
        score = int(score_str)
        return max(1, min(10, score))
    except Exception:
        return 0