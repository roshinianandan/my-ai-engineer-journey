import re
import ollama
from dataclasses import dataclass
from guardrails.pii_detector import PIIDetector
from config import MODEL


@dataclass
class FilterResult:
    """Result of output filtering."""
    is_safe: bool
    original_output: str
    filtered_output: str
    violations: list[str]
    pii_redacted: bool
    safety_score: float    # 0.0 = unsafe, 1.0 = safe


# Patterns that should never appear in outputs
OUTPUT_BLOCK_PATTERNS = [
    r"(?:here\s+is\s+how\s+to\s+)?(?:make|create|build)\s+(?:a\s+)?(?:bomb|explosive)",
    r"step[s]?\s+to\s+(?:hack|exploit|attack)",
    r"(?:child|minor)\s+(?:abuse|exploitation)",
]

# Patterns that need redaction in outputs
OUTPUT_REDACT_PATTERNS = {
    r"my\s+(?:password|secret|token|key)\s+is\s+\S+": "[CREDENTIAL REDACTED]",
    r"api[_\s]?key[:\s]+[A-Za-z0-9_-]{20,}": "[API KEY REDACTED]",
}


class OutputFilter:
    """
    Filters LLM outputs before they reach the user.

    Checks:
    1. Harmful content in LLM response
    2. PII leakage in LLM response
    3. Credential exposure
    4. LLM-as-judge safety scoring
    """

    def __init__(
        self,
        redact_pii: bool = True,
        use_llm_judge: bool = False
    ):
        self.redact_pii = redact_pii
        self.use_llm_judge = use_llm_judge
        self.pii_detector = PIIDetector(severity_threshold="medium")

    def filter(self, output: str, original_query: str = "") -> FilterResult:
        """
        Run all output filters on LLM response.
        Returns FilterResult with safety assessment.
        """
        violations = []
        filtered = output
        pii_redacted = False

        # Check 1: Harmful content in output
        for pattern in OUTPUT_BLOCK_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                violations.append("Harmful content detected in LLM output")
                filtered = "[Response blocked due to policy violation]"
                return FilterResult(
                    is_safe=False,
                    original_output=output,
                    filtered_output=filtered,
                    violations=violations,
                    pii_redacted=False,
                    safety_score=0.0
                )

        # Check 2: Redact sensitive patterns
        for pattern, replacement in OUTPUT_REDACT_PATTERNS.items():
            if re.search(pattern, filtered, re.IGNORECASE):
                filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
                violations.append("Credentials detected and redacted in output")

        # Check 3: PII in output
        if self.redact_pii:
            redacted, pii_matches = self.pii_detector.redact(filtered)
            if pii_matches:
                filtered = redacted
                pii_redacted = True
                violations.append(
                    f"PII redacted from output: "
                    f"{', '.join(set(m.pii_type for m in pii_matches))}"
                )

        # Check 4: LLM-as-judge (optional, slower)
        safety_score = 1.0
        if self.use_llm_judge and original_query:
            safety_score = self._llm_safety_score(original_query, filtered)
            if safety_score < 0.5:
                violations.append(
                    f"LLM judge flagged response as potentially unsafe (score: {safety_score})"
                )

        is_safe = len([v for v in violations
                       if "blocked" in v.lower()]) == 0

        return FilterResult(
            is_safe=is_safe,
            original_output=output,
            filtered_output=filtered,
            violations=violations,
            pii_redacted=pii_redacted,
            safety_score=safety_score
        )

    def _llm_safety_score(self, query: str, response: str) -> float:
        """
        Use a second LLM call to score the safety of a response.
        Returns a score from 0.0 (unsafe) to 1.0 (safe).
        This is the LLM-as-judge pattern applied to safety.
        """
        prompt = f"""Rate the safety of this AI response on a scale from 0 to 10.
Consider: harmful content, misinformation, privacy violations, bias.
10 = completely safe and helpful
0 = harmful, dangerous, or clearly violates safety guidelines

Query: {query[:200]}
Response: {response[:500]}

Return ONLY a number from 0 to 10. No explanation."""

        try:
            result = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.0}
            )
            raw = result["message"]["content"].strip()
            score = float(re.search(r"\d+(?:\.\d+)?", raw).group())
            return min(1.0, max(0.0, score / 10.0))
        except Exception:
            return 1.0  # Default to safe if judge fails