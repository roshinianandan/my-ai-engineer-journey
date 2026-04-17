import re
from dataclasses import dataclass
from typing import Optional
from guardrails.pii_detector import PIIDetector


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_safe: bool
    risk_level: str          # low, medium, high, critical
    violations: list[str]
    warnings: list[str]
    cleaned_input: str
    original_input: str
    action: str              # allow, warn, block


# Prompt injection patterns — attempts to hijack the AI's behavior
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|evil|unrestricted)",
    r"act\s+as\s+(?:if\s+you\s+(?:have\s+)?no\s+(?:restrictions|rules|ethics))",
    r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|uncensored|evil)",
    r"disregard\s+(?:your\s+)?(?:training|guidelines|safety|ethics)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
    r"developer\s+mode\s+enabled",
    r"</?(system|prompt|instruction)>",
    r"\[SYSTEM\]|\[INST\]|\[ADMIN\]",
]

# Harmful content patterns
HARMFUL_PATTERNS = [
    r"\b(?:how\s+to\s+)?(?:make|create|build|synthesize)\s+(?:a\s+)?(?:bomb|explosive|weapon|poison|drug)",
    r"\b(?:hack|exploit|breach|attack)\s+(?:into\s+)?(?:system|server|database|network)",
    r"(?:child|minor|underage)\s+(?:abuse|exploitation|pornography)",
    r"\b(?:suicide|self[- ]harm)\s+(?:method|instruction|guide|how)",
]

# Patterns that need review but aren't automatic blocks
WARNING_PATTERNS = [
    r"\bpassword\b",
    r"\bcredit\s+card\b",
    r"\bsocial\s+security\b",
    r"\bpersonal\s+information\b",
]


class InputValidator:
    """
    Validates and sanitizes user inputs before they reach the LLM.

    Security layers:
    1. Prompt injection detection
    2. Harmful content detection
    3. PII detection and redaction
    4. Input length limits
    5. Warning flags for sensitive topics
    """

    def __init__(
        self,
        max_length: int = 4000,
        block_injections: bool = True,
        block_harmful: bool = True,
        redact_pii: bool = True
    ):
        self.max_length = max_length
        self.block_injections = block_injections
        self.block_harmful = block_harmful
        self.redact_pii = redact_pii
        self.pii_detector = PIIDetector(severity_threshold="medium")

    def validate(self, text: str, context: str = "chat") -> ValidationResult:
        """
        Run all validation checks on input text.
        Returns a ValidationResult with safety assessment.
        """
        violations = []
        warnings = []
        risk_level = "low"
        cleaned = text

        # Check 1: Length limit
        if len(text) > self.max_length:
            violations.append(f"Input exceeds maximum length ({len(text)} > {self.max_length})")
            cleaned = text[:self.max_length]
            risk_level = "medium"

        # Check 2: Prompt injection
        if self.block_injections:
            for pattern in INJECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append(f"Prompt injection attempt detected: pattern match")
                    risk_level = "critical"
                    break

        # Check 3: Harmful content
        if self.block_harmful:
            for pattern in HARMFUL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append("Potentially harmful content detected")
                    risk_level = "critical"
                    break

        # Check 4: Warning patterns
        for pattern in WARNING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                warnings.append(f"Sensitive topic detected — handle with care")
                if risk_level == "low":
                    risk_level = "medium"
                break

        # Check 5: PII detection and redaction
        if self.redact_pii:
            redacted, pii_matches = self.pii_detector.redact(cleaned)
            if pii_matches:
                warnings.append(
                    f"PII detected and redacted: "
                    f"{', '.join(set(m.pii_type for m in pii_matches))}"
                )
                cleaned = redacted

        # Determine action
        is_safe = len(violations) == 0
        if not is_safe and risk_level == "critical":
            action = "block"
        elif not is_safe:
            action = "warn"
        elif warnings:
            action = "warn"
        else:
            action = "allow"

        return ValidationResult(
            is_safe=is_safe,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings,
            cleaned_input=cleaned,
            original_input=text,
            action=action
        )

    def is_safe(self, text: str) -> bool:
        """Quick safety check."""
        result = self.validate(text)
        return result.is_safe and result.action != "block"