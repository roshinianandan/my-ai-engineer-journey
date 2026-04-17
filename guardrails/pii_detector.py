import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """A detected PII item with its type, value, and position."""
    pii_type: str
    value: str
    start: int
    end: int
    replacement: str


# PII detection patterns
PII_PATTERNS = {
    "email": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "replacement": "[EMAIL REDACTED]",
        "severity": "high"
    },
    "phone_india": {
        "pattern": r"\b(?:\+91|91)?[6-9]\d{9}\b",
        "replacement": "[PHONE REDACTED]",
        "severity": "high"
    },
    "phone_us": {
        "pattern": r"\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "replacement": "[PHONE REDACTED]",
        "severity": "high"
    },
    "credit_card": {
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "replacement": "[CARD REDACTED]",
        "severity": "critical"
    },
    "aadhaar": {
        "pattern": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "replacement": "[AADHAAR REDACTED]",
        "severity": "critical"
    },
    "pan_card": {
        "pattern": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "replacement": "[PAN REDACTED]",
        "severity": "critical"
    },
    "ip_address": {
        "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "replacement": "[IP REDACTED]",
        "severity": "medium"
    },
    "url_with_token": {
        "pattern": r"https?://[^\s]+(?:token|key|secret|password|auth)[^\s]*",
        "replacement": "[URL WITH CREDENTIALS REDACTED]",
        "severity": "critical"
    },
}


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information from text.

    Used in two places:
    1. Input validation — redact PII from user inputs before logging
    2. Output filtering — ensure LLM responses don't expose PII
    """

    def __init__(self, severity_threshold: str = "medium"):
        """
        severity_threshold: minimum severity to detect
        Options: low, medium, high, critical
        """
        self.severity_order = ["low", "medium", "high", "critical"]
        self.threshold_idx = self.severity_order.index(severity_threshold)

    def _should_check(self, severity: str) -> bool:
        """Check if this severity level should be detected."""
        return self.severity_order.index(severity) >= self.threshold_idx

    def detect(self, text: str) -> list[PIIMatch]:
        """
        Find all PII in text.
        Returns a list of PIIMatch objects.
        """
        matches = []

        for pii_type, config in PII_PATTERNS.items():
            if not self._should_check(config["severity"]):
                continue

            for match in re.finditer(config["pattern"], text, re.IGNORECASE):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    replacement=config["replacement"]
                ))

        # Sort by position (for correct sequential replacement)
        matches.sort(key=lambda m: m.start)
        return matches

    def redact(self, text: str) -> tuple[str, list[PIIMatch]]:
        """
        Replace all detected PII with safe placeholders.
        Returns (redacted_text, list_of_matches).
        """
        matches = self.detect(text)

        if not matches:
            return text, []

        # Replace from end to start to preserve positions
        redacted = text
        for match in reversed(matches):
            redacted = redacted[:match.start] + match.replacement + redacted[match.end:]

        return redacted, matches

    def has_pii(self, text: str) -> bool:
        """Quick check if text contains any PII."""
        return len(self.detect(text)) > 0

    def pii_report(self, text: str) -> dict:
        """Generate a PII detection report for a text."""
        matches = self.detect(text)
        redacted, _ = self.redact(text)

        return {
            "has_pii": len(matches) > 0,
            "pii_count": len(matches),
            "pii_types": list(set(m.pii_type for m in matches)),
            "redacted_text": redacted,
            "findings": [
                {
                    "type": m.pii_type,
                    "value_masked": m.value[:3] + "***",
                    "replacement": m.replacement
                }
                for m in matches
            ]
        }