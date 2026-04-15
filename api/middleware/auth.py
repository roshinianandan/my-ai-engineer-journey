import os
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production this would come from a database or secret manager
# For local dev we use environment variable or a hardcoded dev key
DEV_API_KEY = os.getenv("API_KEY", "dev-key-aiml-journey-2024")

# Multiple valid keys — useful for different clients or environments
VALID_API_KEYS = {
    DEV_API_KEY,
    "test-key-12345"  # for testing
}


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Verify the API key from the X-API-Key header.
    Returns the API key if valid, raises 403 if not.

    In production: hash and store keys in a database.
    Rate limit per key. Log all requests with key ID.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No API key provided. Add X-API-Key header."
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )

    return api_key


def generate_api_key() -> str:
    """Generate a new secure API key."""
    return secrets.token_urlsafe(32)