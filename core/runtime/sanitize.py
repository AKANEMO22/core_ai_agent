"""Credential scrubbing for user-visible errors, events and logs."""

import re


def sanitize_secrets(text: str) -> str:
    """Mask API keys, bearer tokens, and secrets from user-visible errors and metrics.

    Args:
        text: Raw input message or error text.

    Returns:
        str: Scrubbed text with sensitive credentials redacted.
    """
    if not isinstance(text, str):
        return text

    sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{10,}", "[REDACTED_API_KEY]", text)
    sanitized = re.sub(
        r"(Bearer\s+)[a-zA-Z0-9_\-\.]{10,}",
        r"\1[REDACTED_TOKEN]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{8,})['\"]?",
        r"\1=[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized
