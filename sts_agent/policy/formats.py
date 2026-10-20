"""Prompt presentation formats, independent of section-order templates."""

from __future__ import annotations


DEFAULT_PROMPT_FORMAT = "plain_text"
PROMPT_FORMATS = {"plain_text": "available", "markdown": "under_development"}


def require_prompt_format(value: str) -> str:
    """Reject pending formats before generating requests or loading services."""
    if not isinstance(value, str) or value not in PROMPT_FORMATS:
        raise ValueError(f"unknown prompt format: {value!r}; choose plain_text or markdown")
    if PROMPT_FORMATS[value] != "available":
        raise ValueError("markdown prompt format is under development (待开发); use plain_text")
    return value
