"""Text preprocessing and cleaning utilities for skincare product reviews."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalize and clean review text for NLP feature extraction.

    Args:
        text: Raw review post string.

    Returns:
        Cleaned, normalized review string.
    """
    if not isinstance(text, str):
        return ""

    # Replace newlines and carriage returns with spaces
    cleaned = text.replace("\r", " ").replace("\n", " ")

    # Collapse multiple consecutive whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def normalize_label(label: str) -> str:
    """Standardize taxonomy label strings by trimming whitespace.

    Args:
        label: Raw label string.

    Returns:
        Stripped and normalized label string.
    """
    if not isinstance(label, str):
        return ""
    return label.strip()
