"""
CareerRAG — UI Utilities
Helpers for parsing LLM output sections, extracting scores, and validation badges.
"""

from __future__ import annotations

import re


def parse_sections(answer_text: str) -> dict[str, str]:
    """
    Parse markdown-style ## sections into an ordered dict.

    Example input:
        ## Candidate Overview
        Some text here...
        ## Core Technical Skills
        More text...

    Returns:
        {"Candidate Overview": "Some text here...", "Core Technical Skills": "More text..."}
    """
    if not answer_text or not answer_text.strip():
        return {}

    pattern = r"##\s+(.+?)\n(.*?)(?=\n##\s+|\Z)"
    matches = re.findall(pattern, answer_text, flags=re.DOTALL)

    sections: dict[str, str] = {}
    for title, content in matches:
        sections[title.strip()] = content.strip()

    return sections


def extract_match_score(answer_text: str) -> int | None:
    """
    Extract numeric match score (0–100) from text.
    Handles patterns like: Match Score: 78, Match Score - 78, **Match Score**: 78/100
    """
    if not answer_text:
        return None

    match = re.search(r"Match Score\s*[:\-]?\s*(\d{1,3})", answer_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(score, 100))

    return None


def get_validation_badge(validation_text: str) -> str:
    """Map validation response text to a badge label."""
    if not validation_text:
        return "Unknown"

    text = validation_text.lower()

    # Prefer the explicit ## Validation Status section
    if "## validation status" in text:
        if "unsupported" in text and "partially supported" not in text:
            return "Unsupported"
        if "partially supported" in text:
            return "Partially Supported"
        if "supported" in text:
            return "Supported"

    # Fallback: scan full text
    if "partially supported" in text:
        return "Partially Supported"
    if "unsupported" in text:
        return "Unsupported"
    if "supported" in text:
        return "Supported"

    return "Unknown"