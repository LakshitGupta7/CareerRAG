import re


def parse_sections(answer_text: str) -> dict:
    """
    Parses markdown-style sections like:
    ## Candidate Overview
    ...
    ## Core Technical Skills
    ...

    Returns:
        {
            "Candidate Overview": "...",
            "Core Technical Skills": "...",
            ...
        }
    """
    if not answer_text or not answer_text.strip():
        return {}

    pattern = r"##\s+(.+?)\n(.*?)(?=\n##\s+|\Z)"
    matches = re.findall(pattern, answer_text, flags=re.DOTALL)

    sections = {}
    for title, content in matches:
        sections[title.strip()] = content.strip()

    return sections


def extract_match_score(answer_text: str):
    """
    Extracts numeric match score from text.
    Handles patterns like:
    Match Score: 78
    Match Score - 78
    """
    if not answer_text:
        return None

    match = re.search(r"Match Score\s*[:\-]?\s*(\d{1,3})", answer_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(score, 100))

    return None

def get_validation_badge(validation_text: str) -> str:
    if not validation_text:
        return "Unknown"

    text = validation_text.lower()

    if "## validation status" in text:
        if "unsupported" in text and "partially supported" not in text:
            return "Unsupported"
        if "partially supported" in text:
            return "Partially Supported"
        if "supported" in text:
            return "Supported"

    if "partially supported" in text:
        return "Partially Supported"
    if "unsupported" in text:
        return "Unsupported"
    if "supported" in text:
        return "Supported"

    return "Unknown"