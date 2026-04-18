"""
Extracts clean plain text from SEC Exhibit 99.1 HTML documents.

Apple's 8-K Exhibit 99.1 is the earnings press release. It is an HTML
document containing financial highlights, management quotes, and detailed
segment results. This module strips structural HTML noise (scripts, styles,
excessive blank lines) and returns clean plain text ready for storage and
downstream embedding.

Returns None if the document is empty or contains only whitespace after
cleaning (e.g. placeholder or redirect pages).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def extract_exhibit_text(html_content: str, accession_no: str) -> str | None:
    """
    Extract clean plain text from an Exhibit 99.1 HTML document.

    Processing steps:
    1. Parse HTML with BeautifulSoup + lxml.
    2. Remove all <script> and <style> tags in-place.
    3. Extract plain text with newline separators.
    4. Strip each line; discard empty lines; collapse runs of 3+
       consecutive blank lines to a single blank line.
    5. Return the joined text, or None if nothing remains.

    Args:
        html_content: Raw HTML string of the Exhibit 99.1 document.
        accession_no: Used in log messages for traceability.

    Returns:
        Cleaned plain text string, or None if the document is empty
        after cleaning.
    """
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    text = "\n".join(lines)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text).strip()

    if not text:
        logger.info(
            "Exhibit text is empty after cleaning",
            extra={"accession_no": accession_no},
        )
        return None

    logger.info(
        "Extracted exhibit text",
        extra={"accession_no": accession_no, "char_count": len(text)},
    )
    return text
