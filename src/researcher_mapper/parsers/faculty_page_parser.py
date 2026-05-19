"""
Faculty-page and CV parser for department enrichment.

Department data is the weakest part of the public scholarly graph — OpenAlex
affiliations are institution-level, not department-level.  This module fills
that gap by scraping the researcher's faculty profile page or a plain-text CV.
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from researcher_mapper.settings import load_department_aliases


def fetch_page_text(url: str, timeout: float = 20.0) -> str:
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    # Remove nav, footer, script, style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def extract_department_from_text(
    text: str,
    aliases: dict[str, list[str]] | None = None,
) -> str | None:
    """
    Try to infer a canonical department name from free text.

    Resolution order:
    1. Alias lookup (config/department_aliases.yaml).
    2. Regex pattern "Department/School/Faculty of <X>".
    """
    aliases = aliases or load_department_aliases()
    lowered = text.lower()

    def _matches(term: str, haystack: str) -> bool:
        """Word-boundary aware match — prevents 'cs' firing inside 'eecs'."""
        return bool(re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", haystack))

    # Collect ALL matches as (term_length, canonical) then return the canonical
    # whose matched term was longest.  This prevents short aliases like
    # "computer engineering" (21 chars) winning over "electrical and computer
    # engineering" (35 chars) simply because CS is listed first in the YAML.
    best_len = 0
    best_canonical: str | None = None

    for canonical, variants in aliases.items():
        for term in [canonical, *variants]:
            if _matches(term, lowered) and len(term) > best_len:
                best_len = len(term)
                best_canonical = canonical

    if best_canonical:
        return best_canonical

    # Generic pattern: "Department/School/Faculty of <Name>"
    # Use word-only groups so commas/punctuation never bleed into the capture.
    m = re.search(
        r"(?:department|school|faculty|division)\s+of\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,4})",
        text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip().lower()
        for canonical, variants in aliases.items():
            if canonical in raw:
                return canonical
            for variant in variants:
                if variant in raw:
                    return canonical
        return raw[:64]

    # Fallback: "<Name> Department" pattern (e.g. "EECS Department", "CS Dept")
    m2 = re.search(
        r"([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:department|school|faculty)",
        text,
        re.IGNORECASE,
    )
    if m2:
        raw = m2.group(1).strip().lower()
        for canonical, variants in aliases.items():
            if canonical in raw:
                return canonical
            for variant in variants:
                if variant in raw:
                    return canonical
        return raw[:64]

    return None


def extract_department_from_cv_file(
    cv_path: str | Path,
    aliases: dict[str, list[str]] | None = None,
) -> str | None:
    """Extract department from a CV file (.txt or .pdf)."""
    from researcher_mapper.parsers.cv_parser import extract_text
    text = extract_text(Path(cv_path))
    return extract_department_from_text(text, aliases)
