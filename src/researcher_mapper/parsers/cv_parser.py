"""
CV parser — supports PDF and plain-text files.

Extracted fields
----------------
name, email, ORCID, institution, department, position_rank,
career_start_year, phd_year, research_interests,
publication_titles, publication_dois

Strategy
--------
1. Extract raw text (pdfplumber for PDF, UTF-8 read for .txt).
2. Split text into named sections (Education, Positions, Research Interests,
   Publications, …) using a header-detection pass.
3. Run targeted extractors over each section.
4. Fall back to whole-document search for identifiers (ORCID, email, DOI).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from researcher_mapper.models.schemas import CVData
from researcher_mapper.settings import load_department_aliases

# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(path: Path) -> str:
    """Return concatenated text from all PDF pages using pdfplumber."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required to read PDF files.  "
            "Run: pip install pdfplumber"
        ) from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_text(path: Path) -> str:
    """Dispatch to the right extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    # Plain text (.txt or anything else)
    return path.read_text(encoding="utf-8", errors="replace")


# ── Section splitter ──────────────────────────────────────────────────────────

# Map section label → regex that matches a header line for that section.
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("contact",      re.compile(r"^(?:contact|personal\s+(?:info(?:rmation)?|details)|address)", re.I)),
    ("education",    re.compile(r"^(?:education|academic\s+(?:background|degrees?|training|qualifications?)|degrees?|training)", re.I)),
    ("positions",    re.compile(r"^(?:positions?|employment|experience|academic\s+(?:posts?|appointments?|positions?|experience)|work\s+history|appointments?)", re.I)),
    ("interests",    re.compile(r"^(?:research\s+interests?|research\s+areas?|areas?\s+of\s+(?:interest|expertise|research)|keywords?|research\s+focus|topics?)", re.I)),
    ("publications", re.compile(r"^(?:publications?|papers?|preprints?|journal\s+articles?|conference\s+(?:papers?|proceedings?)|selected\s+publications?|books?)", re.I)),
    ("grants",       re.compile(r"^(?:grants?|funding|awards?|honors?|distinctions?|fellowships?|prizes?)", re.I)),
    ("service",      re.compile(r"^(?:service|professional\s+activities|academic\s+service|reviewing?)", re.I)),
    ("teaching",     re.compile(r"^(?:teaching|courses?|instruction|supervision)", re.I)),
    ("talks",        re.compile(r"^(?:talks?|presentations?|invited\s+talks?)", re.I)),
]

# A header line: short (≤80 chars), no sentence-ending punctuation at the end,
# either ALL CAPS or Title Case or matching a known section keyword.
_HEADER_RE = re.compile(r"^([A-Z][A-Z\s&/\-]{1,60}[A-Z]|[A-Z][a-z]+(?:\s+[A-Za-z]+){0,5})$")


_NUMBERED_PREFIX_RE = re.compile(r"^\d+[.)]\s*")


def _strip_numbering(line: str) -> str:
    """Remove leading section-number prefixes like '4.' or '3)' from a line."""
    return _NUMBERED_PREFIX_RE.sub("", line)


def _is_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    # Strip a leading number prefix (e.g. "4. RESEARCH INTERESTS") before matching
    core = _strip_numbering(stripped)
    for _, pat in _SECTION_PATTERNS:
        if pat.match(core):
            return True
    # All-caps line (possibly after removing the number) with at least 3 chars
    if core.isupper() and len(core) >= 3:
        return True
    return False


def _classify_header(line: str) -> str:
    # Must apply the same de-numbering as _is_header so label lookup stays in sync
    core = _strip_numbering(line.strip())
    for label, pat in _SECTION_PATTERNS:
        if pat.match(core):
            return label
    return "other"


def split_into_sections(text: str) -> dict[str, str]:
    """
    Return a dict mapping section label → text body.
    Text before the first header is labelled "header" (usually name + contact).
    Unknown sections get the label "other".
    """
    sections: dict[str, list[str]] = {"header": []}
    current = "header"

    for line in text.splitlines():
        if _is_header(line):
            current = _classify_header(line)
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "".join(v).strip()}


# ── Identifier extractors ─────────────────────────────────────────────────────

_ORCID_RE   = re.compile(r"(?:orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", re.I)
_EMAIL_RE   = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.]+")
_DOI_RE     = re.compile(r"\b(10\.\d{4,}/[^\s,;>\"')\]]+)", re.I)
_YEAR_RE    = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def _find_orcid(text: str) -> str | None:
    m = _ORCID_RE.search(text)
    return m.group(1) if m else None


def _find_email(text: str) -> str | None:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _find_dois(text: str) -> list[str]:
    return list(dict.fromkeys(_DOI_RE.findall(text)))  # deduplicated, order-preserved


def _find_years(text: str) -> list[int]:
    return sorted({int(y) for y in _YEAR_RE.findall(text)})


# ── Name extraction ───────────────────────────────────────────────────────────

_HONORIFICS = re.compile(
    r"^(?:prof(?:essor)?|dr|mr|ms|mrs|associate|assistant|full|senior|emeritus)\.?\s+",
    re.I,
)
# Looks like a personal name: 2–4 words, each capitalised, no digits
_NAME_RE = re.compile(r"^([A-Z][a-zéàüöä\-']+(?:\s+[A-Z][a-zéàüöä\-']+){1,3})$")
# Labeled format: "Name: Daniel Soudry" or "Full Name: Daniel Soudry"
_LABELED_NAME_RE = re.compile(r"^(?:full\s+)?name\s*:\s*(.+)$", re.I)


def _extract_name(header_text: str) -> str | None:
    for line in header_text.splitlines()[:6]:
        line = line.strip()
        # Check labeled format first ("Name: X" / "Full Name: X")
        m = _LABELED_NAME_RE.match(line)
        if m:
            candidate = _HONORIFICS.sub("", m.group(1).strip()).strip()
            if _NAME_RE.match(candidate):
                return candidate
        # Strip honorifics and check bare name pattern
        line = _HONORIFICS.sub("", line).strip()
        if _NAME_RE.match(line):
            return line
    return None


# ── Institution / department extraction ───────────────────────────────────────

_INSTITUTION_KEYWORDS = re.compile(
    r"(?:university|institute|college|school|technion|mit|caltech|faculty|"
    r"polytechnic|akademie|akademia|universit[äye])",
    re.I,
)

def _extract_institution(text: str) -> str | None:
    """Find the first line that contains a university-like keyword."""
    for line in text.splitlines():
        line = line.strip()
        if _INSTITUTION_KEYWORDS.search(line) and len(line) < 120:
            return line
    return None


def _extract_department(text: str) -> str | None:
    """Delegate to the alias-aware matcher in faculty_page_parser."""
    from researcher_mapper.parsers.faculty_page_parser import extract_department_from_text
    return extract_department_from_text(text)


# ── Rank / position extraction ────────────────────────────────────────────────

_RANK_PATTERNS = [
    ("full",       re.compile(r"(?<!associate )(?<!assistant )(?:full\s+)?professor\b", re.I)),
    ("associate",  re.compile(r"\bassociate\s+professor\b", re.I)),
    ("assistant",  re.compile(r"\bassistant\s+professor\b", re.I)),
    ("postdoc",    re.compile(r"\bpost\s*doc(?:toral)?\b", re.I)),
    ("lecturer",   re.compile(r"\blecturer\b", re.I)),
    ("researcher", re.compile(r"\bresearch(?:er|er\s+fellow)?\b", re.I)),
]

def _extract_rank(text: str) -> str | None:
    """Return the most senior rank found anywhere in the text."""
    _PRIORITY = ["full", "associate", "assistant", "lecturer", "researcher", "postdoc"]
    found: set[str] = set()
    for rank, pat in _RANK_PATTERNS:
        if pat.search(text):
            found.add(rank)
    for rank in _PRIORITY:
        if rank in found:
            return rank
    return None


# ── Career / PhD year ─────────────────────────────────────────────────────────

_PHD_RE = re.compile(
    r"\bph\.?d\.?\b.*?(?:19[5-9]\d|20[0-2]\d)|(?:19[5-9]\d|20[0-2]\d).*?\bph\.?d\.?\b",
    re.I,
)

def _extract_phd_year(education_text: str) -> int | None:
    m = _PHD_RE.search(education_text)
    if m:
        years = _YEAR_RE.findall(m.group(0))
        if years:
            return int(years[-1])  # last year in the PhD line = graduation year
    # Fallback: earliest year in education section
    years = _find_years(education_text)
    return min(years) if years else None


# ── Advisor / mentor extraction ───────────────────────────────────────────────

# Matches patterns like:
#   "Advisor: John Smith"  "PhD Supervisor: Jane Doe"
#   "Postdoc advisor: Alice Brown"  "Host: Prof. Liam Paninski"
#
# Requires a colon after the keyword to avoid false positives (e.g. "hosted by").
# An optional honorific (Prof./Dr.) between the colon and name is stripped.
_ADVISOR_RE = re.compile(
    r"(?:"
    r"(?:ph\.?d\.?[\s\-]+)?(?:advisor|adviser|supervisor|mentor)"
    r"|postdoc(?:toral)?[\s\-]+(?:advisor|adviser|supervisor|host|mentor)"
    r"|host"
    r")"
    r":\s*"
    r"(?:prof(?:essor)?\.?\s+|dr\.?\s+)?"   # optional honorific after colon
    r"([A-Z][a-zA-Zéàüöä'\-]+\s+[A-Z][a-zA-Zéàüöä'\-]+(?:\s+[A-Z][a-zA-Zéàüöä'\-]+){0,2})",
    re.I,
)


def _extract_advisors(text: str) -> list[str]:
    """
    Find advisor/supervisor/postdoc-host names from free text.

    Handles lines like:
      "Advisor: Jane Smith"
      "Ph.D. Supervisor: John Doe"
      "Postdoc advisor: Alice Johnson"
    """
    found: list[str] = []
    for m in _ADVISOR_RE.finditer(text):
        name = m.group(1).strip()
        # Strip honorifics that may have been captured
        name = _HONORIFICS.sub("", name).strip()
        # Discard if it looks like an institution rather than a person
        if _INSTITUTION_KEYWORDS.search(name):
            continue
        if name and name not in found:
            found.append(name)
    return found


# ── Research interests ────────────────────────────────────────────────────────

def _extract_research_interests(section_text: str) -> list[str]:
    """
    Parse research interests from the interests/keywords section.
    Handles: comma-separated lists, semicolon-separated, bullet-point lines,
    and mixed-format sections where some lines are long comma-joined phrases.
    """
    if not section_text.strip():
        return []

    lines = [l.strip() for l in section_text.splitlines() if l.strip()]

    # Long narrative "research statement" sections should stay sentence-like.
    # Splitting wrapped prose on commas produces dozens of noisy fragments,
    # which is what Daniel Soudry's PDF currently does.
    prose_like = (
        len(lines) >= 4
        and sum(1 for line in lines if len(line) >= 40) >= 3
        and not any(re.match(r"^[\-\u2022\u00b7*]", line) for line in lines)
    )
    if prose_like:
        merged = re.sub(r"\s+", " ", " ".join(lines)).strip()
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", merged)
        cleaned = [
            sentence.strip()
            for sentence in sentences
            if 20 <= len(sentence.strip()) <= 400
        ]
        return cleaned[:12]

    # If a small number of lines contain comma-separated lists, join and re-split.
    # This covers the common "Machine learning, deep learning, optimization" format.
    if len(lines) <= 3 and any("," in l or ";" in l for l in lines):
        raw = " ".join(lines)
        items = re.split(r"[,;•·]+", raw)
    else:
        items = lines

    # Secondary pass: any item still longer than 60 chars that contains commas
    # is a compound phrase that was not split above — break it down further.
    expanded: list[str] = []
    for item in items:
        item = item.strip()
        if len(item) > 60 and ("," in item or ";" in item):
            expanded.extend(re.split(r"[,;]+", item))
        else:
            expanded.append(item)

    cleaned: list[str] = []
    for item in expanded:
        item = re.sub(r"^[\s\-•·–—*]+", "", item).strip()
        item = re.sub(r"\s+", " ", item)
        if 3 <= len(item) <= 120:
            cleaned.append(item)

    return cleaned


# ── Publications ──────────────────────────────────────────────────────────────

_TITLE_IN_QUOTES_RE = re.compile(r'["""](.{10,200})["""]')
_TITLE_ITALIC_RE    = re.compile(r"_(.{10,200})_")   # Markdown-style italic

def _extract_publications(section_text: str) -> tuple[list[str], list[str]]:
    """
    Return (titles, dois) extracted from the publications section.
    Titles are extracted from quoted strings or heuristic line patterns.
    """
    dois   = _find_dois(section_text)
    titles: list[str] = []

    # Quoted titles
    for m in _TITLE_IN_QUOTES_RE.finditer(section_text):
        t = m.group(1).strip()
        if t not in titles:
            titles.append(t)

    # Italic titles (markdown)
    for m in _TITLE_ITALIC_RE.finditer(section_text):
        t = m.group(1).strip()
        if t not in titles:
            titles.append(t)

    # Heuristic: numbered/bulleted lines that look like citation entries
    for line in section_text.splitlines():
        line = line.strip()
        # Starts with [N], (N), or N. and is long enough to be a citation
        if re.match(r"^[\[\(]?\d+[\]\).]", line) and len(line) > 40:
            # Strip leading number
            text = re.sub(r"^[\[\(]?\d+[\]\).]\s*", "", line)
            # Strip trailing DOI/URL
            text = re.sub(r"\s+(?:doi|https?|arxiv).*$", "", text, flags=re.I)
            if 20 <= len(text) <= 250 and text not in titles:
                titles.append(text)

    return titles[:100], dois


# ── Master parse function ─────────────────────────────────────────────────────

def parse_cv(path: str | Path) -> CVData:
    """
    Parse a CV file (PDF or plain text) and return a CVData object.

    All fields are best-effort; missing data is left as None / empty list.
    """
    path = Path(path)
    raw_text = extract_text(path)
    sections = split_into_sections(raw_text)

    header_text    = sections.get("header", "")
    contact_text   = sections.get("contact", "")
    positions_text = sections.get("positions", "")
    education_text = sections.get("education", "")
    interests_text = sections.get("interests", "")
    pubs_text      = sections.get("publications", "")

    # Combine header + positions + education for institution/rank/dept search.
    # Education section often contains the primary affiliation (e.g. PhD institution
    # stated as current position) and is needed for department extraction when the
    # positions section is sparse or uses shorthand.
    affiliation_text = f"{header_text}\n{positions_text}\n{education_text}"

    # ── Identity ──────────────────────────────────────────────────────────────
    name  = _extract_name(header_text) or _extract_name(contact_text) or _extract_name(raw_text)
    email = _find_email(raw_text)
    orcid = _find_orcid(raw_text)

    # ── Affiliation ───────────────────────────────────────────────────────────
    institution = _extract_institution(affiliation_text)
    department  = _extract_department(affiliation_text) or _extract_department(raw_text)
    rank        = _extract_rank(affiliation_text) or _extract_rank(raw_text)

    # ── Career timeline ───────────────────────────────────────────────────────
    phd_year = _extract_phd_year(education_text) if education_text else None

    # Career start = PhD year; if missing, earliest year across the whole CV
    career_start: int | None = phd_year
    if career_start is None:
        all_years = _find_years(education_text or raw_text)
        career_start = min(all_years) if all_years else None

    # ── Research interests ────────────────────────────────────────────────────
    interests = _extract_research_interests(interests_text)

    # ── Publications ──────────────────────────────────────────────────────────
    titles, dois = _extract_publications(pubs_text)

    # ── Known advisors / mentors (from Education and Positions sections) ──────
    advisors = _extract_advisors(
        f"{education_text}\n{positions_text}"
    )

    return CVData(
        name=name,
        email=email,
        orcid=orcid,
        institution=institution,
        department=department,
        position_rank=rank,
        career_start_year=career_start,
        phd_year=phd_year,
        research_interests=interests,
        publication_titles=titles,
        publication_dois=dois,
        known_advisors=advisors,
        raw_text=raw_text,
    )
