"""Coauthor and venue network features."""
from __future__ import annotations

from collections import Counter
from datetime import datetime

from researcher_mapper.models.schemas import Publication

CURRENT_YEAR = datetime.utcnow().year


def _short_id(openalex_id: str) -> str:
    return (openalex_id or "").split("/")[-1]


def build_coauthor_vector(
    publications: list[Publication],
    target_author_id: str,
) -> dict[str, float]:
    """
    Frequency-weighted coauthor vector (excluding the target themselves).
    Higher values = more frequent collaborators.
    """
    counts: Counter[str] = Counter()
    target_short = _short_id(target_author_id)
    for pub in publications:
        for aid in pub.author_ids:
            if aid and _short_id(aid) != target_short:
                counts[aid] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def build_venue_vector(publications: list[Publication]) -> dict[str, float]:
    """TF vector over publication venue names."""
    counts: Counter[str] = Counter()
    for pub in publications:
        if pub.venue:
            counts[pub.venue.lower().strip()] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def get_coauthor_ids(publications: list[Publication], target_author_id: str) -> set[str]:
    """Return the set of all co-author OpenAlex IDs."""
    ids: set[str] = set()
    target_short = _short_id(target_author_id)
    for pub in publications:
        for aid in pub.author_ids:
            if aid and _short_id(aid) != target_short:
                ids.add(aid)
    return ids


def get_recent_coauthor_ids(
    publications: list[Publication],
    target_author_id: str,
    recent_years: int = 5,
) -> set[str]:
    """Return co-author IDs from publications within the last `recent_years`."""
    cutoff = CURRENT_YEAR - recent_years
    ids: set[str] = set()
    target_short = _short_id(target_author_id)
    for pub in publications:
        if pub.year and pub.year < cutoff:
            continue
        for aid in pub.author_ids:
            if aid and _short_id(aid) != target_short:
                ids.add(aid)
    return ids


def last_author_fraction(
    publications: list[Publication],
    target_author_id: str,
) -> float:
    """
    Fraction of papers where the target is last author.
    Last-author position signals seniority in many fields.
    """
    if not publications:
        return 0.0
    target_short = _short_id(target_author_id)
    last = sum(
        1
        for pub in publications
        if pub.author_ids and _short_id(pub.author_ids[-1]) == target_short
    )
    return last / len(publications)
