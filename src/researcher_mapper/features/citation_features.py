"""Citation and reference overlap features."""
from __future__ import annotations

from collections import Counter

from researcher_mapper.models.schemas import Publication


def build_reference_vector(publications: list[Publication]) -> dict[str, float]:
    """TF vector over referenced work IDs (bibliographic coupling signal)."""
    counts: Counter[str] = Counter()
    for pub in publications:
        for ref_id in pub.referenced_work_ids:
            counts[ref_id] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def build_cited_works_vector(publications: list[Publication]) -> dict[str, float]:
    """Treats each publication as a node — used for co-citation analysis."""
    return {pub.id: pub.cited_by_count for pub in publications if pub.id}


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Weighted Jaccard similarity for two frequency/weight dictionaries.
    Range [0, 1].  Returns 0 if both are empty.
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity for two sparse weight dictionaries."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
