"""Build topic / concept frequency vectors from works or author-level summaries."""
from __future__ import annotations

from collections import Counter

from researcher_mapper.models.schemas import Publication


def build_topic_vector(publications: list[Publication]) -> dict[str, float]:
    """TF vector over OpenAlex topic IDs, normalized by total topic mentions."""
    counts: Counter[str] = Counter()
    for pub in publications:
        for tid in pub.topic_ids:
            counts[tid] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def build_concept_vector(publications: list[Publication]) -> dict[str, float]:
    """TF vector over OpenAlex concept IDs (older field, broader coverage)."""
    counts: Counter[str] = Counter()
    for pub in publications:
        for cid in pub.concept_ids:
            counts[cid] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def build_concept_vector_from_summary(top_concepts: dict[str, float]) -> dict[str, float]:
    """Use the pre-built author-level concept weights directly (already normalised 0–1)."""
    return dict(top_concepts)


def complementary_topic_score(
    target_concept: dict[str, float],
    candidate_concept: dict[str, float],
) -> float:
    """
    How complementary (but not too distant) are two concept vectors?
    Uses: 1 - weighted_jaccard, but clamped to [0, 0.8].
    High similarity → same area (low complementarity).
    Very low similarity → too distant (also low complementarity).
    The sweet spot is moderate overlap (Jaccard ~0.20–0.50).
    """
    from researcher_mapper.features.citation_features import weighted_jaccard
    sim = weighted_jaccard(target_concept, candidate_concept)
    # bell curve centered at sim=0.35
    raw = 1.0 - abs(sim - 0.35) / 0.35
    return max(0.0, min(raw, 1.0))
