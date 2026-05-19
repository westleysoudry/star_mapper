"""Career-stage and seniority features."""
from __future__ import annotations

from datetime import datetime

CURRENT_YEAR = datetime.utcnow().year


def estimate_seniority(
    first_pub_year: int | None,
    estimated_rank: str | None = None,
    cited_by_count: int = 0,
    works_count: int = 0,
    last_author_fraction: float = 0.0,
) -> float:
    """
    Returns a seniority score in [0, 1].

    Components:
    - Career age (0–30 years mapped to 0–1)
    - Academic title (if known)
    - Citation impact proxy
    - Last-author fraction (seniority signal)
    """
    career_age = 0 if first_pub_year is None else max(CURRENT_YEAR - first_pub_year, 0)

    title_score = {
        "assistant": 0.55,
        "associate": 0.75,
        "full": 1.00,
        "professor": 1.00,
        "unknown": 0.40,
        None: 0.40,
    }.get(estimated_rank, 0.40)

    # Log-scaled citation impact (h-index proxy)
    import math
    citation_score = min(math.log1p(cited_by_count) / math.log1p(5000), 1.0)

    raw = (
        0.35 * min(career_age / 25.0, 1.0)
        + 0.25 * title_score
        + 0.25 * citation_score
        + 0.15 * last_author_fraction
    )
    return min(raw, 1.0)


def estimate_reputation(cited_by_count: int, works_count: int) -> float:
    """
    Rough reputation proxy: citations-per-paper, log-scaled to [0, 1].
    Works best as a relative ranking signal, not an absolute measure.
    """
    import math
    if works_count == 0:
        return 0.0
    cpp = cited_by_count / works_count
    return min(math.log1p(cpp) / math.log1p(200), 1.0)


def career_feature_dict(
    first_pub_year: int | None,
    last_pub_year: int | None,
    cited_by_count: int,
    works_count: int,
    last_author_fraction: float = 0.0,
) -> dict[str, float]:
    """Build the career_features dict stored in FeatureVector."""
    import math

    career_age = 0.0 if first_pub_year is None else float(max(CURRENT_YEAR - first_pub_year, 0))
    pub_recency = 0.0
    if last_pub_year is not None:
        pub_recency = max(0.0, 1.0 - (CURRENT_YEAR - last_pub_year) / 10.0)

    return {
        "career_age_years": career_age,
        "career_age_norm": min(career_age / 25.0, 1.0),
        "log_citations": math.log1p(cited_by_count),
        "log_works": math.log1p(works_count),
        "citations_per_paper": cited_by_count / max(works_count, 1),
        "last_author_fraction": last_author_fraction,
        "pub_recency": pub_recency,
    }


def career_similarity(
    target_features: dict[str, float],
    candidate_features: dict[str, float],
) -> float:
    """
    Similarity on career-stage features.
    Researchers at similar career stages score higher.
    """
    if not target_features or not candidate_features:
        return 0.0

    t_age = target_features.get("career_age_norm", 0.5)
    c_age = candidate_features.get("career_age_norm", 0.5)
    age_sim = 1.0 - abs(t_age - c_age)

    t_cit = target_features.get("log_citations", 0.0)
    c_cit = candidate_features.get("log_citations", 0.0)
    max_cit = max(t_cit, c_cit, 1.0)
    cit_sim = 1.0 - abs(t_cit - c_cit) / max_cit

    return 0.6 * age_sim + 0.4 * cit_sim
