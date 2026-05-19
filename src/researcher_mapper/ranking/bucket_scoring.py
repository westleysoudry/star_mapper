"""Compute per-bucket scores for each candidate."""
from __future__ import annotations


def score_in_area(
    same_area: float,
    collaboration: float,
    venue: float,
    institution_distance_bonus: float,
    recency_overlap: float,
    weights: dict | None = None,
) -> float:
    w = weights or {
        "same_area": 0.45,
        "collaboration": 0.15,
        "venue": 0.20,
        "institution_distance_bonus": 0.10,
        "recency_overlap": 0.10,
    }
    return (
        w["same_area"] * same_area
        + w["collaboration"] * collaboration
        + w["venue"] * venue
        + w["institution_distance_bonus"] * institution_distance_bonus
        + w["recency_overlap"] * recency_overlap
    )


def score_interdisciplinary(
    complementary_topic: float,
    method_complementarity: float,
    citation_bridge: float,
    collaboration_potential: float,
    institutional_proximity: float,
    weights: dict | None = None,
) -> float:
    w = weights or {
        "complementary_topic": 0.35,
        "method_complementarity": 0.20,
        "citation_bridge": 0.15,
        "collaboration_potential": 0.15,
        "institutional_proximity": 0.15,
    }
    return (
        w["complementary_topic"] * complementary_topic
        + w["method_complementarity"] * method_complementarity
        + w["citation_bridge"] * citation_bridge
        + w["collaboration_potential"] * collaboration_potential
        + w["institutional_proximity"] * institutional_proximity
    )


def score_letter_writer(
    same_area: float,
    seniority: float,
    reputation: float,
    no_conflict: float,
    familiarity_proxy: float,
    venue: float = 0.0,
    weights: dict | None = None,
) -> float:
    w = weights or {
        "same_area": 0.30,
        "seniority": 0.20,
        "reputation": 0.15,
        "no_conflict": 0.15,
        "familiarity_proxy": 0.15,
        "venue": 0.05,
    }
    return (
        w["same_area"] * same_area
        + w["seniority"] * seniority
        + w["reputation"] * reputation
        + w["no_conflict"] * no_conflict
        + w["familiarity_proxy"] * familiarity_proxy
        + w.get("venue", 0.0) * venue
    )


def score_dept_mentor(
    department_match: float,
    same_area: float,
    seniority: float,
    institutional_role: float,
    mentoring_signal: float,
    weights: dict | None = None,
) -> float:
    w = weights or {
        "department_match": 0.35,
        "same_area": 0.25,
        "seniority": 0.20,
        "institutional_role": 0.10,
        "mentoring_signal": 0.10,
    }
    return (
        w["department_match"] * department_match
        + w["same_area"] * same_area
        + w["seniority"] * seniority
        + w["institutional_role"] * institutional_role
        + w["mentoring_signal"] * mentoring_signal
    )


def score_area_mentor(
    same_area: float,
    seniority: float,
    reputation: float,
    distance_diversity_bonus: float,
    mentoring_signal: float,
    weights: dict | None = None,
) -> float:
    w = weights or {
        "same_area": 0.35,
        "seniority": 0.25,
        "reputation": 0.15,
        "distance_diversity_bonus": 0.15,
        "mentoring_signal": 0.10,
    }
    return (
        w["same_area"] * same_area
        + w["seniority"] * seniority
        + w["reputation"] * reputation
        + w["distance_diversity_bonus"] * distance_diversity_bonus
        + w["mentoring_signal"] * mentoring_signal
    )


def compute_all_bucket_scores(
    same_area: float,
    complementary_topic: float,
    seniority: float,
    reputation: float,
    no_conflict: float,
    familiarity_proxy: float,
    collaboration: float,
    venue: float,
    institution_distance_bonus: float,
    recency_overlap: float,
    department_match: float,
    institutional_role: float,
    weights: dict | None = None,
) -> dict[str, float]:
    """Convenience wrapper that returns all five bucket scores at once."""
    bw = (weights or {}).get("buckets", {})
    return {
        "in_area_collaborators": score_in_area(
            same_area, collaboration, venue,
            institution_distance_bonus, recency_overlap,
            weights=bw.get("in_area_collaborators"),
        ),
        "interdisciplinary_collaborators": score_interdisciplinary(
            complementary_topic,
            method_complementarity=collaboration * 0.5,  # proxy
            citation_bridge=venue,
            collaboration_potential=collaboration,
            institutional_proximity=institution_distance_bonus,
            weights=bw.get("interdisciplinary_collaborators"),
        ),
        "recommendation_letter_writers": score_letter_writer(
            same_area, seniority, reputation, no_conflict, familiarity_proxy,
            venue=venue,
            weights=bw.get("recommendation_letter_writers"),
        ),
        "dept_mentors": score_dept_mentor(
            department_match, same_area, seniority,
            institutional_role, mentoring_signal=seniority * 0.5,
            weights=bw.get("dept_mentors"),
        ),
        "area_mentors": score_area_mentor(
            same_area, seniority, reputation,
            distance_diversity_bonus=institution_distance_bonus,
            mentoring_signal=seniority * 0.5,
            weights=bw.get("area_mentors"),
        ),
    }
