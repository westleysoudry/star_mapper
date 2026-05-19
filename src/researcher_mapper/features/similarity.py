"""Compute pairwise similarity between a target and a candidate FeatureVector."""
from __future__ import annotations

from researcher_mapper.features.citation_features import weighted_jaccard
from researcher_mapper.features.career_features import career_similarity
from researcher_mapper.models.schemas import FeatureVector


def _best_topic_sim(target: FeatureVector, candidate: FeatureVector) -> float:
    """
    Compare topic vectors using matching ID namespaces only.

    OpenAlex has two concept/topic namespaces:
      T-prefixed: topics     (newer; from publication topic_ids and author.topics)
      C-prefixed: concepts   (older; from author.x_concepts)

    Comparing T-IDs against C-IDs always gives Jaccard = 0 because the ID sets
    are disjoint.  We pick the best same-namespace comparison:
      1. topic vs topic   (T-prefixed)  — preferred when both are non-empty
      2. concept vs concept (C-prefixed) — fallback for older records
    """
    t_topic_sim = (
        weighted_jaccard(target.topic_vector, candidate.topic_vector)
        if target.topic_vector and candidate.topic_vector
        else 0.0
    )
    c_concept_sim = (
        weighted_jaccard(target.concept_vector, candidate.concept_vector)
        if target.concept_vector and candidate.concept_vector
        else 0.0
    )
    return max(t_topic_sim, c_concept_sim)


def compute_base_similarity(
    target: FeatureVector,
    candidate: FeatureVector,
    weights: dict | None = None,
    candidate_cites_target: bool = False,
) -> float:
    """
    Weighted combination of topic, citation, coauthor, venue, and career similarities.

    Default weights (overridden by config/weights.yaml at runtime):
      topic: 0.35, citation: 0.25, collaboration: 0.20, venue: 0.10, career: 0.10

    ``candidate_cites_target``: if True (the candidate has cited one of the
    target's papers), a +0.25 additive bonus is applied after the weighted sum.
    This makes citation-graph proximity a dominant signal when present.
    """
    w = weights or {
        "topic": 0.35,
        "citation": 0.30,
        "collaboration": 0.20,
        "venue": 0.10,
        "career": 0.05,
    }

    topic_sim = _best_topic_sim(target, candidate)
    citation_sim = weighted_jaccard(target.citation_vector, candidate.citation_vector)
    collab_sim = weighted_jaccard(target.coauthor_vector, candidate.coauthor_vector)
    venue_sim = weighted_jaccard(target.venue_vector, candidate.venue_vector)
    career_sim = career_similarity(target.career_features, candidate.career_features)

    score = (
        w["topic"] * topic_sim
        + w["citation"] * citation_sim
        + w["collaboration"] * collab_sim
        + w["venue"] * venue_sim
        + w["career"] * career_sim
    )
    if candidate_cites_target:
        score = min(score + 0.25, 1.0)
    return score


def compute_same_area_score(
    target: FeatureVector,
    candidate: FeatureVector,
    candidate_cites_target: bool = False,
) -> float:
    """
    Combined topic + citation overlap — used as the 'same research area' signal.

    ``candidate_cites_target``: citing the target's papers is strong evidence of
    same-area research.  A floor of 0.25 is applied so citing authors always
    clear the bucket hard-filter thresholds (≥ 0.10–0.12).
    """
    topic_sim = _best_topic_sim(target, candidate)
    citation_sim = weighted_jaccard(target.citation_vector, candidate.citation_vector)
    score = 0.65 * topic_sim + 0.35 * citation_sim
    if candidate_cites_target:
        score = max(score, 0.25)
    return score


def compute_familiarity_proxy(
    target_coauthor_vector: dict[str, float],
    candidate_openalex_id: str,
    target_ref_vector: dict[str, float],
    candidate_work_ids: set[str],
    candidate_cites_target: bool = False,
) -> float:
    """
    Proxy for how familiar the candidate is with the target's work.

    Signals:
      (a) coauthor — they have co-authored with the target
      (b) target→candidate citation — target's papers reference candidate's works
      (c) candidate→target citation — candidate has explicitly cited the target's papers
          (strongest relevance signal; captured via the citing-authors pool strategy)
    """
    coauthor_score = target_coauthor_vector.get(candidate_openalex_id, 0.0)

    # Fraction of candidate's works that appear in target's reference list
    if candidate_work_ids and target_ref_vector:
        overlap = sum(target_ref_vector.get(wid, 0.0) for wid in candidate_work_ids)
        ref_score = min(overlap * 5.0, 1.0)  # scale: even 0.2 overlap is high
    else:
        ref_score = 0.0

    # Candidate has cited one of the target's papers — strong directional signal
    cite_score = 1.0 if candidate_cites_target else 0.0

    return (
        0.35 * min(coauthor_score * 10.0, 1.0)
        + 0.30 * ref_score
        + 0.35 * cite_score
    )
