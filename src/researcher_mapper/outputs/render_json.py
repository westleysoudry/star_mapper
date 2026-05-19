"""JSON output serialiser."""
from __future__ import annotations

import json
from pathlib import Path

from researcher_mapper.models.schemas import CandidateScore, ResearcherProfile


def _score_to_dict(score: CandidateScore) -> dict:
    return {
        "researcher_id": score.candidate_id,
        "name": score.name,
        "institution": score.institution,
        "country_code": score.country_code,
        "department": score.department,
        "israel_affiliated": score.israel_affiliated,
        "base_similarity": round(score.base_similarity, 4),
        "same_area_score": round(score.same_area_score, 4),
        "complementary_topic_score": round(score.complementary_topic_score, 4),
        "seniority_score": round(score.seniority_score, 4),
        "reputation_score": round(score.reputation_score, 4),
        "familiarity_proxy": round(score.familiarity_proxy, 4),
        "candidate_cites_target": score.candidate_cites_target,
        "bucket_scores": {
            k: round(v, 4)
            for k, v in score.bucket_scores.model_dump().items()
        },
        "assigned_bucket": score.assigned_bucket,
        "reasons": score.reasons,
    }


def render_results_json(
    target: ResearcherProfile,
    israel_list: list[CandidateScore],
    world_list: list[CandidateScore],
    resolution_confidence: float = 0.0,
) -> dict:
    return {
        "target": {
            "name": target.canonical_name,
            "openalex_id": target.openalex_author_id,
            "orcid": target.orcid,
            "institution": (
                target.current_institution.name if target.current_institution else None
            ),
            "country_code": (
                target.current_institution.country_code
                if target.current_institution
                else None
            ),
            "department": target.current_department,
            "israel_affiliated": target.israel_affiliated,
            "works_count": target.works_count,
            "cited_by_count": target.cited_by_count,
            "resolution_confidence": round(resolution_confidence, 4),
        },
        "israel_top20": [_score_to_dict(c) for c in israel_list],
        "world_top20": [_score_to_dict(c) for c in world_list],
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
