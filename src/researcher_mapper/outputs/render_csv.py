"""CSV output for the two ranked lists."""
from __future__ import annotations

import csv
from pathlib import Path

from researcher_mapper.models.schemas import CandidateScore

_BUCKET_LABELS = {
    "dept_mentors": "Institutional Mentors",
    "area_mentors": "Area Mentors",
    "recommendation_letter_writers": "Letter Writers",
    "in_area_collaborators": "In-Area Collaborators",
    "interdisciplinary_collaborators": "Interdisciplinary Collaborators",
}

_FIELDS = [
    "rank",
    "name",
    "institution",
    "country_code",
    "department",
    "assigned_bucket",
    "base_similarity",
    "same_area_score",
    "seniority_score",
    "reputation_score",
    "score_in_area_collaborators",
    "score_interdisciplinary",
    "score_letter_writers",
    "score_dept_mentors",
    "score_area_mentors",
    "researcher_id",
]


def _row(rank: int, score: CandidateScore) -> dict:
    bs = score.bucket_scores
    assigned_bucket = score.assigned_bucket or ""
    return {
        "rank": rank,
        "name": score.name,
        "institution": score.institution or "",
        "country_code": score.country_code or "",
        "department": score.department or "",
        "assigned_bucket": _BUCKET_LABELS.get(assigned_bucket, assigned_bucket),
        "base_similarity": round(score.base_similarity, 4),
        "same_area_score": round(score.same_area_score, 4),
        "seniority_score": round(score.seniority_score, 4),
        "reputation_score": round(score.reputation_score, 4),
        "score_in_area_collaborators": round(bs.in_area_collaborators, 4),
        "score_interdisciplinary": round(bs.interdisciplinary_collaborators, 4),
        "score_letter_writers": round(bs.recommendation_letter_writers, 4),
        "score_dept_mentors": round(bs.dept_mentors, 4),
        "score_area_mentors": round(bs.area_mentors, 4),
        "researcher_id": score.candidate_id,
    }


def save_csv(scores: list[CandidateScore], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for rank, score in enumerate(scores, start=1):
            writer.writerow(_row(rank, score))
