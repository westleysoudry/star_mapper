"""Resolve a target researcher across OpenAlex, Semantic Scholar, and ORCID."""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz.fuzz import token_sort_ratio

from researcher_mapper.api.openalex import OpenAlexClient
from researcher_mapper.api.semanticscholar import SemanticScholarClient


@dataclass
class IdentityResolutionResult:
    profile_candidates: list[dict] = field(default_factory=list)
    chosen: dict | None = None
    confidence: float = 0.0
    openalex_id: str | None = None
    s2_author_id: str | None = None
    orcid: str | None = None


def _normalize_orcid(orcid: str | None) -> str | None:
    if not orcid:
        return None
    clean = (
        orcid.replace("https://orcid.org/", "")
        .replace("http://orcid.org/", "")
        .strip()
        .strip("/")
    )
    return clean or None


def _score_openalex_candidate(
    candidate: dict,
    name: str,
    institution: str | None,
) -> float:
    name_score = (
        token_sort_ratio(name.lower(), candidate.get("display_name", "").lower())
        / 100.0
    )
    score = name_score

    if institution:
        aff_names = " ".join(
            (aff.get("institution") or {}).get("display_name", "")
            for aff in candidate.get("affiliations", [])
        )
        last_inst = candidate.get("last_known_institution") or {}
        inst_text = f"{aff_names} {last_inst.get('display_name', '')}".lower()
        inst_score = token_sort_ratio(institution.lower(), inst_text) / 100.0
        score = 0.70 * name_score + 0.30 * inst_score

    return round(score, 4)


def resolve_target_identity(
    name: str,
    institution: str | None = None,
    orcid: str | None = None,
) -> IdentityResolutionResult:
    """
    Attempt to match the researcher in OpenAlex and Semantic Scholar.

    When an ORCID is supplied, OpenAlex candidates must expose the same ORCID.
    A non-matching name-search result is treated as unresolved instead of being
    selected accidentally.
    """
    oa = OpenAlexClient()
    requested_orcid = _normalize_orcid(orcid)

    oa_candidates = oa.search_authors(name, per_page=25)
    if requested_orcid:
        oa_candidates = [
            c
            for c in oa_candidates
            if _normalize_orcid(c.get("orcid")) == requested_orcid
        ]

    scored: list[tuple[float, dict]] = [
        (_score_openalex_candidate(c, name, institution), c)
        for c in oa_candidates
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    chosen = scored[0][1] if scored else None
    confidence = scored[0][0] if scored else 0.0

    resolved_orcid = requested_orcid
    if not requested_orcid and chosen:
        resolved_orcid = _normalize_orcid(chosen.get("orcid"))

    s2_author_id: str | None = None
    try:
        s2 = SemanticScholarClient()
        s2_results = s2.search_author(name, limit=5).get("data", [])
        if s2_results:
            s2_author_id = s2_results[0].get("authorId")
    except Exception:
        pass

    openalex_id = None
    if chosen:
        openalex_id = chosen.get("id", "").split("/")[-1]

    return IdentityResolutionResult(
        profile_candidates=[c for _, c in scored[:10]],
        chosen=chosen,
        confidence=confidence,
        openalex_id=openalex_id,
        s2_author_id=s2_author_id,
        orcid=resolved_orcid,
    )
