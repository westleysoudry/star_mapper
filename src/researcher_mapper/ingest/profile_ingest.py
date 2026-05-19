"""Ingest a researcher's profile and publications from OpenAlex + Crossref."""
from __future__ import annotations

from researcher_mapper.api.openalex import OpenAlexClient
from researcher_mapper.api.crossref import CrossrefClient
from researcher_mapper.models.schemas import (
    InstitutionRef,
    Publication,
    ResearcherProfile,
)


def _infer_dept_from_works(author_id: str, works_raw: list[dict]) -> str:
    """
    Try to infer the target author's department from the institution names
    listed in the authorship entries of their own papers.

    OpenAlex's work-level authorships often use sub-institution records
    (e.g. "Technion – Faculty of Electrical and Computer Engineering") even
    when the author-level ``last_known_institution`` is just the root
    university.  We vote across all papers and return the most common hit.
    """
    from researcher_mapper.ingest.candidate_ingest import _extract_dept_from_inst_name

    short_id = (author_id or "").split("/")[-1]
    if not short_id:
        return ""
    dept_votes: dict[str, int] = {}

    for work in works_raw:
        for authorship in work.get("authorships", []):
            auth_obj = authorship.get("author") or {}
            if (auth_obj.get("id") or "").split("/")[-1] != short_id:
                continue
            for inst in authorship.get("institutions", []):
                dept = _extract_dept_from_inst_name(inst.get("display_name") or "")
                if dept:
                    dept_votes[dept] = dept_votes.get(dept, 0) + 1

    if not dept_votes:
        return ""
    return max(dept_votes, key=lambda k: dept_votes[k])


def _parse_institution(raw: dict | None) -> InstitutionRef | None:
    if not raw:
        return None
    return InstitutionRef(
        openalex_id=raw.get("id"),
        ror=raw.get("ror"),
        name=raw.get("display_name", "Unknown"),
        country_code=raw.get("country_code"),
    )


def _parse_work(work: dict) -> Publication:
    author_ids = [
        a.get("author", {}).get("id", "")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("id")
    ]
    institution_ids = [
        inst.get("id", "")
        for a in work.get("authorships", [])
        for inst in a.get("institutions", [])
        if inst.get("id")
    ]
    # Topics (newer OpenAlex field)
    topic_ids = [
        t.get("id", "")
        for t in work.get("topics", [])
        if t.get("id")
    ]
    # Concepts (older fallback field)
    concept_ids = [
        c.get("id", "")
        for c in work.get("concepts", [])
        if c.get("id")
    ]
    venue = None
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    if source:
        venue = source.get("display_name")

    return Publication(
        id=work.get("id", ""),
        doi=work.get("doi"),
        title=work.get("title") or "",
        year=work.get("publication_year"),
        venue=venue,
        author_ids=author_ids,
        institution_ids=institution_ids,
        topic_ids=topic_ids,
        concept_ids=concept_ids,
        referenced_work_ids=work.get("referenced_works", []) or [],
        cited_by_count=work.get("cited_by_count", 0),
    )


def ingest_openalex_profile(
    author_id: str,
    from_year: int = 2014,
    max_works: int = 500,
) -> ResearcherProfile:
    """
    Fetch and parse an OpenAlex author record + their recent publications.
    Returns a ResearcherProfile with publication list populated.
    """
    oa = OpenAlexClient()
    author = oa.get_author(author_id)

    institutions: list[InstitutionRef] = []
    for aff in author.get("affiliations", []) or []:
        inst = _parse_institution(aff.get("institution"))
        if inst:
            institutions.append(inst)

    current_institution = _parse_institution(author.get("last_known_institution"))

    # Fallback: if last_known_institution is absent, use the most recent
    # affiliation entry (sorted by the latest year in its years list).
    if current_institution is None and institutions:
        def _latest_year(aff_raw: dict) -> int:
            years = aff_raw.get("years") or []
            return max(years) if years else 0

        sorted_affs = sorted(
            author.get("affiliations", []) or [],
            key=_latest_year,
            reverse=True,
        )
        for aff_raw in sorted_affs:
            inst = _parse_institution(aff_raw.get("institution"))
            if inst:
                current_institution = inst
                break

    works_raw = oa.get_author_works(author_id, from_year=from_year, max_works=max_works)
    publications = [_parse_work(w) for w in works_raw]

    # Try to infer the researcher's department from their own authorship entries.
    # Work-level authorships often include sub-institution names (e.g. faculty/
    # department) even when the author-level record only has the root university.
    # This value is stored as a fallback; a CV or faculty-page provides the
    # authoritative department and will override it (see run_target._apply_cv_data).
    inferred_department = _infer_dept_from_works(author_id, works_raw)

    years = [p.year for p in publications if p.year]
    first_pub = min(years) if years else None
    last_pub = max(years) if years else None

    # Build top_topics from author-level topics (T-prefixed, newer field).
    # Count-normalised to match the CandidateSummary representation so that
    # weighted Jaccard comparisons are apples-to-apples.
    top_topics: dict[str, float] = {}
    for t in author.get("topics", []) or []:
        tid = t.get("id")
        count = t.get("count", 0)
        if tid and count > 0:
            top_topics[tid] = float(count)
    if top_topics:
        total_t = sum(top_topics.values())
        top_topics = {k: v / total_t for k, v in top_topics.items()}

    # Build top_concepts from author-level x_concepts (C-prefixed, older fallback)
    top_concepts: dict[str, float] = {}
    for concept in author.get("x_concepts", []) or []:
        cid = concept.get("id")
        score = concept.get("score", 0.0)
        if cid:
            top_concepts[cid] = score / 100.0  # OA scores are 0–100

    israel_affiliated = (
        current_institution is not None
        and current_institution.country_code == "IL"
    )

    return ResearcherProfile(
        canonical_name=author.get("display_name", ""),
        openalex_author_id=author.get("id"),
        orcid=author.get("orcid"),
        institutions=institutions,
        current_institution=current_institution,
        current_department=inferred_department or None,
        publications=publications,
        works_count=author.get("works_count", 0),
        cited_by_count=author.get("cited_by_count", 0),
        first_pub_year=first_pub,
        last_pub_year=last_pub,
        top_topics=top_topics,
        top_concepts=top_concepts,
        israel_affiliated=israel_affiliated,
    )


def backfill_crossref_metadata(publications: list[Publication]) -> list[Publication]:
    """
    Enrich publications with Crossref abstract where the OA record has no abstract.
    Operates in-place (returns the same list for chaining).
    """
    cr = CrossrefClient()
    for pub in publications:
        if pub.abstract or not pub.doi:
            continue
        try:
            data = cr.get_work(pub.doi)
            message = data.get("message", {})
            abstract = cr.extract_abstract(message)
            if abstract:
                pub.abstract = abstract
        except Exception:
            pass
    return publications
