"""Fetch lightweight summaries for a pool of candidate researchers."""
from __future__ import annotations

import re
from datetime import datetime as _dt

from researcher_mapper.api.openalex import OpenAlexClient
from researcher_mapper.models.schemas import CandidateSummary, InstitutionRef

_CURRENT_YEAR = _dt.utcnow().year


# Department-related keywords that indicate an institution name encodes a sub-unit.
# If the portion after the first comma contains any of these, we treat it as a
# department hint rather than a city / country qualifier.
_DEPT_SIGNALS = {
    "department", "faculty", "school", "division", "laboratory", "lab",
    "institute", "college",
    "computer science", "electrical", "mathematics", "physics", "biology",
    "chemistry", "engineering", "medicine", "psychology", "economics",
    "statistics", "neuroscience", "mechanical", "civil", "industrial",
}

# Institution name substrings that reliably indicate a for-profit tech company.
# Used as a fallback when OpenAlex's institution_type field is absent or wrong.
_INDUSTRY_NAME_KEYWORDS: frozenset[str] = frozenset({
    "facebook", "meta ai", "meta platforms",
    "google", "deepmind", "google brain", "google research",
    "apple inc", "apple research",
    "microsoft research", "microsoft corporation",
    "amazon", "aws research",
    "openai",
    "nvidia research",
    "ibm research",
    "salesforce research",
    "adobe research",
    "bytedance", "tiktok",
    "tencent",
    "huawei",
    "qualcomm",
    "intel labs",
    "samsung research",
    "twitter",
    "linkedin",
    "snap research",
    "uber ai",
})


def _is_industry_affiliation(aff_inst: dict) -> bool:
    """Return True if an affiliation institution is a for-profit tech company."""
    if aff_inst.get("type") == "company":
        return True
    name = (aff_inst.get("display_name") or "").lower()
    return any(kw in name for kw in _INDUSTRY_NAME_KEYWORDS)


def _extract_dept_from_inst_name(display_name: str) -> str:
    """
    Try to extract a department hint from an OpenAlex institution display name.

    OpenAlex sometimes points an author's last_known_institution to a sub-unit
    record whose name encodes the department, e.g.:
        "Technion – Israel Institute of Technology, Department of Computer Science"
        "MIT, Computer Science and Artificial Intelligence Laboratory"

    We take the substring after the first comma and check whether it looks like
    a department rather than a geographic qualifier (city, country).
    Returns the lower-cased sub-unit string, or "" if nothing useful is found.
    """
    if not display_name:
        return ""
    if "," not in display_name:
        return ""
    sub_unit = display_name.split(",", 1)[1].strip().lower()
    if any(signal in sub_unit for signal in _DEPT_SIGNALS):
        return sub_unit
    return ""


def _is_israel_affiliated(raw: dict) -> bool:
    """
    Return True if the researcher is currently or substantively based at an
    Israeli institution.

    Four checks in order of reliability:
    1. last_known_institution (current, singular) — corroborated by affiliations
    2. last_known_institutions (current, plural) — corroborated by affiliations
    3. affiliations[].years — sustained recent IL affiliation.  Requires:
         • most recent affiliated year ≥ 2023
         • at least 2 years of IL affiliation (filters 1-year visitors / short stays)
    4. affiliations[].years — broader OpenAlex-lag catch.  Requires:
         • most recent affiliated year ≥ 2021
         • at least 2 years of IL affiliation

    Checks 1 and 2 require corroboration from publication-affiliation history
    (affiliations[]) to guard against stale or incorrect OpenAlex data.
    Example: a researcher whose last_known_institution was updated to
    "Hebrew University of Jerusalem" based on a single visiting-year paper
    should not be classified as Israeli.  If the affiliations list is non-empty
    we require at least one non-company IL entry with ≥ 2 years before accepting
    the current-institution signal.  If the affiliations list is completely empty
    (very new researcher with no publication history) we trust Checks 1/2 as-is.
    """
    affiliations = raw.get("affiliations") or []

    def _il_years_in_affiliations() -> int:
        """Return the max number of years at any single IL non-company institution."""
        max_years = 0
        for aff in affiliations:
            inst = aff.get("institution") or {}
            years = aff.get("years") or []
            if inst.get("country_code") == "IL" and not _is_industry_affiliation(inst):
                max_years = max(max_years, len(years))
        return max_years

    # Check 1: current institution (singular), corroborated by affiliation history.
    lki = raw.get("last_known_institution") or {}
    if lki.get("country_code") == "IL" and not _is_industry_affiliation(lki):
        if not affiliations:
            # No publication-affiliation history to contradict — trust the field.
            return True
        if _il_years_in_affiliations() >= 2:
            return True
        # last_known_institution says IL, but publication history shows < 2 years
        # at any IL institution — likely a visiting appointment or stale data; skip.

    # Check 2: current institutions (plural — newer API field), same corroboration.
    has_il_in_lki_plural = any(
        isinstance(inst, dict)
        and inst.get("country_code") == "IL"
        and not _is_industry_affiliation(inst)
        for inst in (raw.get("last_known_institutions") or [])
    )
    if has_il_in_lki_plural:
        if not affiliations:
            return True
        if _il_years_in_affiliations() >= 2:
            return True
        # Same reasoning as Check 1: < 2 years in affiliations → skip.

    # Check 3: sustained recent affiliation (≥2 years AND most recent ≥2023).
    # Requires at least 2 years to filter out 1-year sabbatical visitors while
    # catching researchers who joined IL institutions recently (e.g. 2022-2023).
    # Company affiliations (Meta Israel, Apple Israel, etc.) are excluded: working
    # at a tech company's Israel office is not the same as academic IL affiliation.
    for aff in raw.get("affiliations") or []:
        inst = aff.get("institution") or {}
        years = aff.get("years") or []
        if not years or inst.get("country_code") != "IL":
            continue
        if _is_industry_affiliation(inst):
            continue
        if len(years) >= 2 and max(years) >= 2023:
            return True

    # Check 4: substantial IL affiliation with most recent year ≥ 2021.
    # OpenAlex records often lag by 2-3 years; a researcher with Hebrew University
    # affiliations up to 2022 is almost certainly still there even if
    # last_known_institution is not yet set.  We require ≥ 2 years to filter out
    # short sabbatical visits.
    # Company affiliations are excluded for the same reason as Check 3.
    for aff in raw.get("affiliations") or []:
        inst = aff.get("institution") or {}
        years = aff.get("years") or []
        if not years or inst.get("country_code") != "IL":
            continue
        if _is_industry_affiliation(inst):
            continue
        if len(years) >= 2 and max(years) >= 2021:
            return True

    # Check 5: cross-institution total IL years.
    # Researchers who held multiple short IL positions (e.g. 1 year at Technion,
    # 1 year at TAU) fail the per-institution >= 2-year checks above, but the
    # aggregate signal clearly indicates sustained IL presence.  Count total
    # non-company IL affiliation years across ALL institutions.
    total_il_years = 0
    max_il_year = 0
    for aff in raw.get("affiliations") or []:
        inst = aff.get("institution") or {}
        years = aff.get("years") or []
        if not years or inst.get("country_code") != "IL":
            continue
        if _is_industry_affiliation(inst):
            continue
        total_il_years += len(years)
        max_il_year = max(max_il_year, max(years))
    if total_il_years >= 2 and max_il_year >= 2021:
        return True

    return False


def _parse_candidate_summary(raw: dict) -> CandidateSummary:
    """Convert a raw OpenAlex author dict into a CandidateSummary."""
    oa_id = raw.get("id", "")
    orcid = raw.get("orcid")

    # Institution — prefer last_known_institution (singular).
    # Fall back to last_known_institutions (plural) when singular is absent:
    # many researchers only have the plural field populated.
    #
    # When iterating the plural list, skip healthcare, nonprofit, and other
    # non-academic types — OpenAlex disambiguation errors sometimes place an
    # ML researcher's record at a hospital or advocacy organisation.  Only
    # accept education / government / facility / unknown types.
    _ACADEMIC_INST_TYPES = {"education", "government", "facility", None}

    last_inst_raw = raw.get("last_known_institution") or {}
    if not last_inst_raw:
        lkis = raw.get("last_known_institutions") or []
        for lki_entry in lkis:
            if isinstance(lki_entry, dict) and lki_entry.get("type") in _ACADEMIC_INST_TYPES:
                last_inst_raw = lki_entry
                break
    current_institution = None
    if last_inst_raw:
        inst_display = last_inst_raw.get("display_name") or "Unknown"
        dept_hint = _extract_dept_from_inst_name(inst_display)
        current_institution = InstitutionRef(
            openalex_id=last_inst_raw.get("id"),
            ror=last_inst_raw.get("ror"),
            name=inst_display,
            country_code=last_inst_raw.get("country_code"),
            institution_type=last_inst_raw.get("type"),  # "education", "company", etc.
            department=dept_hint or None,
        )

    affiliations_raw = raw.get("affiliations") or []

    # ── Skeptic guard: replace weakly-corroborated last_known_institution ─────
    # last_known_institution is sometimes stale (researcher left years ago) or
    # outright wrong (OpenAlex name-disambiguation error assigns a researcher
    # to an institution they have no recent connection to).
    #
    # Rule: if the affiliations list has ≥ 3 entries but the current institution
    # has NOT appeared there with a year ≤ 5 years ago, the field is weakly
    # corroborated.  Replace it with the most-recent non-company affiliation
    # from the history, or blank it out when no suitable replacement exists.
    #
    # Guard: skip when the affiliations list is sparse (< 3 entries) — too little
    # evidence to second-guess last_known_institution (e.g. new researcher whose
    # recent appointment hasn't propagated into OpenAlex's affiliation records yet).
    if current_institution and len(affiliations_raw) >= 3:
        inst_oa_id = (current_institution.openalex_id or "").split("/")[-1]
        inst_name_lower = (current_institution.name or "").lower()[:30]

        best_year_for_current = 0
        for aff in affiliations_raw:
            aff_inst = aff.get("institution") or {}
            years = aff.get("years") or []
            if not years:
                continue
            aff_id = aff_inst.get("id", "").split("/")[-1]
            aff_name_lower = (aff_inst.get("display_name") or "").lower()[:30]
            if (inst_oa_id and aff_id == inst_oa_id) or (
                inst_name_lower and inst_name_lower == aff_name_lower
            ):
                best_year_for_current = max(best_year_for_current, max(years))

        if 0 < best_year_for_current < _CURRENT_YEAR - 5:
            # Institution was found in affiliations but the last entry is stale
            # (researcher left this institution years ago).
            # Note: best_year_for_current == 0 means the institution is absent from
            # affiliations entirely — this is ambiguous (could be a new appointment
            # not yet propagated into OpenAlex) so we leave it alone.
            # Weakly corroborated — find the most recent non-company affiliation
            best_repl_year = 0
            best_repl_inst: dict = {}
            for aff in affiliations_raw:
                aff_inst = aff.get("institution") or {}
                years = aff.get("years") or []
                if not years or aff_inst.get("type") not in _ACADEMIC_INST_TYPES:
                    continue
                max_y = max(years)
                if max_y > best_repl_year:
                    best_repl_year = max_y
                    best_repl_inst = aff_inst

            if best_repl_inst and best_repl_year >= _CURRENT_YEAR - 7:
                inst_display = best_repl_inst.get("display_name") or "Unknown"
                dept_hint = _extract_dept_from_inst_name(inst_display)
                current_institution = InstitutionRef(
                    openalex_id=best_repl_inst.get("id"),
                    ror=best_repl_inst.get("ror"),
                    name=inst_display,
                    country_code=best_repl_inst.get("country_code"),
                    institution_type=best_repl_inst.get("type"),
                    department=dept_hint or None,
                )
            else:
                current_institution = None

    # Override (or create) institution with type="company" if the researcher has
    # a recent industry affiliation that is not superseded by a more recent
    # academic one.  This catches:
    #   - researchers who moved to industry but whose last_known_institution
    #     still reflects their old academic position, AND
    #   - researchers with NO last_known_institution but whose affiliations
    #     history shows a recent tech-company entry (wider lookback because
    #     industry researchers publish less consistently in OpenAlex).
    # Detection uses both OpenAlex's institution_type AND name keywords because
    # OpenAlex sometimes leaves institution_type unset for known tech companies.
    #
    # Guard: if the researcher's MOST RECENT academic affiliation is newer than
    # their most recent industry one, they have moved back to academia — do not
    # flag as industry.  Prevents false positives for professors who once spent
    # a year at a tech company.
    lki_is_none = current_institution is None
    industry_lookback = _CURRENT_YEAR - (5 if lki_is_none else 3)

    best_company_year = 0
    best_company_inst: dict = {}
    best_academic_year = 0
    best_academic_inst: dict = {}
    for aff in affiliations_raw:
        aff_inst = aff.get("institution") or {}
        years = aff.get("years") or []
        if not years:
            continue
        max_year = max(years)
        if _is_industry_affiliation(aff_inst) and max_year >= industry_lookback:
            # Require ≥ 2 years of industry presence before treating a
            # company affiliation as a genuine career move.  A single-year
            # industry entry is typically an OpenAlex name-disambiguation
            # artefact — an unrelated person at a company merged into the
            # researcher's record — and must not override their true
            # academic institution.  Mirrors the ≥2-year rule used by
            # _is_israel_affiliated (SPECS §5 Checks 3/4).
            if len(years) < 2:
                continue
            if max_year > best_company_year:
                best_company_year = max_year
                best_company_inst = aff_inst
        elif aff_inst.get("type") == "education" and max_year > best_academic_year:
            best_academic_year = max_year
            best_academic_inst = aff_inst

    # Only treat as industry when the company year is STRICTLY more recent than
    # the latest academic year.  When both are equal, the researcher has concurrent
    # academic and industry affiliations (common for Israeli professors consulting
    # at tech companies) — default to academic in that case.
    if best_company_year and best_company_year > best_academic_year:
        if current_institution is not None:
            current_institution.institution_type = "company"
        else:
            current_institution = InstitutionRef(
                name=best_company_inst.get("display_name") or "Unknown",
                country_code=best_company_inst.get("country_code"),
                institution_type="company",
            )
    elif lki_is_none and best_academic_inst and best_academic_year >= _CURRENT_YEAR - 5:
        # last_known_institution was absent AND no industry override fired —
        # use the most recent academic affiliation (≤ 5 years) so the candidate
        # has a proper country_code and institution for routing / display.
        # Without this, researchers whose OpenAlex last_known_institution field
        # is blank get routed purely on israel_affiliated, with no institution
        # shown in the output CSV.
        inst_display = best_academic_inst.get("display_name") or "Unknown"
        dept_hint = _extract_dept_from_inst_name(inst_display)
        current_institution = InstitutionRef(
            openalex_id=best_academic_inst.get("id"),
            ror=best_academic_inst.get("ror"),
            name=inst_display,
            country_code=best_academic_inst.get("country_code"),
            institution_type=best_academic_inst.get("type"),
            department=dept_hint or None,
        )

    # Corroborate current_institution.country_code against the affiliations history.
    # If the institution's country does not appear in ANY affiliation entry (and the
    # affiliation list is non-empty), the last_known_institution is likely wrong or
    # stale OpenAlex data — null out the country_code so that the Israel/World routing
    # in assignment.py falls back to israel_affiliated rather than trusting a bogus
    # country (the affiliation history is more reliable).
    # Guard: if affiliations is empty we have no evidence to contradict — trust the field.
    if current_institution and current_institution.country_code and affiliations_raw:
        _affil_countries = {
            (a.get("institution") or {}).get("country_code")
            for a in affiliations_raw
        }
        if current_institution.country_code not in _affil_countries:
            current_institution.country_code = None

    israel_affiliated = _is_israel_affiliated(raw)

    # Topics (T-prefixed IDs — newer field; count-normalized to match publication topic_vector)
    top_topics: dict[str, float] = {}
    raw_topics = raw.get("topics", []) or []
    for t in raw_topics:
        tid = t.get("id")
        count = t.get("count", 0)
        if tid and count > 0:
            top_topics[tid] = float(count)
    if top_topics:
        total = sum(top_topics.values())
        top_topics = {k: v / total for k, v in top_topics.items()}

    # Concepts (C-prefixed IDs — older x_concepts field; kept as fallback)
    top_concepts: dict[str, float] = {}
    for c in raw.get("x_concepts", []) or []:
        cid = c.get("id")
        score = c.get("score", 0.0)
        if cid:
            top_concepts[cid] = score / 100.0

    # Estimate career span from counts_by_year.
    # OpenAlex counts_by_year covers only the last ~10 years.  If a researcher
    # has works (works_count > 0) but none appear in that window, they haven't
    # published recently.  We infer last_pub as (current_year - 11) so the
    # inactivity gate in assignment.py correctly filters deceased / long-retired
    # researchers whose citation counts would otherwise make them appear senior.
    counts_by_year: list[dict] = raw.get("counts_by_year", []) or []
    years_raw_sorted = sorted(
        int(row["year"]) for row in counts_by_year if row.get("works_count", 0) > 0
    )
    # Drop isolated early-year outliers that are typically OpenAlex record-merge
    # artifacts.  A legitimate career has continuous publication activity; a
    # ≥ 5-year gap between the earliest year and the next cluster indicates the
    # early entries come from a different person merged into this record.
    # Without this, first_pub_year is set to the outlier year and inflates
    # career_age_norm in the seniority score — letting PhD students through
    # the min_seniority gate.  Gaps ≥ 10 years also trigger the topic-
    # contamination guard below.
    years_with_works = list(years_raw_sorted)
    major_leading_gap = False
    while len(years_with_works) >= 2 and years_with_works[1] - years_with_works[0] >= 5:
        if years_with_works[1] - years_with_works[0] >= 10:
            major_leading_gap = True
        years_with_works = years_with_works[1:]
    first_pub = min(years_with_works) if years_with_works else None
    if years_with_works:
        last_pub = max(years_with_works)
    elif raw.get("works_count", 0) > 0:
        # Has historical publications but none in the recent tracked window —
        # treat as inactive (last pub > 10 years ago).
        last_pub = _CURRENT_YEAR - 11
    else:
        last_pub = None

    # ── Record-contamination guard ────────────────────────────────────────────
    # OpenAlex sometimes merges a living researcher's record with historical
    # publications from an unrelated person who happened to share the same name
    # (e.g. a 19th-century author, a historical figure, a deceased predecessor).
    # The misattributed publications corrupt the author.topics aggregation:
    # OpenAlex computes author-level topic weights by aggregating across ALL
    # attributed works, so even a handful of off-topic historical papers can
    # introduce foreign topics with large weight.
    #
    # Detection: TWO signals fire this guard —
    #   (a) raw first_pub_year < current_year − 55 (impossible career span)
    #   (b) a ≥ 10-year gap between the earliest year and the rest of the
    #       publication record (leading outlier indicates a merged-in paper
    #       from a different person).
    # Clearing top_topics / top_concepts prevents the corrupted topic vector
    # from inflating same_area scores against any target.  The researcher's
    # other attributes (institution, seniority, reputation) are unaffected.
    _MAX_CAREER_YEARS = 55
    first_pub_raw = min(years_raw_sorted) if years_raw_sorted else None
    span_too_long = (
        first_pub_raw is not None
        and (_CURRENT_YEAR - first_pub_raw) > _MAX_CAREER_YEARS
    )
    if span_too_long or major_leading_gap:
        top_topics = {}
        top_concepts = {}

    return CandidateSummary(
        openalex_id=oa_id,
        name=raw.get("display_name", ""),
        orcid=orcid,
        works_count=raw.get("works_count", 0),
        cited_by_count=raw.get("cited_by_count", 0),
        current_institution=current_institution,
        israel_affiliated=israel_affiliated,
        top_topics=top_topics,
        top_concepts=top_concepts,
        first_pub_year=first_pub,
        last_pub_year=last_pub,
    )


def fetch_candidate_summaries(
    author_ids: list[str],
    oa_client: OpenAlexClient | None = None,
) -> list[CandidateSummary]:
    """
    Batch-fetch OpenAlex author summaries for a list of author IDs.
    Falls back to individual requests (OpenAlex doesn't have a true batch endpoint
    for authors, but filter by OR is supported up to ~50 IDs at a time).
    """
    if not author_ids:
        return []

    oa = oa_client or OpenAlexClient()
    summaries: list[CandidateSummary] = []
    batch_size = 50

    for i in range(0, len(author_ids), batch_size):
        batch = author_ids[i : i + batch_size]
        # OpenAlex supports | (OR) in filter values
        ids_clean = [aid.split("/")[-1] for aid in batch]
        filter_str = "openalex:" + "|".join(ids_clean)
        try:
            data = oa._get(
                "/authors",
                params={
                    "filter": filter_str,
                    "per-page": batch_size,
                },
            )
            for raw in data.get("results", []):
                summaries.append(_parse_candidate_summary(raw))
        except Exception:
            # Fall back to individual fetches for this batch
            for aid in batch:
                try:
                    raw = oa.get_author_summary(aid)
                    summaries.append(_parse_candidate_summary(raw))
                except Exception:
                    pass

    return summaries
