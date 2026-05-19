"""Cross-validate OpenAlex candidate data against ORCID and Semantic Scholar.

Two independent checks:

1. **ORCID institution validation** (candidates with ORCID IDs).
   - Employment: if ORCID's current employer disagrees with OpenAlex's
     institution, update the display name.  ORCID employments are self-reported
     and typically accurate.
   - Note: ORCID works counts are NOT used for merge detection — many
     legitimate researchers have very sparse ORCID profiles.

2. **S2 works-count validation** (all candidates).
   - Search S2 by name, match by citation-count proximity, and compare paper
     counts.  When OpenAlex has far more papers than S2 for the same person,
     the OpenAlex record likely merges multiple namesakes — clear topic vectors.
"""
from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from researcher_mapper.api.orcid import OrcidClient
from researcher_mapper.api.semanticscholar import SemanticScholarClient
from researcher_mapper.models.schemas import CandidateSummary, InstitutionRef

log = logging.getLogger(__name__)

# OpenAlex works_count / S2 paper_count above this ratio triggers
# the merged-record guard (topic vectors cleared).
_MERGE_RATIO_THRESHOLD = 2.5

# S2 search: minimum match quality to accept an author result.
_S2_MIN_MATCH_SCORE = 0.45


# ── ORCID validation (institution only) ──────────────────────────────────────

def _orcid_check_one(
    summary: CandidateSummary,
    orcid_client: OrcidClient,
) -> dict | None:
    """Fetch ORCID employments for a single candidate.

    Uses the client's get_record / extract_affiliations (which include
    retries), but the client is created with a short 10-second timeout
    so a single stall does not block the pipeline.
    """
    if not summary.orcid:
        return None
    try:
        record = orcid_client.get_record(summary.orcid)
        affs = orcid_client.extract_affiliations(record)

        # Current employment = first entry with end_year is None
        current_org = None
        for aff in affs:
            if aff.get("end_year") is None and aff.get("org_name"):
                current_org = aff
                break

        return {"current_org": current_org, "all_affs": affs}
    except Exception as exc:
        log.debug("ORCID fetch failed for %s (%s): %s", summary.name, summary.orcid, exc)
        return None


def _apply_orcid_result(summary: CandidateSummary, orcid_data: dict) -> None:
    """Apply ORCID institution validation to a CandidateSummary in-place."""
    current_org = orcid_data.get("current_org")
    if not current_org:
        return

    orcid_name = current_org.get("org_name") or ""
    if not orcid_name:
        return

    def _orcid_institution() -> InstitutionRef:
        return InstitutionRef(
            name=orcid_name,
            ror=current_org.get("ror"),
            country_code=None,
            institution_type=None,
        )

    if summary.current_institution is None:
        # OpenAlex had no institution — fill from ORCID
        log.info(
            "ORCID cross-check: filling institution for %s → %s",
            summary.name, orcid_name,
        )
        summary.current_institution = _orcid_institution()
        return

    # Compare normalised names to detect real mismatches while ignoring
    # punctuation variations (en-dash "–" vs hyphen, etc.) and sub-unit
    # suffixes ("Columbia University" vs "Columbia University Columbia College").
    def _norm(s: str) -> str:
        s = re.sub(r"[^a-z0-9 ]", "", s.lower())
        return re.sub(r"\s+", " ", s).strip()

    na = _norm(summary.current_institution.name or "")
    nb = _norm(orcid_name)
    if na and nb and (na.startswith(nb) or nb.startswith(na)):
        return  # one is a prefix of the other — same institution

    log.info(
        "ORCID cross-check: institution mismatch for %s — "
        "OA='%s' vs ORCID='%s'; updating",
        summary.name,
        summary.current_institution.name,
        orcid_name,
    )
    summary.current_institution = _orcid_institution()


# ── S2 validation (merged-record detection) ──────────────────────────────────

def _name_score(oa_name: str, s2_name: str) -> float:
    """Score name similarity, handling abbreviated first names (e.g. "R. Meir")."""
    oa_parts = oa_name.lower().split()
    s2_parts = s2_name.lower().split()

    # Standard word-set overlap
    oa_set, s2_set = set(oa_parts), set(s2_parts)
    word_overlap = len(oa_set & s2_set) / max(len(oa_set | s2_set), 1)
    if word_overlap >= 0.5:
        return word_overlap

    # Abbreviated first name: same last name + first initial matches
    if len(oa_parts) >= 2 and len(s2_parts) >= 2:
        if oa_parts[-1] == s2_parts[-1]:  # last names match
            oa_first = oa_parts[0].rstrip(".")
            s2_first = s2_parts[0].rstrip(".")
            if oa_first.startswith(s2_first) or s2_first.startswith(oa_first):
                return 0.8  # strong: last name exact + first initial

    return word_overlap


def _s2_find_best_match(
    oa_name: str,
    oa_citations: int,
    s2_results: list[dict],
) -> dict | None:
    """Match the best S2 search result to an OpenAlex record by name + citations."""
    if not s2_results:
        return None

    best, best_score = None, -1.0

    for result in s2_results:
        s2_name = result.get("name") or ""
        name_overlap = _name_score(oa_name, s2_name)
        if name_overlap < 0.4:
            continue

        s2_cit = result.get("citationCount") or 0
        if oa_citations > 0 and s2_cit > 0:
            cit_sim = 1.0 - abs(math.log(s2_cit / oa_citations)) / math.log(10)
            cit_sim = max(cit_sim, 0.0)
        else:
            cit_sim = 0.2

        score = 0.5 * name_overlap + 0.5 * cit_sim
        if score > best_score:
            best_score = score
            best = result

    return best if best_score >= _S2_MIN_MATCH_SCORE else None


def _s2_check_one(
    summary: CandidateSummary,
    s2_client: SemanticScholarClient,
) -> dict | None:
    """Search S2 for a candidate and return matched paper count.

    Uses a raw httpx GET instead of the client's retrying _get() to avoid
    long retry chains when S2 is rate-limiting.  A single timeout is fine —
    merge detection is best-effort.
    """
    try:
        import httpx
        resp = s2_client.client.get(
            f"https://api.semanticscholar.org/graph/v1/author/search",
            params={
                "query": summary.name,
                "limit": 5,
                "fields": "name,paperCount,citationCount",
            },
            headers=s2_client._headers(),
        )
        if resp.status_code != 200:
            return None
        result = resp.json()
        matches = result.get("data", [])
        best = _s2_find_best_match(summary.name, summary.cited_by_count, matches)
        if best is None:
            return None
        return {"s2_papers": best.get("paperCount") or 0}
    except Exception as exc:
        log.debug("S2 fetch failed for %s: %s", summary.name, exc)
        return None


def _apply_s2_result(summary: CandidateSummary, s2_data: dict) -> None:
    """Apply S2 merged-record detection to a CandidateSummary in-place."""
    s2_papers = s2_data.get("s2_papers", 0)
    if s2_papers > 0 and summary.works_count > 0:
        ratio = summary.works_count / s2_papers
        if ratio >= _MERGE_RATIO_THRESHOLD:
            log.info(
                "S2 cross-check: probable merged record for %s — "
                "OA works=%d vs S2 papers=%d (ratio %.1f); clearing topics",
                summary.name, summary.works_count, s2_papers, ratio,
            )
            summary.top_topics = {}
            summary.top_concepts = {}


# ── Public API ────────────────────────────────────────────────────────────────

def crosscheck_candidates(
    summaries: list[CandidateSummary],
    orcid_client: OrcidClient | None = None,
    s2_client: SemanticScholarClient | None = None,
    max_workers: int = 4,
) -> None:
    """
    Cross-validate OpenAlex candidate summaries against ORCID and S2.

    Mutates summaries in-place:
    - ORCID: corrects institution display names (candidates with ORCID IDs).
    - S2: detects merged records via paper-count divergence (candidates without ORCID).

    Uses short timeouts (10s) to keep total runtime bounded — a single
    timed-out call should not stall the pipeline.
    """
    orcid = orcid_client or OrcidClient(timeout=10.0)
    s2 = s2_client or SemanticScholarClient(timeout=10.0)

    orcid_candidates = [s for s in summaries if s.orcid]
    # S2 merge check only for candidates WITHOUT ORCID — avoid redundant
    # API calls for the majority that already get ORCID validation.
    # Cap at 50 to keep runtime bounded (~1 min at S2's 1-req/s rate limit).
    s2_candidates = [s for s in summaries if not s.orcid][:50]

    log.info(
        "Cross-validation: %d ORCID lookups, %d S2 merge checks",
        len(orcid_candidates), len(s2_candidates),
    )

    # ── ORCID batch (institution validation only) ────────────────────────────
    orcid_results: dict[str, dict | None] = {}
    if orcid_candidates:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_orcid_check_one, s, orcid): s.openalex_id
                for s in orcid_candidates
            }
            for fut in as_completed(futures):
                oa_id = futures[fut]
                orcid_results[oa_id] = fut.result()

        for s in orcid_candidates:
            data = orcid_results.get(s.openalex_id)
            if data:
                _apply_orcid_result(s, data)

    # ── S2 sequential (merged-record detection, rate-limited) ──────────────
    # S2 is called sequentially with a short sleep between calls to stay
    # under the unauthenticated rate limit (~1 req/s).  Concurrent calls
    # would trigger 429 responses and lengthy retry backoffs.
    import time
    if s2_candidates:
        for s in s2_candidates:
            data = _s2_check_one(s, s2)
            if data:
                _apply_s2_result(s, data)
            time.sleep(0.5)  # ~2 req/s, well under the 429 threshold
