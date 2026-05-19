"""
Full end-to-end pipeline.

Usage
-----
    # Python API
    from researcher_mapper.pipelines.run_target import run_target
    result = run_target("Yann LeCun", institution="New York University")

    # CLI
    python -m researcher_mapper.pipelines.run_target \
        --name "Yann LeCun" \
        --institution "New York University" \
        --output-dir ./output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from researcher_mapper.api.openalex import OpenAlexClient
from researcher_mapper.features.career_features import (
    career_feature_dict,
    estimate_reputation,
    estimate_seniority,
)
from researcher_mapper.features.citation_features import build_reference_vector
from researcher_mapper.features.network_features import (
    build_coauthor_vector,
    build_venue_vector,
    get_recent_coauthor_ids,
    last_author_fraction,
)
from researcher_mapper.features.similarity import (
    compute_base_similarity,
    compute_familiarity_proxy,
    compute_same_area_score,
)
from researcher_mapper.features.topic_features import (
    build_concept_vector,
    build_concept_vector_from_summary,
    build_topic_vector,
    complementary_topic_score,
)
from researcher_mapper.graph.build_graph import build_researcher_graph, save_graph_json
from researcher_mapper.identity.resolve_target import resolve_target_identity
from researcher_mapper.ingest.candidate_ingest import fetch_candidate_summaries
from researcher_mapper.ingest.profile_ingest import ingest_openalex_profile
from researcher_mapper.models.schemas import (
    BucketScores,
    CandidateScore,
    CandidateSummary,
    FeatureVector,
    ResearcherProfile,
)
from researcher_mapper.outputs.render_csv import save_csv
from researcher_mapper.outputs.render_json import render_results_json, save_json
from researcher_mapper.ranking.assignment import assign_israel_and_world
from researcher_mapper.ranking.bucket_scoring import compute_all_bucket_scores
from researcher_mapper.ranking.candidate_generation import retrieve_candidate_pool
from researcher_mapper.settings import load_bucket_caps, load_policy, load_weights

log = logging.getLogger(__name__)


# ── Feature builder helpers ────────────────────────────────────────────────────

def _build_target_feature_vector(profile: ResearcherProfile) -> FeatureVector:
    target_id = (profile.openalex_author_id or "").split("/")[-1]
    pubs = profile.publications

    # Build target's topic vector from publications capped to top-25.
    #
    # We intentionally do NOT use profile.top_topics (the OpenAlex author-level
    # summary) here, even when it's populated, because that summary only exposes
    # the top-5 topics — OpenAlex's per-author topic list is truncated far more
    # aggressively than the per-author x_concepts list.  A researcher's author
    # record may show only 5 topics while their papers collectively use 25+
    # relevant topic IDs; relying on the truncated list causes researchers whose
    # primary overlap falls outside the top-5 to be incorrectly suppressed.
    #
    # The publication vector capped at top-25 gives comprehensive coverage
    # without the Jaccard denominator inflation caused by 200+ tail topics.
    raw = build_topic_vector(pubs)
    if len(raw) > 25:
        top_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:25]
        total = sum(v for _, v in top_items)
        topic_vec = {k: v / total for k, v in top_items} if total else {}
    else:
        topic_vec = raw

    # Same cap for the C-prefixed concept vector (fallback for older records).
    raw_concept = build_concept_vector(pubs)
    if len(raw_concept) > 25:
        top_items = sorted(raw_concept.items(), key=lambda x: x[1], reverse=True)[:25]
        total = sum(v for _, v in top_items)
        concept_vec = {k: v / total for k, v in top_items} if total else {}
    else:
        concept_vec = raw_concept
    if not concept_vec:
        concept_vec = build_concept_vector_from_summary(profile.top_concepts)

    ref_vec = build_reference_vector(pubs)
    coauthor_vec = build_coauthor_vector(pubs, target_id)
    venue_vec = build_venue_vector(pubs)

    la_frac = last_author_fraction(pubs, target_id)
    career = career_feature_dict(
        profile.first_pub_year,
        profile.last_pub_year,
        profile.cited_by_count,
        profile.works_count,
        la_frac,
    )

    return FeatureVector(
        researcher_id=profile.openalex_author_id or "",
        topic_vector=topic_vec,
        concept_vector=concept_vec,
        coauthor_vector=coauthor_vec,
        citation_vector=ref_vec,
        venue_vector=venue_vec,
        career_features=career,
    )


def _build_candidate_feature_vector(summary: CandidateSummary) -> FeatureVector:
    """
    Build a feature vector for a candidate from their summary data only.

    Uses two separate ID namespaces so comparisons against the target stay
    in the same namespace:
      topic_vector  — T-prefixed IDs from author.topics  (matches target's publication topic_ids)
      concept_vector — C-prefixed IDs from author.x_concepts (fallback for older records)
    """
    topic_vec = build_concept_vector_from_summary(summary.top_topics)    # T-prefixed
    concept_vec = build_concept_vector_from_summary(summary.top_concepts) # C-prefixed
    career = career_feature_dict(
        summary.first_pub_year,
        summary.last_pub_year,
        summary.cited_by_count,
        summary.works_count,
        last_author_fraction=0.0,  # not available without full works
    )
    return FeatureVector(
        researcher_id=summary.openalex_id,
        topic_vector=topic_vec,
        concept_vector=concept_vec,
        career_features=career,
    )


def _generate_reasons(
    score: CandidateScore,
    candidate_name: str,
    bucket: str,
    shared_topic_ids: list[str] | None = None,
) -> list[str]:
    reasons: list[str] = []

    # CV-derived relationship is the strongest explicit signal — surface it first
    if score.cv_relationship:
        label_map = {"advisor": "PhD advisor", "postdoc_host": "postdoc host"}
        label = label_map.get(score.cv_relationship, score.cv_relationship)
        reasons.append(f"Known CV relationship: {label}")

    if score.same_area_score >= 0.60:
        area_str = f"Strong topic overlap (same-area {score.same_area_score:.2f})"
        if shared_topic_ids:
            short_ids = [tid.split("/")[-1] for tid in shared_topic_ids[:3]]
            area_str += f" — shared topics: {', '.join(short_ids)}"
        reasons.append(area_str)
    elif score.same_area_score >= 0.30:
        area_str = f"Moderate topic overlap (same-area {score.same_area_score:.2f})"
        if shared_topic_ids:
            short_ids = [tid.split("/")[-1] for tid in shared_topic_ids[:2]]
            area_str += f" — shared topics: {', '.join(short_ids)}"
        reasons.append(area_str)

    if score.seniority_score >= 0.70:
        reasons.append("Senior researcher (high career-age / citation profile)")
    if score.reputation_score >= 0.70:
        reasons.append("High citation impact relative to output")

    if score.familiarity_proxy >= 0.60 and not score.cv_relationship:
        reasons.append("Candidate has cited target's papers or shares frequent coauthors")
    elif score.familiarity_proxy >= 0.40 and not score.cv_relationship:
        reasons.append("Demonstrated familiarity with target's work (co-author or citation)")
    elif score.candidate_cites_target and not score.cv_relationship:
        reasons.append("Candidate has cited the target's work")

    if score.same_institution:
        reasons.append("Same institution as target")
    if score.same_department:
        reasons.append("Same department as target")
    if score.recent_coauthor:
        reasons.append("Recent co-authorship with target")
    if bucket == "interdisciplinary_collaborators":
        reasons.append(
            "Adjacent-domain fit "
            f"(same-area {score.same_area_score:.2f}, "
            f"complementary {score.complementary_topic_score:.2f})"
        )
    if bucket == "interdisciplinary_collaborators" and score.complementary_topic_score >= 0.55:
        reasons.append("Complementary research area — good interdisciplinary fit")
    return reasons or ["Selected by base similarity"]


def _top_shared_topic_ids(
    target_vec: dict[str, float],
    candidate_vec: dict[str, float],
    top_n: int = 3,
) -> list[str]:
    """Return the IDs of the top-N topics shared (by min weight) between two vectors."""
    shared = {
        k: min(target_vec[k], candidate_vec[k])
        for k in set(target_vec) & set(candidate_vec)
    }
    return sorted(shared, key=lambda k: (-shared[k], k))[:top_n]


def _score_sort_key(score: CandidateScore) -> tuple[float, float, float, str, str]:
    return (
        score.base_similarity,
        score.same_area_score,
        score.reputation_score,
        (score.name or "").lower(),
        score.candidate_id,
    )


def _match_cv_relationship(candidate_name: str, advisor_names: set[str]) -> str | None:
    """
    Return a CV relationship label if candidate_name matches a known advisor/host.
    Tries full-name match first, then last-name match (≥ 4 chars to avoid collisions).
    """
    cand_lower = candidate_name.lower().strip()
    for advisor in advisor_names:
        if advisor in cand_lower or cand_lower in advisor:
            return "advisor"
        adv_last = advisor.split()[-1] if advisor.split() else advisor
        cand_last = cand_lower.split()[-1] if cand_lower.split() else cand_lower
        if len(adv_last) >= 4 and adv_last == cand_last:
            return "advisor"
    return None


# ── Two-stage rerank ──────────────────────────────────────────────────────────

def _rerank_with_full_profiles(
    scores: list[CandidateScore],
    summaries_by_id: dict,
    target_fv: FeatureVector,
    citing_author_ids: set[str],
    target_recent_coauthors: set[str],
    target_department: str,
    target_institution_id: str,
    weights: dict,
    from_year: int,
    rerank_top_n: int = 60,
    stop_event=None,
) -> None:
    """
    Fetch full publication lists for the top-N candidates by base_similarity,
    rebuild their feature vectors with real citation/venue/coauthor overlap,
    and update CandidateScore objects in-place.

    This is a second-stage refinement: the first stage uses lightweight author
    summaries (topic vectors only); this stage adds real reference overlap,
    venue co-occurrence, and coauthor overlap which are zero in stage one.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from researcher_mapper.ingest.profile_ingest import ingest_openalex_profile

    if rerank_top_n <= 0:
        return

    top_scores = sorted(scores, key=_score_sort_key, reverse=True)[:rerank_top_n]
    ids_to_rerank = [s.candidate_id for s in top_scores]
    log.info("Two-stage rerank: fetching full profiles for top %d candidates …", len(ids_to_rerank))

    def _fetch(cand_id: str):
        try:
            return cand_id, ingest_openalex_profile(cand_id, from_year=from_year, max_works=200)
        except Exception as exc:
            log.debug("Rerank fetch failed for %s: %s", cand_id, exc)
            return cand_id, None

    full_profiles: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch, cid): cid for cid in ids_to_rerank}
        for future in as_completed(futures):
            if stop_event and stop_event.is_set():
                break
            cid, profile = future.result()
            if profile is not None:
                full_profiles[cid] = profile

    log.info("Two-stage rerank: refreshed %d/%d profiles", len(full_profiles), len(ids_to_rerank))

    score_map = {s.candidate_id: s for s in scores}
    for cand_id in ids_to_rerank:
        profile = full_profiles.get(cand_id)
        if profile is None:
            continue
        score = score_map.get(cand_id)
        if score is None:
            continue

        cand_fv = _build_target_feature_vector(profile)  # full pub-based vector
        cand_id_short = cand_id.split("/")[-1]
        candidate_cites_target = cand_id_short in citing_author_ids
        score.candidate_cites_target = candidate_cites_target

        base_sim = compute_base_similarity(
            target_fv, cand_fv,
            weights=weights.get("base_similarity"),
            candidate_cites_target=candidate_cites_target,
        )
        same_area = compute_same_area_score(
            target_fv, cand_fv,
            candidate_cites_target=candidate_cites_target,
        )
        comp_topic = complementary_topic_score(
            target_fv.concept_vector, cand_fv.concept_vector
        )

        # Real citation overlap (candidate_work_ids now available)
        candidate_work_ids = {p.id for p in profile.publications}
        familiarity = compute_familiarity_proxy(
            target_coauthor_vector=target_fv.coauthor_vector,
            candidate_openalex_id=cand_id,
            target_ref_vector=target_fv.citation_vector,
            candidate_work_ids=candidate_work_ids,
            candidate_cites_target=candidate_cites_target,
        )
        # Preserve CV-relationship bonus
        if score.cv_relationship:
            familiarity = min(familiarity + 0.3, 1.0)

        venue_sim = _weighted_jaccard_quick(target_fv.venue_vector, cand_fv.venue_vector)

        # Re-derive same_department with updated same_area (dept flags unchanged)
        cand_dept = score.department or ""
        same_department = bool(
            target_department and cand_dept and _dept_match(target_department, cand_dept)
        )
        dept_explicitly_different = _dept_explicitly_different(
            target_department, cand_dept, same_department
        )
        department_match = 1.0 if same_department else 0.0

        recent_coauthor = score.recent_coauthor
        no_conflict = 0.0 if recent_coauthor else 1.0
        institution_distance_bonus = (
            0.3 if (not score.same_institution and same_area >= 0.40) else 0.0
        )
        recency_overlap = 1.0 if recent_coauthor else same_area * 0.5

        bucket_score_dict = compute_all_bucket_scores(
            same_area=same_area,
            complementary_topic=comp_topic,
            seniority=score.seniority_score,
            reputation=score.reputation_score,
            no_conflict=no_conflict,
            familiarity_proxy=familiarity,
            collaboration=_weighted_jaccard_quick(
                target_fv.coauthor_vector, cand_fv.coauthor_vector
            ),
            venue=venue_sim,
            institution_distance_bonus=institution_distance_bonus,
            recency_overlap=recency_overlap,
            department_match=department_match,
            institutional_role=score.seniority_score,
            weights=weights,
        )

        # Store the updated feature vector on the summary for reason generation
        summary = summaries_by_id.get(cand_id)
        if summary is not None:
            summary.feature_vector = cand_fv

        # Mutate score in-place
        score.base_similarity = base_sim
        score.same_area_score = same_area
        score.complementary_topic_score = comp_topic
        score.familiarity_proxy = familiarity
        score.same_department = same_department
        score.dept_explicitly_different = dept_explicitly_different
        score.bucket_scores = BucketScores(**bucket_score_dict)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _check_stop(event: threading.Event | None) -> None:
    """Raise InterruptedError if the caller has requested a stop."""
    if event is not None and event.is_set():
        raise InterruptedError("Run stopped by user.")


def run_target(
    researcher_name: str,
    institution: str | None = None,
    orcid: str | None = None,
    output_dir: str | Path = Path("output"),
    from_year: int = 2014,
    max_works: int = 400,

    pool_size: int | None = None,
    faculty_page_url: str | None = None,
    cv_path: str | Path | None = None,
    rerank_top_n: int = 60,
    list_size: int | None = None,
    star_map_topics: int = 5,
    include_citation_neighbors: bool = True,
    cv_data=None,
    stop_event: threading.Event | None = None,
) -> dict:
    """
    Run the full researcher-mapping pipeline.

    Returns a dict with keys:
      target, israel_top20, world_top20, graph_path
    """
    output_dir = Path(output_dir)
    weights = load_weights()
    bucket_caps = load_bucket_caps()
    policy = load_policy()
    if list_size is not None:
        policy["target_list_size"] = max(1, min(int(list_size), 100))

    oa = OpenAlexClient()

    # ── Step 1: identity resolution ───────────────────────────────────────────
    log.info("Resolving identity for '%s' …", researcher_name)
    resolved = resolve_target_identity(researcher_name, institution, orcid)
    if not resolved.chosen:
        raise ValueError(
            f"Could not resolve '{researcher_name}' in OpenAlex. "
            "Try adding an institution hint or ORCID."
        )
    log.info(
        "Resolved to OpenAlex ID %s (confidence %.2f)",
        resolved.openalex_id,
        resolved.confidence,
    )

    _check_stop(stop_event)
    # ── Step 2: ingest target publications ────────────────────────────────────
    log.info("Ingesting publications …")
    target_profile = ingest_openalex_profile(
        resolved.openalex_id,
        from_year=from_year,
        max_works=max_works,
    )
    target_profile.semantic_scholar_author_id = resolved.s2_author_id
    if resolved.orcid:
        target_profile.orcid = resolved.orcid

    # ── Step 3: enrich from faculty page and/or CV ────────────────────────────
    if faculty_page_url:
        from researcher_mapper.parsers.faculty_page_parser import (
            extract_department_from_text,
            fetch_page_text,
        )
        try:
            page_text = fetch_page_text(faculty_page_url)
            dept = extract_department_from_text(page_text)
            if dept:
                target_profile.current_department = dept
        except Exception as exc:
            log.warning("Could not fetch faculty page: %s", exc)

    if cv_data is not None:
        try:
            _apply_cv_data(target_profile, cv_data)
            log.info(
                "CV data applied — name: %s | dept: %s | rank: %s | interests: %d | DOIs: %d",
                cv_data.name, cv_data.department, cv_data.position_rank,
                len(cv_data.research_interests), len(cv_data.publication_dois),
            )
        except Exception as exc:
            log.warning("Could not apply CV data: %s", exc)
    elif cv_path:
        from researcher_mapper.parsers.cv_parser import parse_cv
        try:
            cv = parse_cv(cv_path)
            _apply_cv_data(target_profile, cv)
            log.info(
                "CV parsed — name: %s | dept: %s | rank: %s | interests: %d | DOIs: %d",
                cv.name, cv.department, cv.position_rank,
                len(cv.research_interests), len(cv.publication_dois),
            )
        except Exception as exc:
            log.warning("Could not parse CV (%s): %s", cv_path, exc)

    log.info(
        "Target: %s | Works ingested: %d | Institution: %s",
        target_profile.canonical_name,
        len(target_profile.publications),
        target_profile.current_institution.name if target_profile.current_institution else "unknown",
    )

    _check_stop(stop_event)
    # ── Step 4: build target feature vector ───────────────────────────────────
    target_fv = _build_target_feature_vector(target_profile)
    target_id = (target_profile.openalex_author_id or "").split("/")[-1]

    # ── Step 5: generate candidate pool ───────────────────────────────────────
    log.info("Generating candidate pool (global + Israel) …")
    israel_code = policy.get("israel_country_code", "IL")

    raw_pool = retrieve_candidate_pool(
        target_profile,
        oa=oa,
        country_code=israel_code,
        per_concept=50,
        max_pool_size=pool_size,
        include_citation_neighbors=include_citation_neighbors,
    )
    log.info("Raw pool size before summary fetch: %d", len(raw_pool))

    # Track which pool members cited the target — used to boost familiarity_proxy
    # Use both the _strategy field and the _is_citing_author flag (set by
    # candidate_generation before dedup) so that citing-author provenance is
    # preserved even when a candidate was also found by an earlier strategy.
    citing_author_ids: set[str] = {
        c.get("id", "").split("/")[-1]
        for c in raw_pool
        if (c.get("_strategy") == "citing_author" or c.get("_is_citing_author")) and c.get("id")
    }
    log.info("Citing-author candidates in pool: %d", len(citing_author_ids))

    # ── Step 5b: inject CV-named advisors / postdoc-hosts into pool ──────────
    # Advisors extracted from the target's CV are resolved to OpenAlex author
    # IDs here, before the summary fetch, so they participate in scoring and
    # can be matched by `_match_cv_relationship` later.  We use a conservative
    # per_page=3 and take only the top hit to avoid pulling in wrong people.
    if target_profile.known_cv_advisors:
        existing_ids: set[str] = {
            c.get("id", "").split("/")[-1] for c in raw_pool if c.get("id")
        }
        for advisor_name in target_profile.known_cv_advisors:
            try:
                hits = oa.search_authors(advisor_name, per_page=3)
                if hits:
                    aid = hits[0].get("id", "")
                    aid_short = aid.split("/")[-1] if aid else ""
                    if aid_short and aid_short not in existing_ids:
                        raw_pool.append({"id": aid, "_strategy": "cv_advisor"})
                        existing_ids.add(aid_short)
                        log.info(
                            "Injected CV advisor into pool: %s → %s",
                            advisor_name, aid,
                        )
                    else:
                        log.debug(
                            "CV advisor already in pool: %s → %s", advisor_name, aid
                        )
            except Exception as exc:
                log.warning(
                    "Could not resolve CV advisor '%s': %s", advisor_name, exc
                )

    _check_stop(stop_event)
    # ── Step 6: fetch full summaries for the pool ─────────────────────────────
    pool_ids = [c["id"] for c in raw_pool if c.get("id")]
    # Some dicts in raw_pool are already full author records; the rest have
    # only `id` and `_strategy`.  We fetch summaries for all of them to
    # ensure consistent data.
    candidate_summaries: list[CandidateSummary] = fetch_candidate_summaries(
        pool_ids, oa_client=oa
    )
    il_in_summaries = sum(1 for s in candidate_summaries if s.israel_affiliated)
    log.info(
        "Fetched %d candidate summaries (%d Israel-affiliated)",
        len(candidate_summaries), il_in_summaries,
    )
    # Build id→summary map for the two-stage rerank and reason generation.
    summaries_by_id: dict[str, CandidateSummary] = {
        s.openalex_id: s for s in candidate_summaries
    }

    # ── Diagnostic: trace calibration names through the pipeline ──────────
    # Watched names come from SPECS §9 calibration examples. Logs where they
    # appear (or fail to appear) so missing ones can be debugged without
    # re-running from scratch.
    _watch = {n.lower() for n in policy.get("watch_names", ["Nadav Cohen"])}
    if _watch:
        for s in candidate_summaries:
            if s.name and s.name.lower() in _watch:
                strategies = [
                    c.get("_strategy", "?")
                    for c in raw_pool
                    if c.get("id") == s.openalex_id
                ]
                log.info(
                    "WATCH: '%s' present in summaries — id=%s, "
                    "strategies=%s, works=%d, il_affiliated=%s, "
                    "inst=%s, first_pub=%s, last_pub=%s",
                    s.name, s.openalex_id, strategies, s.works_count,
                    s.israel_affiliated,
                    getattr(s.current_institution, "name", None),
                    s.first_pub_year, s.last_pub_year,
                )
        present_names = {s.name.lower() for s in candidate_summaries if s.name}
        missing = [n for n in _watch if n not in present_names]
        if missing:
            log.warning(
                "WATCH: names NOT present in candidate summaries: %s", missing
            )

    # ── Step 6b: cross-validate against ORCID + Semantic Scholar ────────────
    # ORCID: validate institution (self-reported employment) and detect merged
    # records (works count divergence).  Candidates without ORCID get a lighter
    # S2 paper-count check for merge detection only.
    from researcher_mapper.validation.crosscheck import crosscheck_candidates
    crosscheck_candidates(candidate_summaries)
    log.info("Cross-validation complete")

    _check_stop(stop_event)
    # ── Step 7: compute scores for each candidate ─────────────────────────────
    target_institution_id = (
        target_profile.current_institution.openalex_id or ""
        if target_profile.current_institution
        else ""
    )
    log.info(
        "Target institution: %s (id=%s)",
        target_profile.current_institution.name if target_profile.current_institution else "unknown",
        target_institution_id or "MISSING — dept_mentors will be empty",
    )
    target_department = target_profile.current_department or ""
    target_recent_coauthors = get_recent_coauthor_ids(
        target_profile.publications,
        target_id,
        recent_years=policy.get("recent_coauthor_years", 5),
    )
    # All coauthors across the full publication history (used to exclude from output)
    all_coauthor_ids: set[str] = {
        aid.split("/")[-1]
        for pub in target_profile.publications
        for aid in pub.author_ids
        if aid and aid.split("/")[-1] not in ("", target_id)
    }

    # Normalised advisor name set for CV-relationship matching in the scoring loop
    _advisor_name_set: set[str] = {
        n.lower().strip() for n in target_profile.known_cv_advisors
    }
    if _advisor_name_set:
        log.info("CV advisors/hosts to match: %s", list(target_profile.known_cv_advisors))

    all_candidate_scores: list[CandidateScore] = []

    for summary in candidate_summaries:
        cand_fv = _build_candidate_feature_vector(summary)

        cand_id = summary.openalex_id
        cand_id_short = cand_id.split("/")[-1]

        candidate_cites_target = cand_id_short in citing_author_ids
        base_sim = compute_base_similarity(
            target_fv, cand_fv,
            weights=weights.get("base_similarity"),
            candidate_cites_target=candidate_cites_target,
        )
        same_area = compute_same_area_score(
            target_fv, cand_fv,
            candidate_cites_target=candidate_cites_target,
        )
        comp_topic = complementary_topic_score(
            target_fv.concept_vector, cand_fv.concept_vector
        )

        seniority = estimate_seniority(
            summary.first_pub_year,
            cited_by_count=summary.cited_by_count,
            works_count=summary.works_count,
        )
        reputation = estimate_reputation(summary.cited_by_count, summary.works_count)
        recent_coauthor = cand_id in target_recent_coauthors or cand_id_short in target_recent_coauthors
        any_coauthor = cand_id_short in all_coauthor_ids
        no_conflict = 0.0 if recent_coauthor else 1.0

        familiarity = compute_familiarity_proxy(
            target_coauthor_vector=target_fv.coauthor_vector,
            candidate_openalex_id=cand_id,
            target_ref_vector=target_fv.citation_vector,
            candidate_work_ids=set(),  # full work list not fetched at this stage
            candidate_cites_target=candidate_cites_target,
        )

        # CV-derived relationship boost: advisor/postdoc-host gets a strong
        # familiarity boost because the connection is explicitly documented.
        cv_rel = _match_cv_relationship(summary.name, _advisor_name_set) if _advisor_name_set else None
        if cv_rel:
            familiarity = min(familiarity + 0.3, 1.0)

        # Venue and institution proximity proxies
        venue_sim = _weighted_jaccard_quick(target_fv.venue_vector, cand_fv.venue_vector)
        cand_inst_id = (
            summary.current_institution.openalex_id or ""
            if summary.current_institution
            else ""
        )
        same_institution = bool(
            target_institution_id
            and cand_inst_id
            and target_institution_id.split("/")[-1] == cand_inst_id.split("/")[-1]
        )
        institution_distance_bonus = 0.3 if (not same_institution and same_area >= 0.40) else 0.0

        # Department
        cand_dept = summary.current_institution.department or "" if summary.current_institution else ""
        same_department = bool(
            target_department
            and cand_dept
            and _dept_match(target_department, cand_dept)
        )
        dept_explicitly_different = _dept_explicitly_different(
            target_department, cand_dept, same_department
        )
        department_match = 1.0 if same_department else 0.0

        is_academic = _is_academic_institution(summary.current_institution)
        is_non_research = _is_non_research_institution(summary.current_institution)

        recency_overlap = 1.0 if recent_coauthor else same_area * 0.5

        bucket_score_dict = compute_all_bucket_scores(
            same_area=same_area,
            complementary_topic=comp_topic,
            seniority=seniority,
            reputation=reputation,
            no_conflict=no_conflict,
            familiarity_proxy=familiarity,
            collaboration=_weighted_jaccard_quick(
                target_fv.coauthor_vector, cand_fv.coauthor_vector
            ),
            venue=venue_sim,
            institution_distance_bonus=institution_distance_bonus,
            recency_overlap=recency_overlap,
            department_match=department_match,
            institutional_role=seniority,
            weights=weights,
        )

        # Store the candidate feature vector on the summary for later use
        # (reason generation needs it for shared-topic extraction).
        summary.feature_vector = cand_fv

        score = CandidateScore(
            candidate_id=cand_id,
            name=summary.name,
            institution=(
                summary.current_institution.name if summary.current_institution else None
            ),
            country_code=(
                summary.current_institution.country_code
                if summary.current_institution
                else None
            ),
            department=cand_dept or None,
            israel_affiliated=summary.israel_affiliated,
            last_pub_year=summary.last_pub_year,
            works_count=summary.works_count,
            base_similarity=base_sim,
            same_area_score=same_area,
            complementary_topic_score=comp_topic,
            seniority_score=seniority,
            reputation_score=reputation,
            familiarity_proxy=familiarity,
            no_conflict_score=no_conflict,
            same_department=same_department,
            same_institution=same_institution,
            candidate_cites_target=candidate_cites_target,
            recent_coauthor=recent_coauthor,
            any_coauthor=any_coauthor,
            dept_explicitly_different=dept_explicitly_different,
            is_academic_institution=is_academic,
            is_non_research_institution=is_non_research,
            cv_relationship=cv_rel,
            bucket_scores=BucketScores(**bucket_score_dict),
        )
        all_candidate_scores.append(score)

    # Sort by base similarity for overflow assignment
    all_candidate_scores.sort(key=_score_sort_key, reverse=True)

    _check_stop(stop_event)
    # ── Step 7b: two-stage rerank with full publication profiles ──────────────
    if rerank_top_n > 0:
        _rerank_with_full_profiles(
            scores=all_candidate_scores,
            summaries_by_id=summaries_by_id,
            target_fv=target_fv,
            citing_author_ids=citing_author_ids,
            target_recent_coauthors=target_recent_coauthors,
            target_department=target_department,
            target_institution_id=target_institution_id,
            weights=weights,
            from_year=from_year,
            rerank_top_n=rerank_top_n,
            stop_event=stop_event,
        )
        all_candidate_scores.sort(key=_score_sort_key, reverse=True)

    _check_stop(stop_event)
    # ── Step 8: assign final Israel + World lists ─────────────────────────────
    log.info("Assigning final lists …")
    # Inject per-target values that gate relative mentor quality.
    # area_mentors must have reputation_score >= the target's own reputation
    # so that a mentor is always more established than the person they mentor.
    target_reputation = estimate_reputation(
        target_profile.cited_by_count, target_profile.works_count
    )
    policy["target_reputation_score"] = target_reputation
    log.info("Target reputation score: %.4f (used as area_mentor floor)", target_reputation)

    israel_list, world_list = assign_israel_and_world(
        all_candidate_scores, bucket_caps=bucket_caps, policy=policy
    )

    # ── Diagnostic: trace calibration names through assignment ────────────
    if _watch:
        assigned_names = {
            s.name.lower() for s in (*israel_list, *world_list) if s.name
        }
        for sc in all_candidate_scores:
            if sc.name and sc.name.lower() in _watch:
                status = (
                    "ASSIGNED" if sc.name.lower() in assigned_names
                    else "REJECTED_OR_BELOW_CUTOFF"
                )
                bs = sc.bucket_scores
                log.info(
                    "WATCH %s: '%s' same_area=%.3f seniority=%.3f "
                    "reputation=%.3f works=%d bucket=%s coauthor=%s "
                    "academic=%s inactive=%s cites_target=%s | "
                    "in_area=%.3f inter=%.3f letter=%.3f dept=%.3f "
                    "area=%.3f",
                    status, sc.name, sc.same_area_score, sc.seniority_score,
                    sc.reputation_score, sc.works_count,
                    sc.assigned_bucket, sc.any_coauthor,
                    sc.is_academic_institution,
                    sc.last_pub_year is not None and (
                        datetime.utcnow().year - sc.last_pub_year
                        > policy.get("max_inactivity_years", 10)
                    ),
                    getattr(sc, "candidate_cites_target", None),
                    bs.in_area_collaborators, bs.interdisciplinary_collaborators,
                    bs.recommendation_letter_writers, bs.dept_mentors,
                    bs.area_mentors,
                )

    # Populate reasons — include top shared topic IDs when available
    for lst in (israel_list, world_list):
        for score in lst:
            summary = summaries_by_id.get(score.candidate_id)
            cand_fv = summary.feature_vector if summary else None
            shared_ids = (
                _top_shared_topic_ids(
                    target_fv.topic_vector,
                    cand_fv.topic_vector,
                )
                if cand_fv and cand_fv.topic_vector and target_fv.topic_vector
                else []
            )
            score.reasons = _generate_reasons(
                score, score.name, score.assigned_bucket or "",
                shared_topic_ids=shared_ids,
            )

    log.info(
        "Israel list: %d | World list: %d", len(israel_list), len(world_list)
    )

    # ── Step 9: build graph ────────────────────────────────────────────────────
    G = build_researcher_graph(target_profile, all_candidate_scores, israel_list, world_list)

    # ── Step 10: export ────────────────────────────────────────────────────────
    slug = _name_slug(researcher_name)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_dir = output_dir / f"{slug}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = render_results_json(
        target_profile, israel_list, world_list, resolved.confidence
    )
    save_json(result, run_dir / "results.json")
    selected_list_size = int(policy.get("target_list_size", 20))
    save_csv(israel_list, run_dir / f"israel_top{selected_list_size}.csv")
    save_csv(world_list, run_dir / f"world_top{selected_list_size}.csv")
    graph_path = run_dir / "graph.json"
    save_graph_json(G, graph_path)

    # ── Step 11: interactive star-map visualisation ────────────────────────
    star_map_path = run_dir / "star_map.html"
    try:
        from researcher_mapper.visualization.star_map import generate_star_map
        generate_star_map(
            target_profile, israel_list, world_list,
            summaries_by_id, star_map_path,
            target_fv=target_fv,
            n_topic_axes=star_map_topics,
        )
        result["star_map_path"] = str(star_map_path)
    except Exception as exc:
        log.warning("Could not generate star map: %s", exc)

    log.info("Results written to %s", run_dir)
    result["graph_path"] = str(graph_path)
    return result


# ── Small helpers ──────────────────────────────────────────────────────────────

def _apply_cv_data(profile: ResearcherProfile, cv) -> None:
    """
    Merge CVData fields into a ResearcherProfile.
    The pipeline has already fetched the OpenAlex record; CV data fills gaps
    and can correct or refine what the public graph provides.

    Priority: existing profile values take precedence *unless* they are None/empty.
    Exception: ORCID — if the CV has one and the profile doesn't, always apply.
    """
    from researcher_mapper.models.schemas import CVData
    assert isinstance(cv, CVData)

    # ORCID: CV wins if profile has none
    if cv.orcid and not profile.orcid:
        profile.orcid = cv.orcid

    # Department: CV always wins — it is the authoritative source and overrides
    # the department inferred from work-level authorships in profile_ingest.
    if cv.department:
        profile.current_department = cv.department

    # Rank: CV wins if profile has none
    if cv.position_rank and not profile.estimated_rank:
        profile.estimated_rank = cv.position_rank

    # Career-start year: take the earlier of the two
    if cv.career_start_year:
        if profile.first_pub_year is None:
            profile.first_pub_year = cv.career_start_year
        else:
            profile.first_pub_year = min(profile.first_pub_year, cv.career_start_year)

    # Research interests: always append (used for candidate search topic boost)
    if cv.research_interests:
        profile.cv_research_interests = sorted(
            set(profile.cv_research_interests) | set(cv.research_interests)
        )

    # Known advisors / postdoc hosts: merge into profile
    if cv.known_advisors:
        profile.known_cv_advisors = sorted(
            set(profile.known_cv_advisors) | set(cv.known_advisors)
        )

    # Institution: fill if missing from OpenAlex
    if cv.institution and profile.current_institution is None:
        from researcher_mapper.models.schemas import InstitutionRef
        profile.current_institution = InstitutionRef(name=cv.institution)


def _weighted_jaccard_quick(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


def _dept_match(a: str, b: str) -> bool:
    from researcher_mapper.settings import load_department_aliases
    aliases = load_department_aliases()
    a_canon = _resolve_dept(a.lower(), aliases)
    b_canon = _resolve_dept(b.lower(), aliases)
    return bool(a_canon and b_canon and a_canon == b_canon)


def _dept_explicitly_different(
    target_dept: str, cand_dept: str, same_department: bool
) -> bool:
    """
    Return True when both target and candidate departments are *known* and differ.

    A candidate with unknown department (cand_dept == "") is NOT treated as
    explicitly different — the ambiguity is resolved by the same-area floor
    threshold in policy.yaml (min_same_area_for_dept_mentor) rather than a
    hard exclusion, so strong-topic-overlap same-institution researchers still
    qualify even when dept info is absent from OpenAlex.
    """
    return bool(target_dept and cand_dept and not same_department)


def _resolve_dept(raw: str, aliases: dict[str, list[str]]) -> str | None:
    for canonical, variants in aliases.items():
        if canonical in raw:
            return canonical
        for v in variants:
            if v in raw:
                return canonical
    return None


def _name_slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


# Institution types that OpenAlex assigns to companies / industry organisations.
_INDUSTRY_INSTITUTION_TYPES = {"company"}

# Institution types where we require a name-pattern check before treating the
# institution as academic.  "education", "government", "facility", and unknown
# (None) always pass.  "nonprofit" and "healthcare" only pass if their name does
# NOT match the patterns below.
_INSTITUTION_TYPES_NEEDING_NAME_CHECK = {"nonprofit", "healthcare", "archive", "other"}

# Names that clearly indicate a disease-advocacy / honour-society / non-research
# organisation masquerading as an institution in OpenAlex.  All patterns are
# matched case-insensitively against the institution display name.
# Kept deliberately narrow to avoid false positives on medical schools and
# research hospitals (which have "education" or "facility" type anyway).
import re as _re
_NON_RESEARCH_INST_RE = _re.compile(
    r"\b(?:"
    r"honor\s+society"                          # Alpha Omega Alpha Medical Honor Society
    r"|alzheimer"                               # Alzheimer's Association (of Israel / US)
    r"|cancer\s+(?:society|foundation|association|alliance)"
    r"|diabetes\s+(?:society|foundation|association)"
    r"|heart\s+(?:society|foundation|association)"
    r"|(?:mental|behavioral)\s+health\s+(?:association|foundation|alliance)"
    r"|autism\s+(?:society|speaks|foundation)"
    r"|disease\s+(?:society|foundation|association)"
    r"|patient\s+advocacy"
    r"|humanitarian\s+(?:fund|foundation|aid)"
    r")\b",
    _re.I,
)


def _is_non_research_institution(inst) -> bool:
    """
    Return True ONLY when the institution name matches the Tier-1 non-research
    name pattern (disease-advocacy organisations, honour societies, etc.).

    Deliberately does NOT fire for plain company/industry institutions — a
    researcher at Meta AI Research or Google Brain is a valid collaborator.
    This flag is used to block non-research affiliates from ALL buckets,
    whereas is_academic_institution only blocks from mentor/letter-writer roles.

    NOTE: we check the name WITHOUT filtering on institution_type because the
    company-override in candidate_ingest.py mutates institution_type → "company"
    while leaving the display name unchanged.  A researcher whose last_known
    institution is "Alpha Omega Alpha Medical Honor Society" but whose type was
    overridden to "company" by an old Meta affiliation must still be caught here.
    The patterns in _NON_RESEARCH_INST_RE are specific enough to avoid false
    positives against real research institutions.
    """
    if inst is None:
        return False
    name = (getattr(inst, "name", "") or "").lower()
    return bool(_NON_RESEARCH_INST_RE.search(name))


def _is_academic_institution(inst) -> bool:
    """
    Return False when OpenAlex signals that the institution is not a research context.

    Rules (in order):
    1. None  →  True  (unknown → assume academic; filtered later by other signals).
    2. type == "company"  →  False.
    3. type in {nonprofit, healthcare, archive, other} AND name matches a
       known non-research pattern  →  False.
       (Catches "Alpha Omega Alpha Medical Honor Society", "Alzheimer's Association
       of Israel", etc. that OpenAlex labels as nonprofit/healthcare but are not
       research institutions.)
    4. Anything else (education, government, facility, unknown type)  →  True.
    """
    if inst is None:
        return True
    inst_type = getattr(inst, "institution_type", None)
    if inst_type in _INDUSTRY_INSTITUTION_TYPES:
        return False
    if inst_type in _INSTITUTION_TYPES_NEEDING_NAME_CHECK:
        name = (getattr(inst, "name", "") or "").lower()
        if _NON_RESEARCH_INST_RE.search(name):
            return False
    return True


# ── Input file loader ─────────────────────────────────────────────────────────

def _parse_txt_input(text: str) -> dict:
    """
    Parse a plain-text input file.  Format: one ``key: value`` pair per line.
    Lines starting with ``#`` and blank lines are ignored.
    ``=`` is also accepted as separator alongside ``:``.
    """
    data: dict = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Split on first : or =
        if ":" in line:
            key, _, val = line.partition(":")
        elif "=" in line:
            key, _, val = line.partition("=")
        else:
            continue
        key = key.strip().lower().replace("-", "_").replace(" ", "_")
        val = val.strip()
        # Convert bare booleans
        if val.lower() in ("true", "yes"):
            val = True
        elif val.lower() in ("false", "no"):
            val = False
        data[key] = val
    return data


def load_input_file(path: str | Path) -> dict:
    """
    Load a YAML or JSON researcher input file and return it as a plain dict.
    Accepted keys match the CLI arguments (underscores or hyphens both work).
    Raises ValueError for unrecognised keys so typos are caught early.
    """
    import yaml  # already a dependency

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(fh) or {}
        elif path.suffix.lower() == ".json":
            data = json.load(fh)
        elif path.suffix.lower() == ".txt":
            data = _parse_txt_input(fh.read())
        else:
            raise ValueError(f"Unsupported file type '{path.suffix}'. Use .txt, .yaml, or .json.")

    # Normalise keys: hyphens → underscores
    data = {k.replace("-", "_"): v for k, v in data.items()}

    _KNOWN_KEYS = {
        "name", "institution", "orcid",
        "faculty_page", "cv",
        "from_year", "max_works", "pool_size",
        "output_dir", "rerank_top_n", "list_size", "target_list_size",
        "star_map_topics",
        "include_citation_neighbors",
    }
    unknown = set(data) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"Unknown key(s) in input file: {sorted(unknown)}")

    # Strip empty strings so they fall back to defaults/None
    return {k: v for k, v in data.items() if v not in (None, "")}


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Map the closest researchers to a given scholar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # From a YAML file:\n"
            "  python -m researcher_mapper.pipelines.run_target --input researcher.yaml\n\n"
            "  # Inline (CLI flags override any matching key in the file):\n"
            "  python -m researcher_mapper.pipelines.run_target \\\n"
            "      --input researcher.yaml --name 'Yann LeCun'\n\n"
            "  # No file:\n"
            "  python -m researcher_mapper.pipelines.run_target \\\n"
            "      --name 'Nir Friedman' --institution 'Hebrew University of Jerusalem'"
        ),
    )

    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        default=None,
        help="YAML or JSON file with researcher details (see examples/researcher.yaml)",
    )
    parser.add_argument("--name", default=None, help="Full name of the target researcher")
    parser.add_argument("--institution", default=None, help="Affiliation hint (improves disambiguation)")
    parser.add_argument("--orcid", default=None, help="ORCID iD of the target")
    parser.add_argument("--output-dir", default=None, help="Directory for output files (default: output)")
    parser.add_argument("--from-year", type=int, default=None, help="Earliest publication year to ingest (default: 2014)")
    parser.add_argument("--max-works", type=int, default=None, help="Max works to ingest per researcher (default: 400)")
    parser.add_argument("--pool-size", type=int, default=None, help="Candidate pool size before ranking (default: 200)")
    parser.add_argument("--list-size", type=int, default=None, help="Requested number of researchers per Israel/World list (default: 20)")
    parser.add_argument(
        "--include-citation-neighbors",
        action="store_true",
        default=None,
        help="Fetch authors from target reference lists (slower)",
    )
    parser.add_argument("--faculty-page", default=None, help="URL of the researcher's university profile page")
    parser.add_argument("--cv", default=None, help="Path to a plain-text CV file")
    parser.add_argument(
        "--rerank-top-n", type=int, default=None,
        help="Candidates to rerank with full publication profiles (0=off, default 60)",
    )
    parser.add_argument(
        "--star-map-topics", type=int, default=None,
        help="Number of topic axes to show in the star map (2-10, default 5)",
    )

    args = parser.parse_args()

    # ── Merge: file values first, then CLI overrides ──────────────────────────
    cfg: dict = {}
    if args.input:
        cfg = load_input_file(args.input)
        log.info("Loaded input file: %s", args.input)

    # CLI flags override file values when explicitly supplied
    if args.name is not None:
        cfg["name"] = args.name
    if args.institution is not None:
        cfg["institution"] = args.institution
    if args.orcid is not None:
        cfg["orcid"] = args.orcid
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.from_year is not None:
        cfg["from_year"] = args.from_year
    if args.max_works is not None:
        cfg["max_works"] = args.max_works
    if args.pool_size is not None:
        cfg["pool_size"] = args.pool_size
    if args.list_size is not None:
        cfg["list_size"] = args.list_size
    if args.include_citation_neighbors is not None:
        cfg["include_citation_neighbors"] = args.include_citation_neighbors

    if args.faculty_page is not None:
        cfg["faculty_page"] = args.faculty_page
    if args.cv is not None:
        cfg["cv"] = args.cv
    if args.rerank_top_n is not None:
        cfg["rerank_top_n"] = args.rerank_top_n
    if args.star_map_topics is not None:
        cfg["star_map_topics"] = args.star_map_topics

    if not cfg.get("name"):
        parser.error("Researcher name is required — set it in the input file or with --name.")

    result = run_target(
        researcher_name=cfg["name"],
        institution=cfg.get("institution"),
        orcid=cfg.get("orcid"),
        output_dir=Path(cfg.get("output_dir", "output")),
        from_year=int(cfg.get("from_year", 2014)),
        max_works=int(cfg.get("max_works", 400)),

        pool_size=int(cfg["pool_size"]) if cfg.get("pool_size") else None,
        faculty_page_url=cfg.get("faculty_page"),
        cv_path=cfg.get("cv"),
        rerank_top_n=int(cfg["rerank_top_n"]) if cfg.get("rerank_top_n") is not None else 60,
        list_size=(
            int(cfg.get("list_size") or cfg.get("target_list_size"))
            if (cfg.get("list_size") or cfg.get("target_list_size")) is not None
            else None
        ),
        star_map_topics=int(cfg["star_map_topics"]) if cfg.get("star_map_topics") is not None else 5,
        include_citation_neighbors=bool(cfg.get("include_citation_neighbors", True)),
    )

    sys.stdout.buffer.write(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    main()
