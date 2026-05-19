"""
Constrained assignment of candidates to buckets.

Algorithm:
1. Work through buckets in priority order.
2. For each bucket, filter the pool by hard constraints from policy.
3. Sort remaining eligible candidates by their bucket-specific score.
4. Assign up to `cap` candidates; mark them as used.
5. Repeat until 20 total are assigned or pool is exhausted.

Two separate lists are produced: Israel (country_code == IL) and World (global).
"""
from __future__ import annotations

from datetime import datetime
from math import ceil

from researcher_mapper.models.schemas import CandidateScore

_CURRENT_YEAR = datetime.utcnow().year

BUCKET_PRIORITY = [
    "dept_mentors",
    "area_mentors",
    "recommendation_letter_writers",
    "in_area_collaborators",
    "interdisciplinary_collaborators",
]


def passes_hard_filters(candidate: CandidateScore, bucket: str, policy: dict) -> bool:
    """Return True if the candidate meets the hard constraints for this bucket."""
    same_area = candidate.same_area_score
    seniority = candidate.seniority_score
    same_dept = candidate.same_department
    same_inst = candidate.same_institution
    recent_coauthor = candidate.recent_coauthor

    # ── Global hard gates (applied to every bucket) ───────────────────────────

    # Exclude researchers with no publications in the last N years (filters
    # deceased researchers and long-retired faculty who would be irrelevant).
    max_inactivity = policy.get("max_inactivity_years", 10)
    if (
        candidate.last_pub_year is not None
        and (_CURRENT_YEAR - candidate.last_pub_year) > max_inactivity
    ):
        return False

    # Exclude coauthors from the output lists — unless the specific bucket
    # explicitly permits them (e.g. dept_mentors, where coauthorship signals
    # an existing working relationship rather than a conflict of interest).
    # Exception: CV-matched advisors/postdoc-hosts are always allowed in
    # recommendation_letter_writers even if they are coauthors, because the
    # whole point of extracting CV relationships is to surface them here.
    exclude_coauthors = policy.get("exclude_all_coauthors", True)
    bucket_allows_coauthors = policy.get(f"{bucket}_allows_coauthors", False)
    is_cv_advisor = candidate.cv_relationship in ("advisor", "postdoc_host")
    if exclude_coauthors and not bucket_allows_coauthors and candidate.any_coauthor:
        if not (is_cv_advisor and bucket == "recommendation_letter_writers"):
            return False

    # Exclude students / postdocs (very low seniority)
    if seniority < policy.get("min_seniority_for_all", 0.25):
        return False

    # Exclude researchers with too few publications — guards against PhD students
    # and preprint-only researchers whose citation count from a single viral paper
    # can inflate their seniority score above the min_seniority_for_all threshold.
    # works_count is a more reliable career-stage proxy than citations alone.
    min_works = policy.get("min_works_for_all", 0)
    if min_works and candidate.works_count < min_works:
        return False

    # Exclude non-research institutions (disease-advocacy nonprofits, honour
    # societies, etc.) from ALL buckets.  Unlike the industry gate below, this
    # fires for every bucket including in_area_collaborators and
    # interdisciplinary_collaborators — these organisations are not research
    # institutions and their members are not valid collaborators in this context.
    if candidate.is_non_research_institution:
        return False

    # ── Industry filter for academic roles ────────────────────────────────────
    # Researchers at companies cannot serve as mentors or write recommendation
    # letters in an academic context.
    if not candidate.is_academic_institution:
        if bucket in ("dept_mentors", "area_mentors") and policy.get(
            "exclude_industry_from_mentors", True
        ):
            return False
        if bucket == "recommendation_letter_writers" and policy.get(
            "exclude_industry_from_letter_writers", True
        ):
            return False

    min_seniority = policy.get("min_seniority_for_mentors", 0.60)
    min_letter_seniority = policy.get("min_seniority_for_letter_writers", 0.55)
    min_collaborator_seniority = policy.get("min_seniority_for_collaborators")

    if bucket == "dept_mentors":
        if policy.get("dept_mentor_requires_same_department", True) and not same_dept:
            return False
        if policy.get("dept_mentor_requires_same_institution", True) and not same_inst:
            return False
        if seniority < min_seniority:
            return False
        if same_inst and is_cv_advisor:
            return True
        if same_dept:
            return True
        # If we know the candidate is in a different department, keep them out
        # of the regular institutional-mentor pass; the same-institution
        # fallback can still use them if the bucket would otherwise be empty.
        if candidate.dept_explicitly_different:
            return False
        # Require at least some same-area overlap — a statistician or linguist in
        # the same department is not a useful mentor for an ML theorist.
        if same_area < policy.get("min_same_area_for_dept_mentor", 0.0):
            return False
        return True

    if bucket == "area_mentors":
        if same_dept:  # area mentor should be outside the immediate department
            return False
        # Mentor must be more established than the target (measured by
        # citations-per-paper reputation proxy, injected per-run via policy).
        target_reputation = policy.get("target_reputation_score", 0.0)
        return (
            same_area >= policy.get("min_same_area_for_area_mentor", 0.55)
            and seniority >= min_seniority
            and candidate.reputation_score >= target_reputation
        )

    if bucket == "recommendation_letter_writers":
        # CV-matched advisors/postdoc-hosts bypass the same_area threshold and
        # the recent-coauthor gate: OpenAlex topic vectors under-represent a PhD
        # advisor's overlap with the student's specific sub-field.
        if is_cv_advisor:
            return (
                seniority >= policy.get("min_seniority_for_all", 0.20)
                and not same_dept
            )
        if same_area < policy.get("min_same_area_for_letter_writer", 0.55):
            return False
        if seniority < min_letter_seniority:
            return False
        # Reputation threshold: letter writers should be well-established in the field.
        # Filters low-profile researchers whose topical overlap is borderline.
        min_rep_letter = policy.get("min_reputation_for_letter_writers", 0.0)
        if min_rep_letter and candidate.reputation_score < min_rep_letter:
            return False
        if not policy.get("allow_recent_coauthors_as_letter_writers", False) and recent_coauthor:
            return False
        if not policy.get("allow_same_department_letter_writers", False) and same_dept:
            return False
        return True

    if bucket == "in_area_collaborators":
        high_confidence_same_area = policy.get("strong_same_area_for_junior_collaborator")
        if (
            min_collaborator_seniority is not None
            and seniority < min_collaborator_seniority
            and not (high_confidence_same_area and same_area >= high_confidence_same_area)
        ):
            return False
        if same_area < policy.get("min_same_area_for_in_area_collaborator", 0.50):
            return False
        # Scores in the borderline band are often caused by broad OpenAlex topic
        # overlap ("AI", "machine learning") rather than genuine same-area fit.
        # Keep them only when there is an explicit bridge: the candidate cited
        # the target, or the familiarity proxy is strong from citations/coauthors.
        unbridged_floor = policy.get("min_same_area_for_unbridged_in_area_collaborator")
        if unbridged_floor and same_area < unbridged_floor:
            min_bridge = policy.get("min_familiarity_for_borderline_in_area", 0.30)
            strong_country_candidate = (
                candidate.israel_affiliated
                and same_area
                >= policy.get("min_same_area_for_unbridged_country_in_area_collaborator", 1.0)
                and seniority
                >= policy.get("min_seniority_for_unbridged_country_in_area_collaborator", 1.0)
                and candidate.reputation_score
                >= policy.get("min_reputation_for_unbridged_country_in_area_collaborator", 1.0)
            )
            if (
                not candidate.candidate_cites_target
                and candidate.familiarity_proxy < min_bridge
                and not strong_country_candidate
            ):
                return False
        return True

    if bucket == "interdisciplinary_collaborators":
        if (
            min_collaborator_seniority is not None
            and seniority < min_collaborator_seniority
        ):
            return False
        return (
            policy.get("min_same_area_for_interdisciplinary_min", 0.20)
            <= same_area
            < policy.get("min_same_area_for_interdisciplinary_max", 0.65)
            and candidate.complementary_topic_score
            >= policy.get("min_complementary_topic_for_interdisciplinary", 0.40)
        )

    return False


def _bucket_score(candidate: CandidateScore, bucket: str) -> float:
    return getattr(candidate.bucket_scores, bucket, 0.0)


def _effective_bucket_caps(
    bucket_caps: dict[str, int],
    policy: dict,
    target_size: int,
) -> dict[str, int]:
    """Scale caps for the requested list size, keeping in-area as the main bucket."""
    caps = {k: int(v) for k, v in bucket_caps.items()}
    if not policy.get("rebalance_bucket_caps", False):
        return caps

    target_size = max(1, int(target_size))
    caps["in_area_collaborators"] = max(
        caps.get("in_area_collaborators", 0),
        target_size,
    )
    caps["dept_mentors"] = min(caps.get("dept_mentors", 0), max(1, ceil(target_size * 0.10)))
    caps["area_mentors"] = min(caps.get("area_mentors", 0), max(1, ceil(target_size * 0.15)))
    caps["recommendation_letter_writers"] = min(
        caps.get("recommendation_letter_writers", 0),
        max(1, ceil(target_size * 0.15)),
    )
    caps["interdisciplinary_collaborators"] = min(
        caps.get("interdisciplinary_collaborators", 0),
        max(1, ceil(target_size * 0.25)),
    )
    return caps


def passes_supplemental_in_area_filters(candidate: CandidateScore, policy: dict) -> bool:
    """
    Return True for in-area candidates that are plausible enough to fill the list.

    The normal in-area filter remains the first choice. This supplemental pass is
    deliberately conservative: it keeps global safety gates, still excludes
    coauthors, and only relaxes the borderline bridge rule for strong senior
    candidates.
    """
    probe_policy = dict(policy)
    probe_policy["min_same_area_for_in_area_collaborator"] = policy.get(
        "min_same_area_for_supplemental_in_area", 0.25
    )
    probe_policy["min_seniority_for_collaborators"] = policy.get(
        "min_seniority_for_supplemental_in_area",
        policy.get("min_seniority_for_collaborators", 0.45),
    )
    probe_policy["min_same_area_for_unbridged_in_area_collaborator"] = 0
    if not passes_hard_filters(candidate, "in_area_collaborators", probe_policy):
        return False

    if passes_hard_filters(candidate, "in_area_collaborators", policy):
        return True

    same_area = candidate.same_area_score
    seniority = candidate.seniority_score
    reputation = candidate.reputation_score
    min_bridge = policy.get("min_familiarity_for_borderline_in_area", 0.30)

    if candidate.candidate_cites_target or candidate.familiarity_proxy >= min_bridge:
        return True

    if (
        candidate.israel_affiliated
        and same_area
        >= policy.get("min_same_area_for_unbridged_country_in_area_collaborator", 1.0)
        and seniority
        >= policy.get("min_seniority_for_unbridged_country_in_area_collaborator", 1.0)
        and reputation
        >= policy.get("min_reputation_for_unbridged_country_in_area_collaborator", 1.0)
    ):
        return True

    if candidate.israel_affiliated:
        if (
            same_area
            >= policy.get("min_same_area_for_country_moderate_supplemental_in_area", 1.0)
            and seniority
            >= policy.get("min_seniority_for_country_moderate_supplemental_in_area", 1.0)
        ):
            return True

        if (
            same_area
            >= policy.get("min_same_area_for_country_broad_supplemental_in_area", 1.0)
            and same_area
            < policy.get("max_same_area_for_country_broad_supplemental_in_area", 1.0)
            and seniority
            >= policy.get("min_seniority_for_country_broad_supplemental_in_area", 1.0)
            and reputation
            >= policy.get("min_reputation_for_country_broad_supplemental_in_area", 1.0)
            and candidate.base_similarity
            >= policy.get("min_base_similarity_for_country_broad_supplemental_in_area", 1.0)
        ):
            return True

    return (
        same_area >= policy.get("min_same_area_for_senior_supplemental_in_area", 1.0)
        and seniority >= policy.get("min_seniority_for_senior_supplemental_in_area", 1.0)
        and reputation >= policy.get("min_reputation_for_senior_supplemental_in_area", 1.0)
    )


def _supplemental_in_area_sort_key(candidate: CandidateScore) -> tuple[float, float, float, float]:
    return (
        _bucket_score(candidate, "in_area_collaborators"),
        candidate.same_area_score,
        candidate.base_similarity,
        candidate.reputation_score,
    )


def _institutional_mentor_sort_key(
    candidate: CandidateScore,
) -> tuple[bool, bool, bool, float, float, float, float]:
    return (
        candidate.cv_relationship in ("advisor", "postdoc_host"),
        candidate.same_department,
        not candidate.dept_explicitly_different,
        _bucket_score(candidate, "dept_mentors"),
        candidate.seniority_score,
        candidate.reputation_score,
        candidate.same_area_score,
    )


def _passes_institutional_mentor_fallback(candidate: CandidateScore, policy: dict) -> bool:
    """Relaxed same-institution fallback used only if no mentor was assigned."""
    if not candidate.same_institution:
        return False

    max_inactivity = policy.get("max_inactivity_years", 10)
    if (
        candidate.last_pub_year is not None
        and (_CURRENT_YEAR - candidate.last_pub_year) > max_inactivity
    ):
        return False

    exclude_coauthors = policy.get("exclude_all_coauthors", True)
    bucket_allows_coauthors = policy.get("dept_mentors_allows_coauthors", False)
    if exclude_coauthors and not bucket_allows_coauthors and candidate.any_coauthor:
        return False

    min_works = policy.get("min_works_for_all", 0)
    if min_works and candidate.works_count < min_works:
        return False

    if candidate.is_non_research_institution:
        return False

    if (
        not candidate.is_academic_institution
        and policy.get("exclude_industry_from_mentors", True)
    ):
        return False

    return candidate.seniority_score >= policy.get("min_seniority_for_mentors", 0.60)


def _find_assigned_index(assigned: list[CandidateScore], candidate_id: str) -> int | None:
    for idx, candidate in enumerate(assigned):
        if candidate.candidate_id == candidate_id:
            return idx
    return None


def _replacement_index_for_institutional_mentor(
    assigned: list[CandidateScore],
) -> int | None:
    replacement_buckets = [
        "interdisciplinary_collaborators",
        "in_area_collaborators",
        "recommendation_letter_writers",
        "area_mentors",
    ]
    for bucket in replacement_buckets:
        for idx in range(len(assigned) - 1, -1, -1):
            if assigned[idx].assigned_bucket == bucket:
                return idx
    return len(assigned) - 1 if assigned else None


def _ensure_institutional_mentor(
    assigned: list[CandidateScore],
    all_candidates: list[CandidateScore],
    bucket_caps: dict[str, int],
    policy: dict,
    target_size: int,
) -> None:
    if not policy.get("ensure_institutional_mentor", True):
        return
    if bucket_caps.get("dept_mentors", 0) <= 0:
        return
    if any(c.assigned_bucket == "dept_mentors" for c in assigned):
        return

    fallback_pool = [
        c for c in all_candidates if _passes_institutional_mentor_fallback(c, policy)
    ]
    if not fallback_pool:
        return

    fallback_pool.sort(key=_institutional_mentor_sort_key, reverse=True)
    chosen = fallback_pool[0]
    chosen.assigned_bucket = "dept_mentors"

    existing_idx = _find_assigned_index(assigned, chosen.candidate_id)
    if existing_idx is not None:
        assigned.pop(existing_idx)
        assigned.insert(0, chosen)
        return

    if len(assigned) < target_size:
        assigned.insert(0, chosen)
        return

    replace_idx = _replacement_index_for_institutional_mentor(assigned)
    if replace_idx is None:
        return
    replaced = assigned.pop(replace_idx)
    replaced.assigned_bucket = None
    assigned.insert(0, chosen)


def assign_final_list(
    candidates: list[CandidateScore],
    bucket_caps: dict[str, int],
    policy: dict,
    target_size: int = 20,
) -> list[CandidateScore]:
    """
    Assign up to `target_size` candidates under bucket caps and hard filters.
    Returns the assigned list with `assigned_bucket` populated.
    """
    bucket_caps = _effective_bucket_caps(bucket_caps, policy, target_size)
    assigned: list[CandidateScore] = []
    used_ids: set[str] = set()

    for bucket in BUCKET_PRIORITY:
        cap = bucket_caps.get(bucket, 0)
        if cap <= 0:
            continue

        eligible = [
            c for c in candidates
            if c.candidate_id not in used_ids
            and passes_hard_filters(c, bucket, policy)
        ]
        if bucket == "recommendation_letter_writers":
            # CV-derived advisors/hosts should surface in this bucket whenever
            # they pass the hard filters; otherwise obvious PhD advisors or
            # postdoc hosts can disappear behind slightly higher generic scores.
            eligible.sort(
                key=lambda c: (
                    c.cv_relationship in ("advisor", "postdoc_host"),
                    _bucket_score(c, bucket),
                ),
                reverse=True,
            )
        elif bucket == "dept_mentors":
            eligible.sort(key=_institutional_mentor_sort_key, reverse=True)
        else:
            eligible.sort(key=lambda c: _bucket_score(c, bucket), reverse=True)

        bucket_added = 0
        for c in eligible[:cap]:
            c.assigned_bucket = bucket
            assigned.append(c)
            used_ids.add(c.candidate_id)
            bucket_added += 1
            if len(assigned) >= target_size:
                break

        if (
            bucket == "in_area_collaborators"
            and policy.get("enable_supplemental_in_area_fill", False)
            and len(assigned) < target_size
            and bucket_added < cap
        ):
            eligible_ids = {c.candidate_id for c in eligible}
            supplemental = [
                c for c in candidates
                if c.candidate_id not in used_ids
                and c.candidate_id not in eligible_ids
                and passes_supplemental_in_area_filters(c, policy)
            ]
            supplemental.sort(key=_supplemental_in_area_sort_key, reverse=True)
            supplemental_slots = min(cap - bucket_added, target_size - len(assigned))
            for c in supplemental[:supplemental_slots]:
                c.assigned_bucket = "in_area_collaborators"
                assigned.append(c)
                used_ids.add(c.candidate_id)
                if len(assigned) >= target_size:
                    break

        if len(assigned) >= target_size:
            break

    _ensure_institutional_mentor(
        assigned,
        candidates,
        bucket_caps,
        policy,
        target_size,
    )

    return assigned


def assign_israel_and_world(
    all_candidates: list[CandidateScore],
    bucket_caps: dict[str, int],
    policy: dict,
) -> tuple[list[CandidateScore], list[CandidateScore]]:
    """
    Produce two independent ranked lists:
    - israel_list: top 20 among candidates with israel_affiliated == True
    - world_list:  top 20 among all candidates (regardless of country)

    The two lists are scored and assigned independently.
    """
    target_size = policy.get("target_list_size", 20)

    def _currently_in_israel(c: CandidateScore) -> bool:
        # The Israel list is for substantive Israeli academic affiliation, not
        # only current physical location. OpenAlex often reports Israeli ML
        # researchers at their current US/EU institution while the affiliation
        # history clearly shows sustained IL academic ties.
        if c.israel_affiliated:
            if not c.is_academic_institution:
                return False
            return True

        # If a candidate is marked IL by current institution but lacks
        # corroborating IL affiliation history, keep them out of the Israel list.
        if c.country_code == "IL":
            # Researchers at company offices in Israel (Meta Israel, Apple Israel,
            # etc.) are NOT treated as "Israeli" for list assignment — their IL
            # country_code comes from employment at a tech-company Israel office,
            # not from genuine academic affiliation.  Put them on the World list.
            if not c.is_academic_institution:
                return False
            # Require that the affiliation-history check also agrees.
            # This double-check catches researchers whose ingest-layer institution
            # was replaced with an old IL affiliation (e.g. a US-based researcher
            # whose most recent academic affiliation happens to be Israeli but
            # who hasn't been active there since before the recency threshold).
            if not c.israel_affiliated:
                return False
            return True
        if c.country_code:          # known non-IL country → world list
            return False
        return c.israel_affiliated  # unknown country → trust affiliation history

    israel_candidates = [c for c in all_candidates if _currently_in_israel(c)]
    world_candidates = [c for c in all_candidates if not _currently_in_israel(c)]

    israel_list = assign_final_list(
        israel_candidates, bucket_caps, policy, target_size=target_size
    )
    world_list = assign_final_list(
        world_candidates, bucket_caps, policy, target_size=target_size
    )

    return israel_list, world_list
