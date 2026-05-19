"""Tests for hard-filter logic in assignment.py."""
import pytest
from researcher_mapper.models.schemas import BucketScores, CandidateScore
from researcher_mapper.ranking.assignment import (
    passes_hard_filters,
    passes_supplemental_in_area_filters,
)

_POLICY = {
    "allow_recent_coauthors_as_letter_writers": False,
    "allow_same_department_letter_writers": False,
    "dept_mentor_requires_same_department": True,
    "dept_mentor_requires_same_institution": True,
    "min_same_area_for_in_area_collaborator": 0.50,
    "min_same_area_for_letter_writer": 0.55,
    "min_same_area_for_area_mentor": 0.55,
    "min_same_area_for_interdisciplinary_min": 0.20,
    "min_same_area_for_interdisciplinary_max": 0.65,
    "min_complementary_topic_for_interdisciplinary": 0.40,
    "min_seniority_for_mentors": 0.60,
    "min_seniority_for_letter_writers": 0.55,
    "exclude_industry_from_mentors": True,
    "exclude_industry_from_letter_writers": True,
}


def _make_candidate(**kwargs) -> CandidateScore:
    defaults = dict(
        candidate_id="A1",
        name="Test",
        same_area_score=0.70,
        seniority_score=0.70,
        complementary_topic_score=0.50,
        same_department=False,
        same_institution=False,
        recent_coauthor=False,
        bucket_scores=BucketScores(),
    )
    defaults.update(kwargs)
    return CandidateScore(**defaults)


def test_dept_mentor_requires_same_dept_and_inst():
    c = _make_candidate(same_department=False, same_institution=False, seniority_score=0.80)
    assert not passes_hard_filters(c, "dept_mentors", _POLICY)

    c2 = _make_candidate(same_department=True, same_institution=True, seniority_score=0.80)
    assert passes_hard_filters(c2, "dept_mentors", _POLICY)


def test_dept_mentor_accepts_same_department_despite_low_topic_overlap():
    policy = dict(_POLICY, min_same_area_for_dept_mentor=0.30)
    c = _make_candidate(
        same_department=True,
        same_institution=True,
        same_area_score=0.05,
        seniority_score=0.80,
    )
    assert passes_hard_filters(c, "dept_mentors", policy)


def test_dept_mentor_accepts_same_institution_cv_advisor():
    policy = dict(
        _POLICY,
        dept_mentor_requires_same_department=False,
        min_same_area_for_dept_mentor=0.30,
    )
    c = _make_candidate(
        same_department=False,
        same_institution=True,
        same_area_score=0.05,
        seniority_score=0.80,
        cv_relationship="advisor",
    )
    assert passes_hard_filters(c, "dept_mentors", policy)


def test_area_mentor_rejects_same_dept():
    c = _make_candidate(same_department=True, seniority_score=0.80, same_area_score=0.70)
    assert not passes_hard_filters(c, "area_mentors", _POLICY)

    c2 = _make_candidate(same_department=False, seniority_score=0.80, same_area_score=0.70)
    assert passes_hard_filters(c2, "area_mentors", _POLICY)


def test_letter_writer_rejects_recent_coauthor():
    c = _make_candidate(recent_coauthor=True, seniority_score=0.70, same_area_score=0.70)
    assert not passes_hard_filters(c, "recommendation_letter_writers", _POLICY)


def test_letter_writer_rejects_same_dept():
    c = _make_candidate(same_department=True, seniority_score=0.70, same_area_score=0.70)
    assert not passes_hard_filters(c, "recommendation_letter_writers", _POLICY)


def test_letter_writer_requires_field_leader_reputation_when_configured():
    policy = dict(_POLICY, min_reputation_for_letter_writers=0.75)
    c_weak_reputation = _make_candidate(
        seniority_score=0.70,
        same_area_score=0.70,
        reputation_score=0.72,
    )
    assert not passes_hard_filters(
        c_weak_reputation, "recommendation_letter_writers", policy
    )

    c_field_leader = _make_candidate(
        seniority_score=0.70,
        same_area_score=0.70,
        reputation_score=0.80,
    )
    assert passes_hard_filters(
        c_field_leader, "recommendation_letter_writers", policy
    )


def test_in_area_collaborator_threshold():
    c_low = _make_candidate(same_area_score=0.30)
    assert not passes_hard_filters(c_low, "in_area_collaborators", _POLICY)

    c_high = _make_candidate(same_area_score=0.60)
    assert passes_hard_filters(c_high, "in_area_collaborators", _POLICY)


def test_in_area_borderline_requires_explicit_bridge():
    policy = dict(
        _POLICY,
        min_same_area_for_in_area_collaborator=0.18,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
    )
    c_unbridged = _make_candidate(same_area_score=0.44, familiarity_proxy=0.0)
    assert not passes_hard_filters(c_unbridged, "in_area_collaborators", policy)

    c_cites_target = _make_candidate(
        same_area_score=0.37,
        familiarity_proxy=0.0,
        candidate_cites_target=True,
    )
    assert passes_hard_filters(c_cites_target, "in_area_collaborators", policy)

    c_familiar = _make_candidate(same_area_score=0.37, familiarity_proxy=0.35)
    assert passes_hard_filters(c_familiar, "in_area_collaborators", policy)


def test_borderline_country_candidate_can_pass_when_senior_and_reputable():
    policy = dict(
        _POLICY,
        min_same_area_for_in_area_collaborator=0.18,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_unbridged_country_in_area_collaborator=0.35,
        min_seniority_for_unbridged_country_in_area_collaborator=0.50,
        min_reputation_for_unbridged_country_in_area_collaborator=0.55,
    )
    c_local_theorist = _make_candidate(
        same_area_score=0.36,
        familiarity_proxy=0.0,
        israel_affiliated=True,
        seniority_score=0.56,
        reputation_score=0.56,
    )
    assert passes_hard_filters(c_local_theorist, "in_area_collaborators", policy)

    c_local_too_weak = _make_candidate(
        same_area_score=0.36,
        familiarity_proxy=0.0,
        israel_affiliated=True,
        seniority_score=0.59,
        reputation_score=0.51,
    )
    assert not passes_hard_filters(c_local_too_weak, "in_area_collaborators", policy)


def test_supplemental_in_area_rejects_remote_nonlocal_match():
    policy = dict(
        _POLICY,
        min_same_area_for_supplemental_in_area=0.25,
        min_seniority_for_supplemental_in_area=0.45,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_senior_supplemental_in_area=0.35,
        min_seniority_for_senior_supplemental_in_area=0.60,
        min_reputation_for_senior_supplemental_in_area=0.58,
    )
    c_remote = _make_candidate(
        same_area_score=0.44,
        seniority_score=0.51,
        reputation_score=0.53,
        familiarity_proxy=0.0,
        candidate_cites_target=False,
    )
    assert not passes_supplemental_in_area_filters(c_remote, policy)


def test_supplemental_in_area_allows_senior_nonlocal_match():
    policy = dict(
        _POLICY,
        min_same_area_for_supplemental_in_area=0.25,
        min_seniority_for_supplemental_in_area=0.45,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_senior_supplemental_in_area=0.35,
        min_seniority_for_senior_supplemental_in_area=0.60,
        min_reputation_for_senior_supplemental_in_area=0.58,
    )
    c_senior = _make_candidate(
        same_area_score=0.44,
        seniority_score=0.70,
        reputation_score=0.72,
        familiarity_proxy=0.0,
        candidate_cites_target=False,
    )
    assert passes_supplemental_in_area_filters(c_senior, policy)


def test_supplemental_in_area_allows_moderate_local_match():
    policy = dict(
        _POLICY,
        min_same_area_for_supplemental_in_area=0.25,
        min_seniority_for_supplemental_in_area=0.45,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_country_moderate_supplemental_in_area=0.40,
        min_seniority_for_country_moderate_supplemental_in_area=0.48,
    )
    c_local = _make_candidate(
        same_area_score=0.41,
        seniority_score=0.50,
        reputation_score=0.45,
        familiarity_proxy=0.0,
        israel_affiliated=True,
    )
    assert passes_supplemental_in_area_filters(c_local, policy)


def test_supplemental_in_area_rejects_broad_local_above_upper_same_area_bound():
    policy = dict(
        _POLICY,
        min_same_area_for_supplemental_in_area=0.25,
        min_seniority_for_supplemental_in_area=0.45,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_country_broad_supplemental_in_area=0.25,
        min_seniority_for_country_broad_supplemental_in_area=0.50,
        max_same_area_for_country_broad_supplemental_in_area=0.35,
        min_reputation_for_country_broad_supplemental_in_area=0.0,
        min_base_similarity_for_country_broad_supplemental_in_area=0.18,
    )
    c_too_far = _make_candidate(
        same_area_score=0.36,
        seniority_score=0.60,
        reputation_score=0.52,
        base_similarity=0.24,
        familiarity_proxy=0.0,
        israel_affiliated=True,
    )
    assert not passes_supplemental_in_area_filters(c_too_far, policy)


def test_supplemental_in_area_allows_reputable_broad_local_match():
    policy = dict(
        _POLICY,
        min_same_area_for_supplemental_in_area=0.25,
        min_seniority_for_supplemental_in_area=0.45,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_country_broad_supplemental_in_area=0.25,
        max_same_area_for_country_broad_supplemental_in_area=0.35,
        min_seniority_for_country_broad_supplemental_in_area=0.50,
        min_reputation_for_country_broad_supplemental_in_area=0.0,
        min_base_similarity_for_country_broad_supplemental_in_area=0.18,
    )
    c_local = _make_candidate(
        same_area_score=0.30,
        seniority_score=0.58,
        reputation_score=0.46,
        base_similarity=0.20,
        familiarity_proxy=0.0,
        israel_affiliated=True,
    )
    assert passes_supplemental_in_area_filters(c_local, policy)


def test_collaborators_can_require_stronger_career_stage():
    policy = dict(_POLICY, min_seniority_for_collaborators=0.45)
    c_studentish = _make_candidate(same_area_score=0.70, seniority_score=0.42)
    assert not passes_hard_filters(c_studentish, "in_area_collaborators", policy)

    c_established = _make_candidate(same_area_score=0.70, seniority_score=0.46)
    assert passes_hard_filters(c_established, "in_area_collaborators", policy)


def test_strong_same_area_allows_junior_collaborator():
    policy = dict(
        _POLICY,
        min_seniority_for_collaborators=0.45,
        strong_same_area_for_junior_collaborator=0.55,
    )
    c_strong_area = _make_candidate(same_area_score=0.57, seniority_score=0.43)
    assert passes_hard_filters(c_strong_area, "in_area_collaborators", policy)

    c_weak_area = _make_candidate(same_area_score=0.28, seniority_score=0.43)
    assert not passes_hard_filters(c_weak_area, "in_area_collaborators", policy)


def test_interdisciplinary_sweet_spot():
    c = _make_candidate(same_area_score=0.40, complementary_topic_score=0.55)
    assert passes_hard_filters(c, "interdisciplinary_collaborators", _POLICY)

    c_too_similar = _make_candidate(same_area_score=0.70, complementary_topic_score=0.55)
    assert not passes_hard_filters(c_too_similar, "interdisciplinary_collaborators", _POLICY)

    c_too_distant = _make_candidate(same_area_score=0.10, complementary_topic_score=0.55)
    assert not passes_hard_filters(c_too_distant, "interdisciplinary_collaborators", _POLICY)


# ── New: dept_explicitly_different ────────────────────────────────────────────

def test_dept_explicitly_different_blocks_dept_mentor():
    """A candidate whose department is known to differ from the target's is
    excluded from dept_mentors regardless of the same_institution flag."""
    c_diff_dept = _make_candidate(
        same_institution=True,
        same_department=False,
        dept_explicitly_different=True,
        seniority_score=0.80,
    )
    # dept_mentor_requires_same_department=True in _POLICY would also block this,
    # but the new dept_explicitly_different check fires first and is independent.
    assert not passes_hard_filters(c_diff_dept, "dept_mentors", _POLICY)


def test_dept_explicitly_different_does_not_block_other_buckets():
    """dept_explicitly_different must not affect buckets other than dept_mentors."""
    c = _make_candidate(
        same_institution=True,
        same_department=False,
        dept_explicitly_different=True,
        same_area_score=0.70,
        seniority_score=0.80,
    )
    assert passes_hard_filters(c, "area_mentors", _POLICY)
    assert passes_hard_filters(c, "in_area_collaborators", _POLICY)


# ── New: is_academic_institution ─────────────────────────────────────────────

def test_industry_candidate_excluded_from_mentor_buckets():
    """Candidates at companies must be blocked from both mentor buckets and
    recommendation_letter_writers when the policy flags are enabled."""
    c_industry = _make_candidate(
        is_academic_institution=False,
        seniority_score=0.80,
        same_area_score=0.70,
        same_institution=False,
        same_department=False,
    )
    assert not passes_hard_filters(c_industry, "dept_mentors", _POLICY)
    assert not passes_hard_filters(c_industry, "area_mentors", _POLICY)
    assert not passes_hard_filters(
        c_industry, "recommendation_letter_writers", _POLICY
    )


def test_industry_candidate_allowed_as_collaborator():
    """Industry researchers may still appear as collaborators."""
    c_inarea = _make_candidate(is_academic_institution=False, same_area_score=0.70)
    assert passes_hard_filters(c_inarea, "in_area_collaborators", _POLICY)

    # same_area_score must be inside the interdisciplinary window [0.20, 0.65)
    c_interdisciplinary = _make_candidate(
        is_academic_institution=False,
        same_area_score=0.40,
        complementary_topic_score=0.55,
    )
    assert passes_hard_filters(
        c_interdisciplinary,
        "interdisciplinary_collaborators",
        dict(_POLICY,
             min_same_area_for_interdisciplinary_min=0.20,
             min_same_area_for_interdisciplinary_max=0.65),
    )


def test_industry_filter_off_allows_industry_mentor():
    """When the policy flags are disabled, industry candidates pass through."""
    policy_no_filter = dict(
        _POLICY,
        exclude_industry_from_mentors=False,
        exclude_industry_from_letter_writers=False,
    )
    c_industry = _make_candidate(
        is_academic_institution=False,
        seniority_score=0.80,
        same_area_score=0.70,
        same_institution=False,
        same_department=False,
    )
    assert passes_hard_filters(c_industry, "area_mentors", policy_no_filter)
