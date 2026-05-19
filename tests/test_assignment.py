"""Tests for the constrained assignment algorithm."""
import pytest
from researcher_mapper.models.schemas import BucketScores, CandidateScore, InstitutionRef
from researcher_mapper.ranking.assignment import assign_final_list, assign_israel_and_world

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
    "target_list_size": 5,
}

_CAPS = {
    "in_area_collaborators": 2,
    "interdisciplinary_collaborators": 1,
    "recommendation_letter_writers": 1,
    "dept_mentors": 1,
    "area_mentors": 1,
}


def _make(cid: str, **kw) -> CandidateScore:
    defaults = dict(
        candidate_id=cid,
        name=cid,
        base_similarity=0.50,
        same_area_score=0.60,
        seniority_score=0.50,
        complementary_topic_score=0.30,
        same_department=False,
        same_institution=False,
        recent_coauthor=False,
        israel_affiliated=False,
        bucket_scores=BucketScores(
            in_area_collaborators=0.60,
            interdisciplinary_collaborators=0.40,
            recommendation_letter_writers=0.55,
            dept_mentors=0.70,
            area_mentors=0.65,
        ),
    )
    defaults.update(kw)
    return CandidateScore(**defaults)


def test_assign_respects_caps():
    candidates = [
        _make("c1", same_area_score=0.70),
        _make("c2", same_area_score=0.65),
        _make("c3", same_area_score=0.60),
        _make("c4", same_area_score=0.55),
        _make("c5", same_area_score=0.52),
        _make("c6", same_area_score=0.80),
    ]
    result = assign_final_list(candidates, _CAPS, _POLICY, target_size=5)
    # Should not exceed 5 total
    assert len(result) <= 5

    # in_area_collaborators cap is 2
    in_area = [c for c in result if c.assigned_bucket == "in_area_collaborators"]
    assert len(in_area) <= 2


def test_assign_guarantees_same_institution_mentor_when_available():
    institutional = _make(
        "same_inst",
        same_institution=True,
        same_department=False,
        same_area_score=0.05,
        seniority_score=0.85,
        bucket_scores=BucketScores(dept_mentors=0.10),
    )
    candidates = [
        institutional,
        _make("other1", same_area_score=0.80, seniority_score=0.75),
        _make("other2", same_area_score=0.75, seniority_score=0.72),
        _make("other3", same_area_score=0.70, seniority_score=0.70),
    ]
    policy = dict(
        _POLICY,
        target_list_size=2,
        dept_mentor_requires_same_department=False,
        min_same_area_for_dept_mentor=0.30,
    )
    caps = dict(_CAPS, dept_mentors=1, in_area_collaborators=2)

    result = assign_final_list(candidates, caps, policy, target_size=2)

    assert len(result) == 2
    assert result[0].candidate_id == "same_inst"
    assert result[0].assigned_bucket == "dept_mentors"


def test_interdisciplinary_fit_is_not_consumed_by_in_area_first():
    interdisciplinary = _make(
        "interdisciplinary",
        same_area_score=0.35,
        complementary_topic_score=0.70,
        seniority_score=0.75,
        familiarity_proxy=0.40,
        bucket_scores=BucketScores(
            in_area_collaborators=0.95,
            interdisciplinary_collaborators=0.90,
        ),
    )
    candidates = [
        interdisciplinary,
        _make("in_area", same_area_score=0.70, seniority_score=0.75),
    ]
    policy = dict(
        _POLICY,
        target_list_size=2,
        min_same_area_for_in_area_collaborator=0.18,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_interdisciplinary_min=0.20,
        min_same_area_for_interdisciplinary_max=0.45,
        min_complementary_topic_for_interdisciplinary=0.55,
    )
    caps = {
        "dept_mentors": 0,
        "area_mentors": 0,
        "recommendation_letter_writers": 0,
        "interdisciplinary_collaborators": 1,
        "in_area_collaborators": 2,
    }

    result = assign_final_list(candidates, caps, policy, target_size=2)

    assigned = {c.candidate_id: c.assigned_bucket for c in result}
    assert assigned["interdisciplinary"] == "interdisciplinary_collaborators"
    assert assigned["in_area"] == "in_area_collaborators"


def test_interdisciplinary_caption_matches_assigned_bucket():
    from researcher_mapper.pipelines.run_target import _generate_reasons

    candidate = _make(
        "candidate",
        same_area_score=0.35,
        complementary_topic_score=0.70,
    )

    in_area_reasons = _generate_reasons(candidate, "Candidate", "in_area_collaborators")
    interdisciplinary_reasons = _generate_reasons(
        candidate,
        "Candidate",
        "interdisciplinary_collaborators",
    )

    assert not any("good interdisciplinary fit" in r for r in in_area_reasons)
    assert any("good interdisciplinary fit" in r for r in interdisciplinary_reasons)


def test_rebalanced_caps_make_in_area_main_bucket():
    candidates = [
        _make(f"mentor{i}", same_area_score=0.70, seniority_score=0.80)
        for i in range(8)
    ] + [
        _make(
            f"inarea{i}",
            same_area_score=0.62,
            seniority_score=0.55,
            bucket_scores=BucketScores(in_area_collaborators=0.90 - i * 0.01),
        )
        for i in range(12)
    ]
    caps = {
        "in_area_collaborators": 20,
        "interdisciplinary_collaborators": 4,
        "recommendation_letter_writers": 3,
        "dept_mentors": 2,
        "area_mentors": 3,
    }
    policy = dict(
        _POLICY,
        target_list_size=20,
        rebalance_bucket_caps=True,
        dept_mentor_requires_same_department=False,
        dept_mentor_requires_same_institution=False,
        min_same_area_for_in_area_collaborator=0.18,
    )

    result = assign_final_list(candidates, caps, policy, target_size=20)
    in_area_count = sum(c.assigned_bucket == "in_area_collaborators" for c in result)

    assert len(result) == 20
    assert in_area_count >= 11


def test_supplemental_in_area_fill_reaches_requested_size():
    candidates = [
        _make(
            f"supp{i}",
            same_area_score=0.38,
            seniority_score=0.70,
            reputation_score=0.72,
            bucket_scores=BucketScores(in_area_collaborators=0.50 - i * 0.01),
        )
        for i in range(6)
    ]
    caps = dict(_CAPS, in_area_collaborators=6)
    policy = dict(
        _POLICY,
        target_list_size=6,
        enable_supplemental_in_area_fill=True,
        min_same_area_for_in_area_collaborator=0.18,
        min_same_area_for_unbridged_in_area_collaborator=0.45,
        min_familiarity_for_borderline_in_area=0.30,
        min_same_area_for_supplemental_in_area=0.25,
        min_seniority_for_supplemental_in_area=0.45,
        min_same_area_for_senior_supplemental_in_area=0.35,
        min_seniority_for_senior_supplemental_in_area=0.60,
        min_reputation_for_senior_supplemental_in_area=0.58,
    )

    result = assign_final_list(candidates, caps, policy, target_size=6)

    assert len(result) == 6
    assert all(c.assigned_bucket == "in_area_collaborators" for c in result)


def test_assign_no_duplicates():
    candidates = [_make(f"c{i}", same_area_score=0.60 + i * 0.01) for i in range(10)]
    result = assign_final_list(candidates, _CAPS, _POLICY, target_size=5)
    ids = [c.candidate_id for c in result]
    assert len(ids) == len(set(ids))


def test_israel_world_split():
    candidates = [
        _make("il1", israel_affiliated=True, base_similarity=0.90, same_area_score=0.75),
        _make("il2", israel_affiliated=True, base_similarity=0.85, same_area_score=0.70),
        _make("w1", israel_affiliated=False, base_similarity=0.80, same_area_score=0.65),
        _make("w2", israel_affiliated=False, base_similarity=0.75, same_area_score=0.60),
    ]
    policy = dict(_POLICY, target_list_size=2)
    israel_list, world_list = assign_israel_and_world(candidates, _CAPS, policy)

    assert all(c.israel_affiliated for c in israel_list)
    assert len(world_list) <= 2


# ── Non-research institution filter (Tier-1 Semantic Scholar fallback) ────────

def test_is_academic_institution_company_returns_false():
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    inst = InstitutionRef(name="Meta AI Research", institution_type="company")
    assert not _is_academic_institution(inst)


def test_is_academic_institution_none_returns_true():
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    assert _is_academic_institution(None)


def test_is_academic_institution_education_returns_true():
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    # Medical school is education type — must pass
    inst = InstitutionRef(name="Harvard Medical School", institution_type="education")
    assert _is_academic_institution(inst)


def test_is_academic_institution_nonprofit_honor_society_returns_false():
    """Alpha Omega Alpha Medical Honor Society (type=nonprofit) must be excluded."""
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    inst = InstitutionRef(
        name="Alpha Omega Alpha Medical Honor Society",
        institution_type="nonprofit",
    )
    assert not _is_academic_institution(inst)


def test_is_academic_institution_nonprofit_alzheimer_returns_false():
    """Alzheimer's Association of Israel (type=nonprofit or healthcare) must be excluded."""
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    for typ in ("nonprofit", "healthcare"):
        inst = InstitutionRef(name="Alzheimer's Association of Israel", institution_type=typ)
        assert not _is_academic_institution(inst), f"Failed for type={typ}"


def test_is_academic_institution_legitimate_nonprofit_passes():
    """Simons Foundation (nonprofit research funder) must NOT be excluded."""
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    inst = InstitutionRef(name="Simons Foundation", institution_type="nonprofit")
    assert _is_academic_institution(inst)


def test_is_academic_institution_government_lab_passes():
    """Government / facility research labs must pass."""
    from researcher_mapper.pipelines.run_target import _is_academic_institution
    for typ, name in [
        ("government", "National Institutes of Health"),
        ("facility", "Los Alamos National Laboratory"),
    ]:
        inst = InstitutionRef(name=name, institution_type=typ)
        assert _is_academic_institution(inst), f"Incorrectly excluded: {name}"


# ── is_non_research_institution global gate ───────────────────────────────────

def test_is_non_research_institution_plain_company_returns_false():
    """Plain company names are NOT flagged as non-research — they can still collaborate."""
    from researcher_mapper.pipelines.run_target import _is_non_research_institution
    inst = InstitutionRef(name="Meta AI Research", institution_type="company")
    assert not _is_non_research_institution(inst)


def test_is_non_research_institution_honor_society_company_type_returns_true():
    """If company-override fires on a non-research institution, name check still catches it."""
    from researcher_mapper.pipelines.run_target import _is_non_research_institution
    # Simulates Naumov: institution_type overridden to "company" by candidate_ingest,
    # but the display name is still Alpha Omega Alpha Medical Honor Society.
    inst = InstitutionRef(
        name="Alpha Omega Alpha Medical Honor Society",
        institution_type="company",  # overridden by candidate_ingest company-override
    )
    assert _is_non_research_institution(inst)


def test_is_non_research_institution_none_returns_false():
    from researcher_mapper.pipelines.run_target import _is_non_research_institution
    assert not _is_non_research_institution(None)


def test_is_non_research_institution_honor_society_returns_true():
    from researcher_mapper.pipelines.run_target import _is_non_research_institution
    inst = InstitutionRef(
        name="Alpha Omega Alpha Medical Honor Society",
        institution_type="nonprofit",
    )
    assert _is_non_research_institution(inst)


def test_is_non_research_institution_alzheimer_returns_true():
    from researcher_mapper.pipelines.run_target import _is_non_research_institution
    for typ in ("nonprofit", "healthcare"):
        inst = InstitutionRef(name="Alzheimer's Association of Israel", institution_type=typ)
        assert _is_non_research_institution(inst), f"Failed for type={typ}"


def test_is_non_research_institution_simons_foundation_returns_false():
    """Legitimate research nonprofit must NOT be flagged."""
    from researcher_mapper.pipelines.run_target import _is_non_research_institution
    inst = InstitutionRef(name="Simons Foundation", institution_type="nonprofit")
    assert not _is_non_research_institution(inst)


def test_non_research_institution_excluded_from_all_buckets():
    """Candidate at a non-research institution must be excluded from ALL buckets,
    including in_area_collaborators and interdisciplinary_collaborators."""
    from researcher_mapper.ranking.assignment import passes_hard_filters
    policy = dict(_POLICY)
    # High scores so would qualify for every bucket except the gate
    c = _make(
        "nonresearch",
        same_area_score=0.60,
        seniority_score=0.70,
        is_academic_institution=False,
        is_non_research_institution=True,
    )
    for bucket in [
        "in_area_collaborators",
        "interdisciplinary_collaborators",
        "area_mentors",
        "recommendation_letter_writers",
    ]:
        assert not passes_hard_filters(c, bucket, policy), (
            f"Non-research institution should be excluded from {bucket}"
        )


def test_industry_company_allowed_in_collaborator_buckets():
    """Industry company (Meta, Google) can still appear in collaborator buckets."""
    from researcher_mapper.ranking.assignment import passes_hard_filters
    policy = dict(_POLICY)
    c = _make(
        "industry",
        same_area_score=0.60,
        seniority_score=0.70,
        is_academic_institution=False,
        is_non_research_institution=False,  # plain company, not non-research
    )
    # Should pass in_area_collaborators
    assert passes_hard_filters(c, "in_area_collaborators", policy), (
        "Industry collaborator should be allowed in in_area_collaborators"
    )


def test_cv_advisor_prioritized_in_letter_writer_bucket():
    """A CV-derived advisor/host should not be crowded out of letter-writer slots."""
    caps = {
        "in_area_collaborators": 0,
        "interdisciplinary_collaborators": 0,
        "recommendation_letter_writers": 2,
        "dept_mentors": 0,
        "area_mentors": 0,
    }
    policy = dict(_POLICY, target_list_size=2)
    candidates = [
        _make(
            "cv_host",
            same_area_score=0.10,
            seniority_score=0.70,
            any_coauthor=True,
            cv_relationship="advisor",
            bucket_scores=BucketScores(recommendation_letter_writers=0.30),
        ),
        _make(
            "generic_1",
            same_area_score=0.70,
            seniority_score=0.70,
            bucket_scores=BucketScores(recommendation_letter_writers=0.95),
        ),
        _make(
            "generic_2",
            same_area_score=0.68,
            seniority_score=0.70,
            bucket_scores=BucketScores(recommendation_letter_writers=0.90),
        ),
    ]

    result = assign_final_list(candidates, caps, policy, target_size=2)
    result_ids = {c.candidate_id for c in result}

    assert "cv_host" in result_ids
    assert len(result) == 2
    assert "generic_2" not in result_ids
    assert all(c.assigned_bucket == "recommendation_letter_writers" for c in result)


def test_israel_country_code_without_affiliation_history_goes_to_world():
    """A researcher whose country_code is IL but israel_affiliated is False
    (i.e. OpenAlex has the wrong current institution) must land on the world
    list, not the Israel list."""
    candidates = [
        # country_code="IL" from a wrong OpenAlex record, but no IL publication history
        _make(
            "wrong_il",
            country_code="IL",
            israel_affiliated=False,
            is_academic_institution=True,
            base_similarity=0.80,
            same_area_score=0.70,
        ),
        # Genuine Israeli researcher
        _make(
            "real_il",
            country_code="IL",
            israel_affiliated=True,
            is_academic_institution=True,
            base_similarity=0.75,
            same_area_score=0.65,
        ),
    ]
    policy = dict(_POLICY, target_list_size=5)
    israel_list, world_list = assign_israel_and_world(candidates, _CAPS, policy)

    israel_ids = {c.candidate_id for c in israel_list}
    world_ids = {c.candidate_id for c in world_list}
    assert "real_il" in israel_ids
    assert "wrong_il" not in israel_ids
    assert "wrong_il" in world_ids


def test_israel_affiliation_routes_to_israel_even_when_currently_abroad():
    """Substantive Israeli academic affiliation should control Israel routing
    even if OpenAlex reports the current institution outside Israel."""
    candidates = [
        _make(
            "israeli_abroad",
            country_code="US",
            israel_affiliated=True,
            is_academic_institution=True,
            base_similarity=0.80,
            same_area_score=0.70,
        ),
        _make(
            "non_israeli_abroad",
            country_code="US",
            israel_affiliated=False,
            is_academic_institution=True,
            base_similarity=0.75,
            same_area_score=0.65,
        ),
    ]
    policy = dict(_POLICY, target_list_size=5)
    israel_list, world_list = assign_israel_and_world(candidates, _CAPS, policy)

    israel_ids = {c.candidate_id for c in israel_list}
    world_ids = {c.candidate_id for c in world_list}
    assert "israeli_abroad" in israel_ids
    assert "israeli_abroad" not in world_ids
    assert "non_israeli_abroad" in world_ids


def test_company_with_israel_history_stays_off_israel_academic_list():
    candidates = [
        _make(
            "company_il",
            country_code="US",
            israel_affiliated=True,
            is_academic_institution=False,
            base_similarity=0.80,
            same_area_score=0.70,
        ),
    ]
    policy = dict(_POLICY, target_list_size=5)
    israel_list, world_list = assign_israel_and_world(candidates, _CAPS, policy)

    assert not {c.candidate_id for c in israel_list}
    assert "company_il" in {c.candidate_id for c in world_list}
