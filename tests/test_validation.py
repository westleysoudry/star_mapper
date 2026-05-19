"""Tests for cross-validation mutations."""

from researcher_mapper.models.schemas import CandidateSummary, InstitutionRef
from researcher_mapper.validation.crosscheck import _apply_orcid_result


def test_orcid_institution_replacement_clears_stale_openalex_metadata():
    summary = CandidateSummary(
        openalex_id="https://openalex.org/A1",
        name="Test Researcher",
        current_institution=InstitutionRef(
            name="Wrong Israeli Institution",
            country_code="IL",
            institution_type="education",
            ror="https://ror.org/wrong",
        ),
    )

    _apply_orcid_result(
        summary,
        {
            "current_org": {
                "org_name": "Correct University",
                "ror": "https://ror.org/correct",
            }
        },
    )

    assert summary.current_institution is not None
    assert summary.current_institution.name == "Correct University"
    assert summary.current_institution.ror == "https://ror.org/correct"
    assert summary.current_institution.country_code is None
    assert summary.current_institution.institution_type is None
