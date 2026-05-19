"""Regression tests for candidate ingest edge cases."""

from researcher_mapper.ingest.candidate_ingest import _is_israel_affiliated


def test_israel_affiliation_excludes_company_name_when_type_missing():
    raw = {
        "last_known_institution": {
            "country_code": "IL",
            "display_name": "Meta AI Research",
            "type": None,
        },
        "affiliations": [
            {
                "institution": {
                    "country_code": "IL",
                    "display_name": "Meta AI Research",
                    "type": None,
                },
                "years": [2024, 2025],
            }
        ],
    }

    assert not _is_israel_affiliated(raw)
