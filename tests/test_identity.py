"""Tests for identity resolution helpers."""
import pytest
from researcher_mapper.identity.dedupe import dedupe_candidates
from researcher_mapper.identity.resolve_target import _score_openalex_candidate


def test_dedupe_by_openalex_id():
    candidates = [
        {"id": "https://openalex.org/A123", "display_name": "Alice"},
        {"id": "https://openalex.org/A123", "display_name": "Alice (dup)"},
        {"id": "https://openalex.org/A456", "display_name": "Bob"},
    ]
    result = dedupe_candidates(candidates)
    assert len(result) == 2
    assert result[0]["display_name"] == "Alice"


def test_dedupe_by_orcid():
    candidates = [
        {"id": "", "orcid": "https://orcid.org/0000-0001-2345-6789", "display_name": "Alice"},
        {"id": "", "orcid": "https://orcid.org/0000-0001-2345-6789", "display_name": "A. Smith"},
        {"id": "", "orcid": "https://orcid.org/0000-0009-9999-9999", "display_name": "Bob"},
    ]
    result = dedupe_candidates(candidates)
    assert len(result) == 2


def test_dedupe_preserves_order():
    candidates = [
        {"id": "https://openalex.org/A1", "display_name": "First"},
        {"id": "https://openalex.org/A2", "display_name": "Second"},
    ]
    result = dedupe_candidates(candidates)
    assert result[0]["display_name"] == "First"


def test_identity_scoring_uses_nested_affiliation_institution():
    candidate = {
        "display_name": "Daniel Soudry",
        "affiliations": [
            {"institution": {"display_name": "Technion Israel Institute of Technology"}},
        ],
        "last_known_institution": {},
    }

    score = _score_openalex_candidate(
        candidate,
        name="Daniel Soudry",
        institution="Technion",
    )

    assert score > 0.8


def test_resolve_target_identity_requires_matching_orcid(monkeypatch):
    import researcher_mapper.identity.resolve_target as mod

    class FakeOpenAlexClient:
        def search_authors(self, name, per_page=25):
            return [
                {
                    "id": "https://openalex.org/A_WRONG",
                    "display_name": name,
                    "orcid": "https://orcid.org/0000-0000-0000-0001",
                },
                {
                    "id": "https://openalex.org/A_RIGHT",
                    "display_name": "D. Soudry",
                    "orcid": "https://orcid.org/0000-0000-0000-0002",
                },
            ]

    class FakeSemanticScholarClient:
        def search_author(self, name, limit=5):
            return {"data": []}

    monkeypatch.setattr(mod, "OpenAlexClient", FakeOpenAlexClient)
    monkeypatch.setattr(mod, "SemanticScholarClient", FakeSemanticScholarClient)

    result = mod.resolve_target_identity(
        "Daniel Soudry",
        orcid="0000-0000-0000-0002",
    )

    assert result.openalex_id == "A_RIGHT"
    assert result.orcid == "0000-0000-0000-0002"
