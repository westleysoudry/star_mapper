"""Tests for network feature ID handling."""

import pytest

from researcher_mapper.features.network_features import (
    build_coauthor_vector,
    get_recent_coauthor_ids,
    last_author_fraction,
)
from researcher_mapper.models.schemas import Publication


def test_coauthor_helpers_compare_short_and_full_openalex_ids():
    target = "A1"
    coauthor = "https://openalex.org/A2"
    publications = [
        Publication(
            id="W1",
            title="First",
            year=2025,
            author_ids=["https://openalex.org/A1", coauthor],
        ),
        Publication(
            id="W2",
            title="Second",
            year=2025,
            author_ids=[coauthor, "https://openalex.org/A1"],
        ),
    ]

    coauthors = build_coauthor_vector(publications, target)
    recent = get_recent_coauthor_ids(publications, target, recent_years=5)

    assert "https://openalex.org/A1" not in coauthors
    assert coauthors[coauthor] == pytest.approx(1.0)
    assert recent == {coauthor}
    assert last_author_fraction(publications, target) == pytest.approx(0.5)
