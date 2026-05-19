"""Tests for feature vector computation."""
import pytest
from researcher_mapper.features.citation_features import weighted_jaccard, cosine_similarity
from researcher_mapper.features.topic_features import complementary_topic_score
from researcher_mapper.features.career_features import estimate_seniority, estimate_reputation


def test_weighted_jaccard_identical():
    a = {"A": 0.5, "B": 0.5}
    assert weighted_jaccard(a, a) == pytest.approx(1.0)


def test_weighted_jaccard_disjoint():
    a = {"A": 1.0}
    b = {"B": 1.0}
    assert weighted_jaccard(a, b) == pytest.approx(0.0)


def test_weighted_jaccard_partial():
    a = {"A": 0.6, "B": 0.4}
    b = {"A": 0.4, "C": 0.6}
    result = weighted_jaccard(a, b)
    assert 0.0 < result < 1.0


def test_cosine_similarity_identical():
    a = {"X": 1.0, "Y": 2.0}
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = {"X": 1.0}
    b = {"Y": 1.0}
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_complementary_topic_score_range():
    a = {"A": 0.8, "B": 0.2}
    b = {"C": 0.7, "D": 0.3}
    score = complementary_topic_score(a, b)
    assert 0.0 <= score <= 1.0


def test_seniority_junior():
    s = estimate_seniority(first_pub_year=2022, cited_by_count=10, works_count=3)
    assert s < 0.5


def test_seniority_senior():
    s = estimate_seniority(first_pub_year=1995, cited_by_count=15000, works_count=200)
    assert s >= 0.7


def test_reputation_zero_works():
    assert estimate_reputation(cited_by_count=100, works_count=0) == 0.0


def test_reputation_high():
    r = estimate_reputation(cited_by_count=10000, works_count=100)
    assert r > 0.7
