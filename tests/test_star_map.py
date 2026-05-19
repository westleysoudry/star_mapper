"""Tests for star-map topic-axis selection."""

from researcher_mapper.visualization import star_map


def test_clamp_topic_axis_count():
    assert star_map._clamp_topic_axis_count(None) == 5
    assert star_map._clamp_topic_axis_count(1) == 2
    assert star_map._clamp_topic_axis_count(12) == 10


def test_neighborhood_axes_include_adjacent_candidate_topics(monkeypatch):
    metadata = {
        "T1": {
            "name": "Neural Networks",
            "field_id": "F1",
            "subfield_id": "S1",
        },
        "T2": {
            "name": "Optimization Algorithms",
            "field_id": "F1",
            "subfield_id": "S2",
        },
        "T3": {
            "name": "Computational Neuroscience",
            "field_id": "F1",
            "subfield_id": "S3",
        },
    }
    monkeypatch.setattr(star_map, "_fetch_topic_metadata", lambda ids: metadata)

    result = star_map._build_neighborhood_openalex_axes(
        target_top={"T1": 0.35, "T2": 0.30},
        cand_top_vecs=[
            {"T1": 0.20, "T3": 0.40},
            {"T2": 0.20, "T3": 0.40},
            {"T3": 0.50},
        ],
        keyword_phrases=["neural networks optimization"],
        n_axes=3,
    )

    topics, names, cand_vecs = result

    assert topics is not None
    assert len(topics) == 3
    assert "T3" in {tid for tid, _ in topics}
    assert names["T3"] == "Computational Neuroscience"
    assert any("T3" in vec for vec in cand_vecs)


def test_reason_topic_ids_render_as_names():
    text = star_map._format_reason_text(
        ["Moderate topic overlap - shared topics: T11612, T10320"],
        {
            "T11612": "Stochastic Gradient Optimization Techniques",
            "T10320": "Neural Network Training",
        },
    )

    assert "T11612" not in text
    assert "T10320" not in text
    assert "Stochastic Gradient Optimization Techniques" in text
    assert "Neural Network Training" in text


def test_candidate_hover_omits_scores():
    hover = star_map._candidate_hover_text(
        "Example Researcher",
        "Example University",
        "In-Area Collaborators",
        "World",
    )

    assert "same-area" not in hover
    assert "seniority" not in hover
    assert "reputation" not in hover
    assert "Example Researcher" in hover
