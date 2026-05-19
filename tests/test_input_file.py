"""Tests for researcher input files."""

from researcher_mapper.pipelines.run_target import load_input_file


def test_input_file_accepts_star_map_topics_and_citation_neighbors(tmp_path):
    p = tmp_path / "researcher.txt"
    p.write_text(
        "name: Test Researcher\n"
        "list_size: 30\n"
        "star_map_topics: 7\n"
        "include_citation_neighbors: false\n",
        encoding="utf-8",
    )

    cfg = load_input_file(p)

    assert cfg["star_map_topics"] == "7"
    assert cfg["list_size"] == "30"
    assert cfg["include_citation_neighbors"] is False
