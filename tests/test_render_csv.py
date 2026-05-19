"""Tests for CSV presentation output."""

import csv

from researcher_mapper.models.schemas import CandidateScore
from researcher_mapper.outputs.render_csv import save_csv


def test_save_csv_writes_display_bucket_label(tmp_path):
    score = CandidateScore(
        candidate_id="A1",
        name="Ran El-Yaniv",
        assigned_bucket="dept_mentors",
    )
    path = tmp_path / "israel_top20.csv"

    save_csv([score], path)

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["assigned_bucket"] == "Institutional Mentors"
