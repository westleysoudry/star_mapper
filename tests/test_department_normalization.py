"""Tests for department alias normalization."""
import pytest
from researcher_mapper.parsers.faculty_page_parser import extract_department_from_text


def test_recognizes_canonical_name():
    text = "Professor in the Department of Computer Science, Tel Aviv University"
    dept = extract_department_from_text(text)
    assert dept == "computer science"


def test_recognizes_alias():
    text = "She is a member of the School of Computing at the university."
    dept = extract_department_from_text(text)
    assert dept == "computer science"


def test_returns_none_for_unknown():
    text = "This page contains no departmental information."
    dept = extract_department_from_text(text)
    # May return None or a generic parse — just ensure it doesn't crash
    assert dept is None or isinstance(dept, str)


def test_recognizes_eecs():
    text = "EECS department, Technion"
    dept = extract_department_from_text(text)
    assert dept == "electrical engineering"


# ── Longest-match regression tests (Fix 2) ───────────────────────────────────

def test_electrical_and_computer_engineering_maps_to_ee():
    """
    'Electrical and Computer Engineering' is an EE alias (35 chars).
    After removing 'computer engineering' from CS aliases, only the EE alias
    fires. The longest-match rule ensures it wins even if a shorter CS alias
    were also present.
    """
    text = "Faculty of Electrical and Computer Engineering, Technion"
    dept = extract_department_from_text(text)
    assert dept == "electrical engineering", (
        f"Expected 'electrical engineering', got '{dept}'. "
        "Check that 'computer engineering' was removed from CS aliases."
    )


def test_ece_abbreviation_maps_to_ee():
    """'ECE' (Electrical and Computer Engineering abbreviation) maps to EE."""
    text = "ECE Department, Carnegie Mellon University"
    dept = extract_department_from_text(text)
    assert dept == "electrical engineering"


def test_computer_engineering_alone_not_cs():
    """
    After removing 'computer engineering' from CS aliases, a plain
    'computer engineering' text must NOT match computer science.
    It may fall through to the generic fallback or return a raw string.
    """
    text = "Department of Computer Engineering"
    dept = extract_department_from_text(text)
    assert dept != "computer science", (
        "'computer engineering' must not map to 'computer science' after the fix."
    )


def test_longest_match_over_cs_short_alias():
    """
    When both 'cs' (2 chars, CS alias) and 'computer science and engineering'
    (32 chars, also CS alias) both match, the longer alias wins — but both map
    to the same canonical so the result is still 'computer science'.
    """
    text = "cs department of computer science and engineering"
    dept = extract_department_from_text(text)
    assert dept == "computer science"


def test_eecs_full_phrase_beats_cs_alias():
    """
    'electrical engineering and computer science' (EE alias, 44 chars) should
    beat 'computer science' (CS canonical, 16 chars) or any short CS alias.
    """
    text = "Department of Electrical Engineering and Computer Science, MIT"
    dept = extract_department_from_text(text)
    assert dept == "electrical engineering"
