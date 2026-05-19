"""Tests for the CV parser (text-mode — no real PDF needed)."""
import textwrap
from pathlib import Path

import pytest

from researcher_mapper.parsers.cv_parser import (
    _extract_name,
    _extract_institution,
    _extract_rank,
    _extract_phd_year,
    _extract_research_interests,
    _extract_publications,
    _extract_advisors,
    _find_orcid,
    _find_email,
    _find_dois,
    split_into_sections,
    parse_cv,
)


_SAMPLE_CV = textwrap.dedent("""\
    Nir Friedman
    Department of Computer Science
    Hebrew University of Jerusalem
    nir.friedman@cs.huji.ac.il | ORCID: 0000-0002-1234-5678

    RESEARCH INTERESTS
    Probabilistic graphical models, Bayesian inference, machine learning, causal reasoning

    EDUCATION
    Ph.D. in Computer Science, Stanford University, 1997
    B.Sc. in Mathematics and CS, Hebrew University, 1991

    POSITIONS
    2005–present  Full Professor, Dept. of Computer Science, Hebrew University of Jerusalem
    2000–2005     Associate Professor, Weizmann Institute of Science

    PUBLICATIONS
    [1] Friedman N., Koller D. (2003). "Being Bayesian about Network Structure".
        Machine Learning. DOI: 10.1023/A:1020249912095
    [2] Friedman N. et al. (1997). "Bayesian Network Classifiers".
        Machine Learning. doi:10.1023/A:1007465528199
""")


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "cv.txt"
    p.write_text(_SAMPLE_CV, encoding="utf-8")
    return p


def test_find_orcid():
    assert _find_orcid(_SAMPLE_CV) == "0000-0002-1234-5678"


def test_find_email():
    assert _find_email(_SAMPLE_CV) == "nir.friedman@cs.huji.ac.il"


def test_extract_name():
    header = "Nir Friedman\nDepartment of Computer Science"
    assert _extract_name(header) == "Nir Friedman"


def test_extract_institution():
    text = "Full Professor, Hebrew University of Jerusalem"
    inst = _extract_institution(text)
    assert inst is not None
    assert "hebrew university" in inst.lower()


def test_extract_rank_full():
    assert _extract_rank("Full Professor, Dept. of CS") == "full"


def test_extract_rank_associate():
    assert _extract_rank("Associate Professor, Weizmann Institute") == "associate"


def test_extract_rank_priority():
    # If both full and associate appear, full should win
    text = "2005 Full Professor\n2000 Associate Professor"
    assert _extract_rank(text) == "full"


def test_extract_phd_year():
    edu = "Ph.D. in Computer Science, Stanford University, 1997"
    assert _extract_phd_year(edu) == 1997


def test_extract_research_interests_comma_list():
    section = "Probabilistic graphical models, Bayesian inference, machine learning"
    interests = _extract_research_interests(section)
    assert len(interests) >= 3
    assert any("bayesian" in i.lower() for i in interests)


def test_extract_research_interests_bullet_lines():
    section = "- Machine learning\n- Computer vision\n- Deep learning"
    interests = _extract_research_interests(section)
    assert len(interests) == 3


def test_extract_publications_dois():
    _, dois = _extract_publications(_SAMPLE_CV)
    assert len(dois) >= 1
    assert any("10.1023" in d for d in dois)


def test_extract_publications_titles():
    titles, _ = _extract_publications(_SAMPLE_CV)
    assert any("bayesian" in t.lower() for t in titles)


def test_split_sections():
    sections = split_into_sections(_SAMPLE_CV)
    assert "education" in sections
    assert "interests" in sections
    assert "publications" in sections


def test_parse_cv_txt(sample_txt: Path):
    cv = parse_cv(sample_txt)
    assert cv.name == "Nir Friedman"
    assert cv.orcid == "0000-0002-1234-5678"
    assert cv.email == "nir.friedman@cs.huji.ac.il"
    assert cv.position_rank == "full"
    assert cv.phd_year == 1997
    assert len(cv.research_interests) >= 3
    assert len(cv.publication_dois) >= 1
    assert cv.department is not None


def test_apply_cv_data_fills_gaps():
    """CVData should fill profile fields that are None without overwriting set ones."""
    from researcher_mapper.models.schemas import CVData, ResearcherProfile
    from researcher_mapper.pipelines.run_target import _apply_cv_data

    profile = ResearcherProfile(
        canonical_name="Nir Friedman",
        orcid=None,
        current_department=None,
        estimated_rank=None,
    )
    cv = CVData(
        orcid="0000-0002-1234-5678",
        department="computer science",
        position_rank="full",
        career_start_year=1991,
        research_interests=["machine learning", "bayesian inference"],
    )
    _apply_cv_data(profile, cv)

    assert profile.orcid == "0000-0002-1234-5678"
    assert profile.current_department == "computer science"
    assert profile.estimated_rank == "full"
    assert profile.first_pub_year == 1991
    assert "machine learning" in profile.cv_research_interests


# ── Numbered-header regression tests (Fix 1) ─────────────────────────────────

_NUMBERED_CV = textwrap.dedent("""\
    Daniel Soudry
    Technion — Israel Institute of Technology
    daniel@ee.technion.ac.il

    1. Education
    Ph.D. in Applied Mathematics, Columbia University, 2013
    Advisor: Liam Paninski
    B.Sc. in Electrical Engineering, Technion, 2008

    2. Appointments
    2014–present  Assistant Professor, Department of Electrical Engineering, Technion
    2013–2014     Postdoctoral Fellow, Technion. Postdoc advisor: Ron Meir

    3. Research Interests
    Machine learning, deep learning theory, optimization, neural networks

    4. Selected Publications
    [1] Soudry D. et al. (2018). "The Implicit Bias of Gradient Descent on Separable Data".
        JMLR. doi:10.1234/jmlr.v19.soudry18
    [2] Soudry D., Carmon Y. (2016). "No bad local minima: Data independent training error
        guarantees for single layer neural networks". arXiv:1605.08361.
""")


def test_numbered_header_detected():
    """'3. Research Interests' must be detected as a header."""
    from researcher_mapper.parsers.cv_parser import _is_header, _classify_header
    assert _is_header("3. Research Interests")
    assert _classify_header("3. Research Interests") == "interests"


def test_numbered_header_all_caps():
    """'4. SELECTED PUBLICATIONS' (all-caps after stripping number) must be a header."""
    from researcher_mapper.parsers.cv_parser import _is_header
    assert _is_header("4. SELECTED PUBLICATIONS")


def test_numbered_cv_sections_split():
    """Numbered sections in a Daniel-style CV should parse to the right labels."""
    sections = split_into_sections(_NUMBERED_CV)
    assert "education" in sections, "education section not found"
    assert "interests" in sections, "interests section not found"
    assert "publications" in sections, "publications section not found"
    # Advisor name must land in the education section body
    assert "Liam Paninski" in sections["education"]


def test_extract_advisors_phd():
    """PhD advisor pattern should be extracted."""
    text = "Ph.D. in Applied Mathematics, Columbia University, 2013\nAdvisor: Liam Paninski"
    advisors = _extract_advisors(text)
    assert any("Paninski" in a for a in advisors), f"Paninski not found in {advisors}"


def test_extract_advisors_postdoc():
    """Postdoc advisor pattern should be extracted."""
    text = "2013–2014  Postdoctoral Fellow, Technion. Postdoc advisor: Ron Meir"
    advisors = _extract_advisors(text)
    assert any("Meir" in a for a in advisors), f"Meir not found in {advisors}"


def test_numbered_cv_full_parse(tmp_path: Path):
    """Full parse of a Daniel-style numbered CV."""
    p = tmp_path / "daniel.txt"
    p.write_text(_NUMBERED_CV, encoding="utf-8")
    cv = parse_cv(p)

    assert cv.phd_year == 2013
    assert len(cv.research_interests) >= 3
    assert any("machine learning" in i.lower() or "deep learning" in i.lower()
               for i in cv.research_interests)
    assert len(cv.publication_dois) >= 1
    assert len(cv.known_advisors) >= 2, f"Expected ≥2 advisors, got: {cv.known_advisors}"
    advisor_text = " ".join(cv.known_advisors).lower()
    assert "paninski" in advisor_text, f"Paninski missing from advisors: {cv.known_advisors}"
    assert "meir" in advisor_text, f"Meir missing from advisors: {cv.known_advisors}"


# ── ACADEMIC-prefixed header tests (Fix 3 — Daniel's real CV format) ─────────

def test_academic_degrees_classifies_as_education():
    """'ACADEMIC DEGREES' (Daniel's real CV header) must map to education."""
    from researcher_mapper.parsers.cv_parser import _is_header, _classify_header
    assert _is_header("ACADEMIC DEGREES"), "'ACADEMIC DEGREES' not detected as header"
    assert _classify_header("ACADEMIC DEGREES") == "education", (
        f"Expected 'education', got '{_classify_header('ACADEMIC DEGREES')}'"
    )


def test_academic_appointments_classifies_as_positions():
    """'ACADEMIC APPOINTMENTS' (Daniel's real CV header) must map to positions."""
    from researcher_mapper.parsers.cv_parser import _is_header, _classify_header
    assert _is_header("ACADEMIC APPOINTMENTS"), "'ACADEMIC APPOINTMENTS' not detected as header"
    assert _classify_header("ACADEMIC APPOINTMENTS") == "positions", (
        f"Expected 'positions', got '{_classify_header('ACADEMIC APPOINTMENTS')}'"
    )


def test_personal_details_classifies_as_contact():
    """'PERSONAL DETAILS' must map to the contact section."""
    from researcher_mapper.parsers.cv_parser import _is_header, _classify_header
    assert _is_header("1. PERSONAL DETAILS")
    assert _classify_header("1. PERSONAL DETAILS") == "contact"


def test_extract_advisors_host_with_honorific():
    """'Host: Prof. Liam Paninski' — standalone Host: label with Prof. honorific."""
    text = "2013–2014  Postdoctoral Fellow, Columbia University. Host: Prof. Liam Paninski"
    advisors = _extract_advisors(text)
    assert any("Paninski" in a for a in advisors), f"Paninski not found in {advisors}"
    # Honorific must be stripped — should not appear as the name
    assert not any(a.startswith("Prof") for a in advisors), f"Honorific not stripped: {advisors}"


def test_extract_advisors_advisor_with_honorific():
    """'Advisor: Prof. Ron Meir' — advisor label with Prof. honorific."""
    text = "Ph.D. Advisor: Prof. Ron Meir, Technion, 2013"
    advisors = _extract_advisors(text)
    assert any("Meir" in a for a in advisors), f"Meir not found in {advisors}"
    assert not any(a.startswith("Prof") for a in advisors), f"Honorific not stripped: {advisors}"


_DANIEL_CV = textwrap.dedent("""\
    Daniel Soudry
    Technion — Israel Institute of Technology
    daniel@ee.technion.ac.il

    ACADEMIC DEGREES
    Ph.D. in Applied Mathematics, Columbia University, 2013
    Advisor: Prof. Liam Paninski
    B.Sc. in Electrical Engineering, Technion, 2008

    ACADEMIC APPOINTMENTS
    2014–present  Assistant Professor, Department of Electrical Engineering, Technion
    2013–2014     Postdoctoral Fellow, Columbia University. Host: Prof. Ron Meir

    RESEARCH INTERESTS
    Machine learning, deep learning theory, optimization, neural networks

    SELECTED PUBLICATIONS
    [1] Soudry D. et al. (2018). "The Implicit Bias of Gradient Descent on Separable Data".
        JMLR. doi:10.1234/jmlr.v19.soudry18
""")


_REAL_DANIEL_STYLE_CV = textwrap.dedent("""\
    Date: 12/4/2026
    R E S U M E
    1. PERSONAL DETAILS
    Full Name: Daniel Soudry
    E-mail: daniel.soudry@technion.ac.il

    2. ACADEMIC DEGREES
    2008-2013 PhD, direct track
    Electrical Engineering, Technion, Israel (Advisor: Prof. Ron Meir)

    3. ACADEMIC APPOINTMENTS
    2021-Present Associate Professor
    Electrical and Computer Engineering, Technion, Israel
    2014-2017 Post-doc
    Statistics, Columbia University, USA (Host: Prof. Liam Paninski)

    4. RESEARCH INTERESTS (briefly)
    Despite the impressive recent progress using Artificial Neural Nets (ANNs), they are still far behind
    the capabilities of biological neural nets in most areas. With the long-term aim of closing this gap,
    my research focuses on theoretically understanding how ANNs learn and operate, and how they can be improved.
""")


def test_daniel_cv_sections_split():
    """ACADEMIC DEGREES/APPOINTMENTS headers must produce correct section labels."""
    sections = split_into_sections(_DANIEL_CV)
    assert "education" in sections, f"education section not found; keys={list(sections)}"
    assert "positions" in sections, f"positions section not found; keys={list(sections)}"
    assert "interests" in sections, f"interests section not found"
    assert "publications" in sections, f"publications section not found"
    assert "Liam Paninski" in sections["education"], "Paninski not in education section"
    assert "Ron Meir" in sections["positions"], "Meir not in positions section"


def test_daniel_cv_full_parse(tmp_path: Path):
    """Full parse of Daniel-style all-caps-header CV with Host: and Prof. honorifics."""
    p = tmp_path / "daniel.txt"
    p.write_text(_DANIEL_CV, encoding="utf-8")
    cv = parse_cv(p)

    assert cv.phd_year == 2013
    assert len(cv.known_advisors) >= 2, f"Expected ≥2 advisors, got: {cv.known_advisors}"
    advisor_text = " ".join(cv.known_advisors).lower()
    assert "paninski" in advisor_text, f"Paninski missing: {cv.known_advisors}"
    assert "meir" in advisor_text, f"Meir missing: {cv.known_advisors}"
    # Department should resolve to electrical engineering, not physics
    assert cv.department == "electrical engineering", (
        f"Expected 'electrical engineering', got '{cv.department}'"
    )


def test_real_daniel_style_name_and_prose_interests_parse(tmp_path: Path):
    """Daniel's PDF-style personal-details header and prose interests should parse cleanly."""
    p = tmp_path / "daniel_real_style.txt"
    p.write_text(_REAL_DANIEL_STYLE_CV, encoding="utf-8")
    cv = parse_cv(p)

    assert cv.name == "Daniel Soudry"
    assert cv.email == "daniel.soudry@technion.ac.il"
    assert cv.position_rank == "associate"
    assert cv.department == "electrical engineering"
    assert any("Paninski" in a for a in cv.known_advisors)
    assert any("Ron Meir" in a for a in cv.known_advisors)
    assert 1 <= len(cv.research_interests) <= 4, cv.research_interests
    assert any("Artificial Neural Nets" in item for item in cv.research_interests)


def test_apply_cv_data_cv_department_always_wins():
    """CV department should overwrite an existing (inferred) department value.
    CV data is authoritative; inferred departments from work-level authorships
    are only fallbacks and should be replaced when a CV provides the real value.
    Non-department fields (orcid, rank) are not overwritten by CV when already set."""
    from researcher_mapper.models.schemas import CVData, ResearcherProfile
    from researcher_mapper.pipelines.run_target import _apply_cv_data

    profile = ResearcherProfile(
        canonical_name="Nir Friedman",
        orcid="0000-0000-0000-0001",
        current_department="mathematics",
        estimated_rank="associate",
    )
    cv = CVData(
        orcid="0000-0002-1234-5678",
        department="computer science",
        position_rank="full",
    )
    _apply_cv_data(profile, cv)

    assert profile.orcid == "0000-0000-0000-0001"         # unchanged (CV orcid not applied when profile already has one)
    assert profile.current_department == "computer science"  # CV wins over inferred department
    assert profile.estimated_rank == "associate"           # unchanged (CV rank not applied when profile already has one)
