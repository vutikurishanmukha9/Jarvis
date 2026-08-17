"""
Tests for ATS helper functions: section detection, experience years, education level, seniority, and negation phrases.
"""

from src.modules.career.scorer.services.ats_helpers import (
    calculate_recency_bonus,
    detect_education_level,
    detect_resume_sections,
    detect_seniority_level,
    extract_achievements,
    normalize_text,
)


def test_normalize_text_whitespace_and_casing():
    """Verify normalize_text collapses whitespace and lowers case."""
    raw = "  Senior   Python   DEVELOPER\n\nwith Docker  "
    norm = normalize_text(raw)
    assert norm == "senior python developer with docker"


def test_detect_education_level_phd_and_masters():
    """Verify detection of PhD and Master's degrees."""
    text_phd = "Doctor of Philosophy in Computer Science"
    res_phd = detect_education_level(text_phd)
    assert res_phd["highest_level"] == "phd"
    assert res_phd["level_score"] == 4

    text_ms = "Master of Science in Data Science"
    res_ms = detect_education_level(text_ms)
    assert res_ms["highest_level"] == "masters"
    assert res_ms["level_score"] == 3


def test_detect_education_level_false_positive_filtering():
    """Verify 'Scrum Master' or 'Mastered' does not trigger a Master's degree classification."""
    text_fp = "Certified Scrum Master with expertise in agile sprints."
    res_fp = detect_education_level(text_fp)
    assert res_fp["highest_level"] != "masters" or res_fp["level_score"] == 0


def test_detect_seniority_levels():
    """Verify detection of lead, senior, and junior seniority markers."""
    assert detect_seniority_level("Staff Software Architect")["level"] == "lead"
    assert detect_seniority_level("Senior Backend Engineer")["level"] == "senior"
    assert detect_seniority_level("Junior Developer Intern")["level"] == "entry"


def test_detect_resume_sections():
    """Verify detection of Experience, Education, Skills, and Projects sections."""
    resume = (
        "PROFESSIONAL EXPERIENCE\n- Senior Dev at TechCorp (2020-2024)\n\n"
        "EDUCATION\n- BS in CS\n\n"
        "TECHNICAL SKILLS\n- Python, PyTorch, Docker\n\n"
        "PROJECTS\n- Distributed Cache System\n"
    )
    sections = detect_resume_sections(resume)
    assert "experience" in sections
    assert "education" in sections
    assert "skills" in sections
    assert "projects" in sections


def test_extract_achievements_quantified():
    """Verify extraction of action-verb bullet points with quantified impact."""
    text = (
        "Reduced database query latency by 45% across all microservices.\n"
        "Scaled distributed cluster to 500k requests per second.\n"
        "Attended team meetings every morning.\n"
    )
    achievements = extract_achievements(text)
    assert len(achievements) >= 2
    assert "percentage" in achievements[0]["metrics"]
    assert any("Reduced" in a["text"] for a in achievements)


def test_calculate_recency_bonus():
    """Verify recency bonus is positive for recent/current roles."""
    current_resume = "Senior Engineer (2024 - Present) building AI agents."
    recency = calculate_recency_bonus(current_resume)
    assert recency["bonus"] >= 3
    assert recency["most_recent_year"] == 2024
    assert recency["gap_years"] <= 2


def test_extract_years_from_text_4_digit_years():
    """Verify extract_years_from_text returns complete 4-digit years (e.g. 2020, 1999)."""
    from src.modules.career.scorer.services.ats_helpers import extract_years_from_text

    text = "Worked at Google from 2018 to 2023, previously at Microsoft in 1999."
    years = extract_years_from_text(text)
    assert 2018 in years
    assert 2023 in years
    assert 1999 in years
    assert all(y >= 1900 for y in years)
