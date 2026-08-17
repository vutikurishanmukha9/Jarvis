"""
Tests for Skill Extractor: 13-domain technical skills taxonomy and extraction.
"""

from src.modules.career.scorer.utils.skill_extractor import extract_skills, get_all_skills_flat


def test_extract_skills_categorized():
    """Verify extracting skills categorized across technical domains."""
    text = "Full-stack engineer experienced in Python, React, PostgreSQL, Docker, AWS, PyTorch, and Kubernetes."
    skills = extract_skills(text)

    assert isinstance(skills, dict)
    assert len(skills) > 0

    all_matched = [s.lower() for cat in skills.values() for s in cat]
    assert "python" in all_matched
    assert "docker" in all_matched
    assert "react" in all_matched
    assert "postgresql" in all_matched


def test_extract_skills_case_insensitivity():
    """Verify skill extraction handles UPPERCASE, lowercase, and MixedCase tokens."""
    text = "Proficient in PYTHON, docker, PyTorch, and KUBERNETES."
    skills = extract_skills(text)
    all_matched = [s.lower() for cat in skills.values() for s in cat]
    assert "python" in all_matched
    assert "pytorch" in all_matched
    assert "kubernetes" in all_matched


def test_get_all_skills_flat():
    """Verify flat listing of extracted skill names."""
    text = "Skilled in Python, Docker, Redis, Git, Linux, and MongoDB."
    categorized = extract_skills(text)
    flat_set = get_all_skills_flat(categorized)
    assert isinstance(flat_set, (set, list))
    flat_lower = [s.lower() for s in flat_set]
    assert "python" in flat_lower
    assert "docker" in flat_lower
