"""
Tests for Career Bridge: LangChain career tools, standalone resume parsing, and compensation estimation.
"""

import pytest
from src.modules.career.career_bridge import (
    analyze_resume_and_ats,
    extract_candidate_skills,
    predict_career_salary_and_role,
    get_salary_and_role_estimate,
    get_career_tools
)

def test_career_tools_suite_registration():
    """Verify all career intelligence tools are registered."""
    tools = get_career_tools()
    assert len(tools) == 3
    names = [t.name for t in tools]
    assert "analyze_resume_and_ats" in names
    assert "extract_candidate_skills" in names
    assert "predict_career_salary_and_role" in names

def test_standalone_resume_career_profile(sample_resume_text):
    """Verify analyzing a resume without a target job description."""
    res = analyze_resume_and_ats.invoke({"resume_text": sample_resume_text, "target_job_description": ""})
    assert "Standalone Resume Career Profile" in res
    assert "Detected Skills by Domain" in res

def test_salary_and_role_estimation_formula(sample_resume_text):
    """Verify compensation estimation calculates positive numbers, bounds, and confidence."""
    estimate = get_salary_and_role_estimate(sample_resume_text)
    assert "job_title" in estimate
    assert "salary_estimate" in estimate
    sal = estimate["salary_estimate"]
    assert sal["base"] > 0
    assert sal["range"]["min"] < sal["base"] < sal["range"]["max"]
    assert sal["currency"] == "₹"

def test_career_tool_too_short_text():
    """Verify graceful handling when very short resume text is passed."""
    res = analyze_resume_and_ats.invoke({"resume_text": "Short", "target_job_description": ""})
    assert "Please provide substantive resume text" in res
