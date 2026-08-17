"""
Unit tests for Career & ATS Sub-Agent tools.
"""

from src.modules.career import (
    calculate_deep_ats_metrics,
    extract_candidate_skills,
    get_career_tools,
    get_salary_and_role_estimate,
)


def test_career_tools_collection() -> None:
    """Ensure get_career_tools returns functional LangChain structured tools."""
    tools = get_career_tools()
    assert len(tools) >= 3
    tool_names = [t.name for t in tools]
    assert "analyze_resume_and_ats" in tool_names
    assert "extract_candidate_skills" in tool_names
    assert "predict_career_salary_and_role" in tool_names


def test_extract_candidate_skills_real_text() -> None:
    """Test extracting technical and soft skills from resume text."""
    sample_resume = (
        "Senior Python Engineer with 6 years of experience in Django, FastAPI, Docker, Kubernetes, "
        "PostgreSQL, React, and Agile team leadership."
    )
    result = extract_candidate_skills.invoke({"text": sample_resume})
    assert isinstance(result, str)
    assert "Python" in result or "FastAPI" in result or "Docker" in result or "Skills" in result


def test_calculate_deep_ats_metrics_calculation() -> None:
    """Test ATS metric computation against job requirements."""
    resume_text = "Experienced with Python, SQL, AWS cloud infrastructure, CI/CD pipelines, and PyTest."
    jd_text = "Looking for a Python developer proficient in SQL, AWS, and unit testing."

    metrics = calculate_deep_ats_metrics(resume_text, jd_text)
    assert isinstance(metrics, dict)
    assert "ats_score" in metrics
    assert 0.0 <= float(metrics["ats_score"]) <= 100.0


def test_get_salary_and_role_estimate_logic() -> None:
    """Test salary and seniority prediction based on resume text."""
    resume = "Senior Python Developer with 5 years experience in machine learning, PyTorch, and distributed systems."
    estimate = get_salary_and_role_estimate(resume_text=resume)
    assert isinstance(estimate, dict)
    assert "job_title" in estimate
    assert "salary_estimate" in estimate
