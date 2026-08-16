"""
Tests for 5-pillar ATS Scorer algorithm: formula weights, range clamping, and sub-score metrics.
"""

import pytest
from src.modules.career.scorer.services.ats_scorer import ATSScorer
from src.modules.career.scorer.services.ats_constants import SCORING_WEIGHTS

def test_ats_weights_sum_to_one():
    """Verify ATS formula weights mathematically sum to 1.00."""
    positive_sum = (
        SCORING_WEIGHTS["skill_match"] +
        SCORING_WEIGHTS["title_match"] +
        SCORING_WEIGHTS["experience"] +
        SCORING_WEIGHTS["achievement"] +
        SCORING_WEIGHTS["education"]
    )
    assert positive_sum == pytest.approx(0.95, abs=1e-5)
    assert SCORING_WEIGHTS["formatting_penalty"] == pytest.approx(0.05, abs=1e-5)

def test_ats_score_computation_deep_mode(sample_resume_text, sample_job_description):
    """Verify deep ATS analysis produces sub-scores, interpretations, and suggestions."""
    scorer = ATSScorer(mode="deep")
    res = scorer.calculate_ats_score(
        resume_text=sample_resume_text,
        jd_text=sample_job_description,
        jd_title="Senior Software Engineer",
        required_years=5
    )

    assert "ats_score" in res
    assert 0 <= res["ats_score"] <= 100
    assert "sub_scores" in res
    assert "interpretation" in res
    assert "missing_keywords" in res
    assert "suggestions" in res

    # Verify all 6 sub-scores are present
    sub = res["sub_scores"]
    assert "skill_match" in sub
    assert "title_match" in sub
    assert "experience" in sub
    assert "achievement" in sub
    assert "education" in sub
    assert "formatting_penalty" in sub

def test_ats_score_clamping_to_zero_to_hundred():
    """Verify ATS scores never exceed 100 or fall below 0 under extreme inputs."""
    scorer = ATSScorer(mode="deep")

    # Mismatched resume
    low_res = scorer.calculate_ats_score(
        resume_text="Art historian with focus on renaissance painting.",
        jd_text="Senior Site Reliability Engineer Kubernetes Terraform AWS."
    )
    assert 0 <= low_res["ats_score"] <= 100

    # Highly aligned resume
    high_res = scorer.calculate_ats_score(
        resume_text="Senior Python Developer 10 years experience Python PyTorch Docker Kubernetes AWS.",
        jd_text="Python Developer Python PyTorch Docker."
    )
    assert 0 <= high_res["ats_score"] <= 100

def test_ats_score_quick_scan_mode():
    """Verify fast keyword matching mode returns a valid ratio score."""
    scorer = ATSScorer(mode="quick")
    res = scorer.calculate_ats_score(
        resume_text="Python Docker AWS FastAPI",
        jd_text="Python Docker Kubernetes AWS"
    )
    assert res["mode"] == "quick"
    assert 0 <= res["ats_score"] <= 100
    assert len(res["matched_keywords"]) > 0

def test_ats_score_interpretation_badges():
    """Verify score interpretation provides human-readable status badge and color."""
    scorer = ATSScorer(mode="deep")
    res = scorer.calculate_ats_score(
        resume_text="Python developer with 3 years experience.",
        jd_text="Looking for a python engineer."
    )
    interp = res["interpretation"]
    assert "badge" in interp
    assert "color" in interp
    assert "message" in interp
