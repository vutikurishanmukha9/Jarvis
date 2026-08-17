"""
Career Intelligence & ATS Resume Engine package for Jarvis.
"""

from .career_bridge import (
    analyze_resume_and_ats,
    calculate_deep_ats_metrics,
    extract_candidate_skills,
    get_career_tools,
    get_resume_skills_categorized,
    get_salary_and_role_estimate,
    predict_career_salary_and_role,
)

__all__ = [
    "get_career_tools",
    "analyze_resume_and_ats",
    "extract_candidate_skills",
    "predict_career_salary_and_role",
    "calculate_deep_ats_metrics",
    "get_resume_skills_categorized",
    "get_salary_and_role_estimate",
]
