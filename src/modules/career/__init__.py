"""
Career Intelligence & ATS Resume Engine package for Jarvis.
"""

from .career_bridge import (
    get_career_tools,
    analyze_resume_and_ats,
    extract_candidate_skills,
    predict_career_salary_and_role,
    calculate_deep_ats_metrics,
    get_resume_skills_categorized,
    get_salary_and_role_estimate
)

__all__ = [
    "get_career_tools",
    "analyze_resume_and_ats",
    "extract_candidate_skills",
    "predict_career_salary_and_role",
    "calculate_deep_ats_metrics",
    "get_resume_skills_categorized",
    "get_salary_and_role_estimate"
]
