"""
Career Intelligence Bridge for Jarvis Super-Intelligence.
Integrates ATS Scoring, Skill Extraction, Salary Prediction, and Career Analytics.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool, BaseTool

from src.modules.career.scorer.services.ats_scorer import ATSScorer
from src.modules.career.scorer.services.analysis import analyze_resume, calculate_jd_resume_match
from src.modules.career.scorer.utils.skill_extractor import extract_skills, get_all_skills_flat
from src.modules.career.scorer.utils.keyword_extractor import extract_keywords
from src.modules.career.scorer.utils.feature_extractor import extract_resume_features

logger = logging.getLogger(__name__)

def calculate_deep_ats_metrics(resume_text: str, jd_text: str, mode: str = "deep") -> Dict[str, Any]:
    """
    Computes a comprehensive ATS compatibility audit across 5 pillars.
    """
    try:
        scorer = ATSScorer(mode=mode)
        results = scorer.calculate_ats_score(resume_text=resume_text, jd_text=jd_text)
        return results
    except Exception as e:
        logger.error(f"ATS scoring calculation failed: {str(e)}", exc_info=True)
        return {
            "ats_score": 0,
            "interpretation": {"badge": "Error", "color": "red", "message": str(e)},
            "sub_scores": {},
            "missing_keywords": {"critical": [], "important": [], "optional": []},
            "suggestions": [f"Error analyzing ATS score: {str(e)}"]
        }

def get_resume_skills_categorized(text: str) -> Dict[str, List[str]]:
    """Extract and categorize all skills found in text."""
    try:
        return extract_skills(text)
    except Exception as e:
        logger.error(f"Skill extraction failed: {str(e)}")
        return {}

def get_salary_and_role_estimate(resume_text: str) -> Dict[str, Any]:
    """Predict candidate's job role, category, and salary range."""
    try:
        predicted_job, matches, predicted_salary, salary_details = analyze_resume(resume_text)
        features = salary_details.get("features", {})
        base_sal = int(predicted_salary) if predicted_salary else 1500000
        return {
            "job_title": predicted_job,
            "category": "Technology" if any(t in str(predicted_job).lower() for t in ["engineer", "developer", "data", "software", "tech"]) else "General",
            "matches": matches,
            "salary_estimate": {
                "base": base_sal,
                "range": {
                    "min": int(base_sal * 0.85),
                    "max": int(base_sal * 1.15)
                },
                "currency": "₹",
                "confidence": "High" if features.get("years_experience", 0) > 0 else "Moderate"
            },
            "experience_years": features.get("years_experience", 0),
            "education_level": features.get("education_level", "Degree / Professional")
        }
    except Exception as e:
        logger.error(f"Salary and role prediction failed: {str(e)}")
        return {
            "job_title": "Software Professional",
            "category": "Technology",
            "matches": [],
            "salary_estimate": {
                "base": 1500000,
                "range": {"min": 1200000, "max": 1800000},
                "currency": "₹",
                "confidence": "Moderate"
            },
            "experience_years": 3,
            "education_level": "Bachelor / Master",
            "error": str(e)
        }

@tool
def analyze_resume_and_ats(resume_text: str, target_job_description: str = "") -> str:
    """
    Performs an in-depth ATS (Applicant Tracking System) compatibility analysis on a candidate's resume text.
    Evaluates keyword matches, technical skill coverage, experience duration, education, and formatting.
    Identifies missing critical keywords and actionable bullet-point improvements to maximize interview callbacks.
    Use this tool whenever the user asks to review, audit, critique, or score a resume against a job description.
    """
    if not resume_text or len(resume_text.strip()) < 30:
        return "Please provide substantive resume text (at least 30 characters) to analyze."

    if not target_job_description:
        # Standalone resume analysis
        role_info = get_salary_and_role_estimate(resume_text)
        skills = get_resume_skills_categorized(resume_text)
        
        flat_skills = []
        for cat, items in skills.items():
            flat_skills.append(f"  - {cat.replace('_', ' ').title()}: {', '.join(items)}")
        skills_str = "\n".join(flat_skills) if flat_skills else "  - No standard technical skills detected"

        sal = role_info.get("salary_estimate", {})
        sal_str = ""
        if sal and "range" in sal:
            sal_str = f"- Estimated Market Salary: {sal.get('currency', '₹')}{sal['range'].get('min', 0):,} - {sal['range'].get('max', 0):,} ({sal.get('confidence', 'Moderate')} confidence)\n"

        out = (
            f"=== Standalone Resume Career Profile ===\n"
            f"- Predicted Best-Fit Job Title: {role_info.get('job_title')}\n"
            f"- Industry Category: {role_info.get('category')}\n"
            f"- Experience Level: {role_info.get('experience_years')} years\n"
            f"- Education Detected: {role_info.get('education_level')}\n"
            f"{sal_str}"
            f"- Detected Skills by Domain:\n{skills_str}\n\n"
            f"Tip: Provide a specific Job Description to calculate a full 0-100 ATS compatibility score and uncover missing critical keywords."
        )
        return out

    # Full ATS Comparison
    ats_results = calculate_deep_ats_metrics(resume_text, target_job_description)
    score = ats_results.get("ats_score", 0)
    interp = ats_results.get("interpretation", {})
    sub = ats_results.get("sub_scores", {})
    missing = ats_results.get("missing_keywords", {})
    suggestions = ats_results.get("suggestions", [])

    crit_miss = ", ".join(missing.get("critical", [])) or "None (all critical keywords matched)"
    imp_miss = ", ".join(missing.get("important", [])) or "None"
    
    sugg_str = "\n".join([f"  {idx+1}. {s}" for idx, s in enumerate(suggestions[:5])])

    out = (
        f"=== ATS Resume Compatibility Audit ===\n"
        f"- Overall ATS Score: {score}/100 ({interp.get('badge', 'Status')} — {interp.get('message', '')})\n"
        f"- Sub-Scores Breakdown:\n"
        f"  * Skill Match: {sub.get('skill_match', 0)}/100\n"
        f"  * Title Match: {sub.get('title_match', 0)}/100\n"
        f"  * Experience Match: {sub.get('experience', 0)}/100\n"
        f"  * Achievement Impact: {sub.get('achievement', 0)}/100\n"
        f"  * Education Match: {sub.get('education', 0)}/100\n"
        f"  * Formatting & Hygiene: {100 - sub.get('formatting_penalty', 0)}/100\n\n"
        f"- Critical Missing Keywords (Must Add): {crit_miss}\n"
        f"- Important Missing Keywords: {imp_miss}\n\n"
        f"- Priority Suggestions for Optimization:\n{sugg_str}"
    )
    return out

@tool
def extract_candidate_skills(text: str) -> str:
    """
    Extracts all technical, cloud, data science, programming, and domain skills from a resume, job description, or profile text.
    Organizes skills into categories (e.g. Programming Languages, Cloud/DevOps, AI/ML, Databases, Web Frameworks).
    """
    if not text or len(text.strip()) < 10:
        return "No text provided for skill extraction."
    
    skills = get_resume_skills_categorized(text)
    if not skills:
        return "No known technical or domain skills were identified in the provided text."
    
    lines = ["=== Identified Candidate Skills ==="]
    total_count = 0
    for category, items in skills.items():
        cat_name = category.replace("_", " ").title()
        lines.append(f"• {cat_name} ({len(items)}): {', '.join(items)}")
        total_count += len(items)
    
    lines.append(f"\nTotal Technical Skills Detected: {total_count}")
    return "\n".join(lines)

@tool
def predict_career_salary_and_role(resume_text: str) -> str:
    """
    Predicts a candidate's market value, salary range, and best-fit job classification based on experience, education, and skills.
    Use this tool to evaluate market compensation and career trajectory.
    """
    if not resume_text or len(resume_text.strip()) < 30:
        return "Please provide complete resume text to evaluate salary and role expectations."
    
    info = get_salary_and_role_estimate(resume_text)
    sal = info.get("salary_estimate", {})
    curr = sal.get("currency", "₹")

    out = [
        "=== Career & Compensation Projection ===",
        f"• Target Job Classification: {info.get('job_title')}",
        f"• Industry Domain: {info.get('category')}",
        f"• Experience Evaluation: {info.get('experience_years')} years",
        f"• Highest Education Tier: {info.get('education_level')}"
    ]

    if sal and "range" in sal:
        min_s = sal["range"].get("min", 0)
        max_s = sal["range"].get("max", 0)
        base = sal.get("base", 0)
        out.append(f"• Estimated Market Base: {curr}{base:,}")
        out.append(f"• Competitive Salary Band: {curr}{min_s:,} – {curr}{max_s:,}")
        out.append(f"• Prediction Confidence: {sal.get('confidence', 'Moderate')}")
    
    return "\n".join(out)

def get_career_tools() -> List[BaseTool]:
    """Retrieve all Career Intelligence and Resume Optimization tools."""
    return [
        analyze_resume_and_ats,
        extract_candidate_skills,
        predict_career_salary_and_role
    ]

