import pytest
import io
import pandas as pd
from PIL import Image

from src.modules.vision import (
    get_vision_tools,
    register_uploaded_image,
    clear_active_images,
    _ACTIVE_IMAGES
)
from src.modules.career import (
    calculate_deep_ats_metrics,
    analyze_resume_and_ats,
    extract_candidate_skills,
    predict_career_salary_and_role,
    get_career_tools
)
from src.modules.outreach import (
    CampaignManager,
    EmailDispatcher,
    draft_personalized_outreach,
    generate_multi_stage_sequence,
    preview_campaign_batch,
    dispatch_email_campaign,
    get_outreach_tools
)

# 1. Vision Tests
def test_vision_bridge_registration():
    """Verify image registration, memory storage, and tool retrieval."""
    clear_active_images()
    assert len(_ACTIVE_IMAGES) == 0

    # Create dummy image in memory
    img = Image.new("RGB", (320, 240), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class DummyUploadedFile:
        def __init__(self, name, data):
            self.name = name
            self._data = data
        def getvalue(self):
            return self._data

    dummy_file = DummyUploadedFile("test_camera.jpg", buf.getvalue())
    res = register_uploaded_image(dummy_file)
    assert res["status"] == "success"
    assert res["dimensions"] == (320, 240)
    assert "test_camera.jpg" in _ACTIVE_IMAGES

    tools = get_vision_tools()
    assert len(tools) == 1
    assert tools[0].name == "analyze_uploaded_images"

# 2. Career Tests
def test_career_ats_scoring():
    """Verify ATS Scorer computes compatibility, sub-scores, and suggestions."""
    resume = "Senior Python Developer with 5 years experience in PyTorch, Docker, FastAPI, and PostgreSQL. Master of Science in Computer Science."
    jd = "Seeking a Senior Python Developer with 5+ years experience. Required: Python, PyTorch, Docker, Kubernetes, PostgreSQL. MS in Computer Science."

    metrics = calculate_deep_ats_metrics(resume, jd)
    assert metrics["ats_score"] > 0
    assert "sub_scores" in metrics
    assert "interpretation" in metrics
    assert "missing_keywords" in metrics

    tool_res = analyze_resume_and_ats.invoke({"resume_text": resume, "target_job_description": jd})
    assert "ATS Resume Compatibility Audit" in tool_res
    assert "Overall ATS Score" in tool_res

def test_career_skill_and_salary_tools():
    """Verify Candidate Skill Extractor and Salary Prediction tools."""
    profile_text = "Experienced software engineer with proficiency in Python, PyTorch, AWS, Docker, and Kubernetes. 6 years experience."
    
    # 1. Skill extractor
    skills_out = extract_candidate_skills.invoke({"text": profile_text})
    assert "Identified Candidate Skills" in skills_out
    assert "python" in skills_out.lower()

    # 2. Salary prediction
    salary_out = predict_career_salary_and_role.invoke({"resume_text": profile_text})
    assert "Career & Compensation Projection" in salary_out

    # 3. Tool registry
    tools = get_career_tools()
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert "analyze_resume_and_ats" in tool_names
    assert "extract_candidate_skills" in tool_names
    assert "predict_career_salary_and_role" in tool_names

# 3. Outreach Tests
def test_outreach_campaign_manager():
    """Verify CampaignManager parses CSV data, renders tags, and builds 4-stage sequences."""
    csv_data = (
        "email,firstName,company,role,department\n"
        "alex@company.com,Alex,Acme Corp,VP Engineering,AI Systems\n"
        "sam@startup.io,Sam,StartupLab,Recruiter,Talent"
    )
    
    # 1. Parse CSV
    records = CampaignManager.parse_recipients_data(csv_data)
    assert len(records) == 2
    assert records[0]["email"] == "alex@company.com"
    assert records[0]["firstName"] == "Alex"
    assert records[0]["company"] == "Acme Corp"

    # 2. Template Rendering
    template = "Hi {firstName}, reaching out regarding {role} at {company}."
    rendered = CampaignManager.render_template(template, records[0])
    assert "Hi Alex, reaching out regarding VP Engineering at Acme Corp." == rendered

    # 3. Multi-Stage Sequence Generation
    seq = CampaignManager.build_multi_stage_sequence(
        target_role="Staff Engineer",
        target_company="Acme Corp",
        candidate_name="John Doe",
        key_skills="Python & PyTorch",
        key_achievement="built high-throughput inference engine"
    )
    assert len(seq) == 4
    assert "Stage 1" in seq[0]["stage"]
    assert "Stage 4" in seq[3]["stage"]

def test_outreach_bridge_and_dispatcher():
    """Verify Outreach Agent Tools and simulated campaign dispatcher."""
    csv_sample = "email,firstName,company,role\nrecruiter@meta.com,Sarah,Meta,Lead Recruiter"

    # 1. Draft outreach
    draft = draft_personalized_outreach.invoke({
        "recipient_role": "Lead Recruiter",
        "company": "Meta",
        "candidate_background": "Python AI and Computer Vision systems"
    })
    assert "Draft Cold Outreach Email" in draft

    # 2. Preview batch
    preview = preview_campaign_batch.invoke({
        "subject_template": "Question for {firstName} at {company}",
        "body_template": "Hi {firstName}, exploring {role} opportunities.",
        "recipients_csv_text": csv_sample
    })
    assert "Campaign Batch Preview" in preview
    assert "recruiter@meta.com" in preview

    # 3. Simulated Dispatch
    dispatch_res = dispatch_email_campaign.invoke({
        "subject_template": "Connecting with {firstName} at {company}",
        "body_template": "Hi {firstName}, hello from Jarvis.",
        "recipients_csv_text": csv_sample,
        "simulated": True
    })
    assert "Campaign Dispatch Execution Report" in dispatch_res
    assert "DELIVERED / SIMULATED: 1" in dispatch_res.upper()

    # 4. Tool Registry
    tools = get_outreach_tools()
    assert len(tools) == 4
    names = [t.name for t in tools]
    assert "draft_personalized_outreach" in names
    assert "generate_multi_stage_sequence" in names
    assert "preview_campaign_batch" in names
    assert "dispatch_email_campaign" in names
