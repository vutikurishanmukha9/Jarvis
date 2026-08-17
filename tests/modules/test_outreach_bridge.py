"""
Tests for Outreach Bridge tools and mandatory agent simulation security gate.
"""

from src.modules.outreach.outreach_bridge import (
    dispatch_email_campaign,
    draft_personalized_outreach,
    get_outreach_tools,
    preview_campaign_batch,
)


def test_outreach_tools_suite_registration():
    """Verify all 4 outreach engine tools are registered."""
    tools = get_outreach_tools()
    assert len(tools) == 4
    names = [t.name for t in tools]
    assert "draft_personalized_outreach" in names
    assert "generate_multi_stage_sequence" in names
    assert "preview_campaign_batch" in names
    assert "dispatch_email_campaign" in names


def test_draft_personalized_outreach():
    """Verify drafting outreach pitch for a given role and company."""
    draft = draft_personalized_outreach.invoke(
        {
            "recipient_role": "VP of Engineering",
            "company": "NeuralLink",
            "candidate_background": "Autonomous Agent Systems & Neural Interfaces",
        }
    )
    assert "Draft Cold Outreach Email" in draft
    assert "NeuralLink" in draft
    assert "VP of Engineering" in draft


def test_preview_campaign_batch_renders_samples():
    """Verify previewing campaign renders first recipient samples."""
    csv_text = "email,firstName,company,role\nrecruiter@apple.com,Sarah,Apple,Talent Lead"
    preview = preview_campaign_batch.invoke(
        {
            "subject_template": "Question regarding {role} at {company}",
            "body_template": "Hi {firstName}, reaching out to {company}.",
            "recipients_csv_text": csv_text,
        }
    )
    assert "Campaign Batch Preview" in preview
    assert "Sarah" in preview
    assert "Apple" in preview


def test_dispatch_tool_enforces_simulated_mode():
    """Verify dispatch_email_campaign always executes in simulation mode."""
    csv_text = "email,firstName,company,role\nrecruiter@google.com,Dave,Google,Recruiter"
    res = dispatch_email_campaign.invoke(
        {
            "subject_template": "Connecting with {company}",
            "body_template": "Hi {firstName}",
            "recipients_csv_text": csv_text,
        }
    )
    assert "Simulation" in res
    assert "SIMULATED: 1" in res.upper()
