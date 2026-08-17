"""
Unit tests for HR Outreach Sub-Agent tools and campaign management.
"""

from src.modules.outreach import (
    draft_personalized_outreach,
    generate_multi_stage_sequence,
    get_outreach_tools,
    preview_campaign_batch,
)


def test_outreach_tools_collection() -> None:
    """Ensure get_outreach_tools returns functional LangChain tools."""
    tools = get_outreach_tools()
    assert len(tools) >= 3
    tool_names = [t.name for t in tools]
    assert "draft_personalized_outreach" in tool_names
    assert "preview_campaign_batch" in tool_names
    assert "generate_multi_stage_sequence" in tool_names


def test_draft_personalized_outreach_formatting() -> None:
    """Test cold outreach email drafting with dynamic parameters."""
    draft = draft_personalized_outreach.invoke({
        "recipient_role": "Head of Engineering",
        "company": "Stark Industries",
        "candidate_background": "LLM Agents, Distributed Systems, PyTorch",
        "outreach_goal": "Explore Principal AI Architect opportunities",
        "tone": "Professional and concise",
    })
    assert isinstance(draft, str)
    assert "Stark Industries" in draft or "Head of Engineering" in draft


def test_generate_multi_stage_sequence_stages() -> None:
    """Test generating a 4-step follow-up email sequence."""
    seq = generate_multi_stage_sequence.invoke({
        "target_role": "Staff Backend Engineer",
        "target_company": "Acme Corp",
        "candidate_name": "Tony",
        "key_skills": "Distributed Systems, Go, Kubernetes",
    })
    assert isinstance(seq, str)
    assert "Stage 1" in seq or "Initial" in seq or "Follow" in seq or "Subject" in seq


def test_preview_campaign_batch_with_sample_leads() -> None:
    """Test campaign batch preview with simulated lead data."""
    sample_csv = "name,email,company,role\nAlice,alice@example.com,Acme Corp,Tech Lead\nBob,bob@example.com,Beta Inc,DevOps"

    preview = preview_campaign_batch.invoke({
        "subject_template": "Hi {name}, question for {company}",
        "body_template": "We saw your work in {role}.",
        "recipients_csv_text": sample_csv,
    })
    assert isinstance(preview, str)
    assert "alice@example.com" in preview or "Alice" in preview
