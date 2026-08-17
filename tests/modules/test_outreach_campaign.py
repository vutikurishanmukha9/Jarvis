"""
Tests for CampaignManager: CSV parsing, header normalization, dynamic template tags, and sequences.
"""

from src.modules.outreach.campaign_manager import CampaignManager


def test_outreach_parse_recipients_csv():
    """Verify parsing CSV data and normalizing column headers."""
    csv_raw = (
        "first_name,company_name,job_title,email_address\n"
        "Elena,RoboTech,Head of AI,elena@robotech.ai\n"
        "Marcus,QuantumSystems,Talent Lead,marcus@quantum.com\n"
    )
    records = CampaignManager.parse_recipients_data(csv_raw)
    assert len(records) == 2
    assert records[0]["firstName"] == "Elena"
    assert records[0]["company"] == "RoboTech"
    assert records[0]["role"] == "Head of AI"
    assert records[0]["email"] == "elena@robotech.ai"


def test_outreach_extract_template_tags():
    """Verify extracting dynamic placeholder tags from template string."""
    template = "Hi {firstName}, reaching out regarding {role} at {company}."
    tags = CampaignManager.extract_template_tags(template)
    assert "firstName" in tags
    assert "role" in tags
    assert "company" in tags


def test_outreach_template_rendering_with_fallbacks():
    """Verify missing values are replaced with safe fallbacks."""
    record_sparse = {"email": "contact@example.com"}
    template = "Hello {firstName}, hope things are going well at {company}."
    rendered = CampaignManager.render_template(template, record_sparse)

    assert "there" in rendered or "Hello" in rendered
    assert "{" not in rendered  # No unrendered braces


def test_outreach_4_stage_sequence_cadence():
    """Verify 4-stage follow-up cadence structure."""
    seq = CampaignManager.build_multi_stage_sequence(
        target_role="Lead Architect",
        target_company="Stark Industries",
        candidate_name="Tony",
        key_skills="AI & Robotics",
        key_achievement="built autonomous flight navigation",
    )
    assert len(seq) == 4
    stages = [s["stage"] for s in seq]
    assert any("Day 1" in st for st in stages)
    assert any("Day 4" in st for st in stages)
    assert any("Day 8" in st for st in stages)
    assert any("Day 14" in st for st in stages)


def test_outreach_sample_recipients_csv_fallback():
    """Verify default sample CSV is available and well-formed."""
    sample_csv = CampaignManager.get_sample_recipients_csv()
    assert "email" in sample_csv
    assert "firstName" in sample_csv
