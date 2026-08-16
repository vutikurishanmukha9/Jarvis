"""
Outreach Bridge for Jarvis Super-Intelligence.
Exposes LangChain agent tools for cold outreach copywriting, dynamic spreadsheet personalization,
multi-stage follow-up sequences, and batch campaign dispatching.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool, BaseTool

from .campaign_manager import CampaignManager
from .email_dispatcher import EmailDispatcher

logger = logging.getLogger(__name__)

@tool
def draft_personalized_outreach(
    recipient_role: str,
    company: str,
    candidate_background: str,
    outreach_goal: str = "Request a 10-minute exploratory conversation",
    tone: str = "Professional, concise, and value-driven"
) -> str:
    """
    Drafts a high-converting cold outreach email and follow-up sequence for recruiters, hiring managers, or executives.
    Uses dynamic personalization placeholders ({firstName}, {company}, {role}) and focuses on concrete achievements.
    Use this tool whenever the user wants to write a cold email pitch, connect with a recruiter, or apply for a position directly.
    """
    first_name_tag = "{firstName}"
    company_tag = "{company}"
    role_tag = "{role}"

    prompt_summary = (
        f"=== Draft Cold Outreach Email ===\n"
        f"Target: {recipient_role} at {company}\n"
        f"Objective: {outreach_goal}\n"
        f"Tone: {tone}\n\n"
        f"Subject Line Options:\n"
        f"1. Quick question regarding {role_tag} at {company_tag} — {{candidateName}}\n"
        f"2. {company_tag}'s {role_tag} team / {{candidateName}} introduction\n\n"
        f"Email Body Template:\n"
        f"Hi {first_name_tag},\n\n"
        f"I've been following {company_tag}'s technical advancements and noticed you're actively scaling the engineering team for the {role_tag} role.\n\n"
        f"With my background in {candidate_background}, where I recently led key technical deliverables, I believe I can drive measurable impact for your upcoming sprints.\n\n"
        f"Would you have 10 minutes next Tuesday or Wednesday for a brief chat to see if my skill set aligns with your team's goals?\n\n"
        f"Best regards,\n{{candidateName}}\n{{portfolioUrl}}"
    )
    return prompt_summary

@tool
def generate_multi_stage_sequence(
    target_role: str,
    target_company: str,
    candidate_name: str,
    key_skills: str,
    key_achievement: str = "",
    portfolio_url: str = ""
) -> str:
    """
    Generates a full 4-stage cold outreach campaign cadence (Day 1 Pitch, Day 4 Value Add, Day 8 Soft Nudge, Day 14 Breakup).
    Each stage contains an optimized subject line and personalized body copy.
    """
    seq = CampaignManager.build_multi_stage_sequence(
        target_role=target_role,
        target_company=target_company,
        candidate_name=candidate_name,
        key_skills=key_skills,
        key_achievement=key_achievement,
        portfolio_url=portfolio_url
    )

    out = [f"=== 4-Stage Multi-Touch Outreach Sequence for {target_role} at {target_company} ==="]
    for step in seq:
        out.append(f"\n[{step['stage']}]")
        out.append(f"Subject: {step['subject']}")
        out.append(f"Body:\n{step['body']}\n" + "-"*40)
    
    return "\n".join(out)

@tool
def preview_campaign_batch(
    subject_template: str,
    body_template: str,
    recipients_csv_text: str
) -> str:
    """
    Parses a CSV list of recipients and renders a live personalized preview for the first 3 recipients.
    Validates dynamic {tag} variables and reports detected columns.
    """
    records = CampaignManager.parse_recipients_data(recipients_csv_text)
    if not records:
        return "No valid recipient records could be parsed. Ensure the CSV contains an 'email' column."

    tags = CampaignManager.extract_template_tags(subject_template + " " + body_template)
    
    out = [
        f"=== Campaign Batch Preview ({len(records)} Total Recipients) ===",
        f"Detected Dynamic Tags in Template: {', '.join(['{' + t + '}' for t in tags]) or 'None'}",
        "\n--- Previewing First 3 Rendered Emails ---"
    ]

    for idx, rec in enumerate(records[:3]):
        rend_subj = CampaignManager.render_template(subject_template, rec)
        rend_body = CampaignManager.render_template(body_template, rec)
        out.append(f"\n[Recipient {idx+1}: {rec.get('email', 'No Email')} ({rec.get('company', 'No Org')})]")
        out.append(f"Subject: {rend_subj}")
        out.append(f"Body:\n{rend_body}\n" + "-"*30)

    return "\n".join(out)

@tool
def dispatch_email_campaign(
    subject_template: str,
    body_template: str,
    recipients_csv_text: str
) -> str:
    """
    Executes a personalized bulk cold outreach campaign in simulation mode.
    Formats all messages using dynamic {tag} personalization and writes an audit spreadsheet into the workspace.
    
    SECURITY: This tool always runs in simulation mode. Live SMTP dispatch
    requires explicit human approval through the Streamlit UI and cannot be
    triggered autonomously by the agent.
    """
    records = CampaignManager.parse_recipients_data(recipients_csv_text)
    if not records:
        return "Cannot execute campaign: No valid recipient records found in CSV."

    # SECURITY GATE: Agent-invoked dispatch is ALWAYS simulated.
    # Live dispatch must go through the UI with human-in-the-loop approval.
    result = EmailDispatcher.dispatch(
        subject_template=subject_template,
        body_template=body_template,
        recipients=records,
        simulated=True  # Hardcoded — agent cannot override
    )

    out = [
        "=== Campaign Dispatch Execution Report ===",
        f"• Status: {result.get('status', 'complete').upper()}",
        f"• Mode: Simulation (Agent-invoked dispatch is always simulated)",
        f"• Total Processed: {result.get('total', 0)} recipients",
        f"• Simulated: {result.get('sent', 0)}",
        f"• Failed: {result.get('failed', 0)}",
        f"• Audit Log File: {result.get('audit_file', 'Recorded in campaigns store')}",
        f"• Summary: {result.get('message', '')}",
        "",
        "Note: To send live emails, use the Streamlit UI with explicit human approval."
    ]
    return "\n".join(out)

def get_outreach_tools() -> List[BaseTool]:
    """Retrieve all Smart HR Outreach & Cold Email Engine tools."""
    return [
        draft_personalized_outreach,
        generate_multi_stage_sequence,
        preview_campaign_batch,
        dispatch_email_campaign
    ]
