"""
Smart HR Outreach & Cold Email Engine for Jarvis.
Provides recipient spreadsheet parsing, dynamic tag substitution,
multi-stage follow-up sequence generation, and campaign delivery tracking.
"""

from .campaign_manager import CampaignManager
from .email_dispatcher import EmailDispatcher
from .outreach_bridge import (
    dispatch_email_campaign,
    draft_personalized_outreach,
    generate_multi_stage_sequence,
    get_outreach_tools,
    preview_campaign_batch,
)

__all__ = [
    "CampaignManager",
    "EmailDispatcher",
    "get_outreach_tools",
    "draft_personalized_outreach",
    "preview_campaign_batch",
    "dispatch_email_campaign",
    "generate_multi_stage_sequence",
]
