"""
Campaign Manager for Smart HR Outreach & Cold Email Engine.
Handles recipient spreadsheet parsing, dynamic tag substitution, sequence generation,
and campaign state management.
"""

import io
import re
import csv
import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ...config import OUTREACH_DIR

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
CAMPAIGNS_LOG_FILE = OUTREACH_DIR / "campaigns.json"
ENGINE_TEMPLATES_FILE = MODULE_DIR / "templates" / "campaign_templates.json"
ENGINE_SAMPLE_CSV = MODULE_DIR / "data" / "sample_recipients_tech_recruiters.csv"
ENGINE_ANALYTICS_FILE = MODULE_DIR / "data" / "outreach_analytics.json"
_campaign_log_lock = threading.Lock()

class CampaignManager:
    """Manages cold email outreach campaigns, recipient parsing, and multi-stage sequences."""

    @staticmethod
    def get_template_library() -> Dict[str, Any]:
        """Load bundled outreach templates from outreach_engine/templates/."""
        if ENGINE_TEMPLATES_FILE.exists():
            try:
                with open(ENGINE_TEMPLATES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load engine templates file: {e}")
        return {}

    @staticmethod
    def get_sample_recipients_csv() -> str:
        """Load bundled tech recruiter leads CSV."""
        if ENGINE_SAMPLE_CSV.exists():
            try:
                with open(ENGINE_SAMPLE_CSV, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return (
            "email,firstName,company,role,department\n"
            "sarah.connor@techcorp.io,Sarah,TechCorp,VP of Engineering,AI Infrastructure\n"
            "alex.rivera@cloudscale.ai,Alex,CloudScale,Lead Recruiter,Talent Acquisition\n"
            "jordan.lee@fintechx.com,Jordan,FinTechX,Director of Engineering,Core Platform"
        )

    @staticmethod
    def get_benchmarks() -> Dict[str, Any]:
        """Load industry open rates and deliverability benchmarks."""
        if ENGINE_ANALYTICS_FILE.exists():
            try:
                with open(ENGINE_ANALYTICS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def parse_recipients_data(source: Any) -> List[Dict[str, Any]]:
        """
        Parse recipient records from CSV text, CSV/Excel bytes, or pandas DataFrame.
        Normalizes column headers to camelCase and snake_case tags.
        """
        records: List[Dict[str, Any]] = []
        df = None

        if isinstance(source, pd.DataFrame):
            df = source
        elif isinstance(source, str):
            # Parse raw CSV text
            try:
                df = pd.read_csv(io.StringIO(source))
            except Exception:
                # Line-by-line fallback
                lines = [l.strip() for l in source.strip().splitlines() if l.strip()]
                for line in lines:
                    if "@" in line:
                        parts = [p.strip() for p in line.split(",")]
                        email = parts[0] if "@" in parts[0] else (parts[1] if len(parts) > 1 and "@" in parts[1] else "")
                        name = parts[0] if email != parts[0] else ""
                        records.append({
                            "email": email,
                            "firstName": name.split()[0] if name else "there",
                            "name": name or "Colleague",
                            "company": "your organization",
                            "role": "Team Lead"
                        })
                return records
        elif isinstance(source, bytes):
            try:
                df = pd.read_excel(io.BytesIO(source))
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(source))
                except Exception as e:
                    logger.error(f"Failed to parse recipient file bytes: {e}")
                    return []

        if df is not None:
            # Clean dataframe
            df = df.dropna(how="all").fillna("")
            raw_records = df.to_dict(orient="records")
            
            for row in raw_records:
                clean_row: Dict[str, Any] = {}
                for k, v in row.items():
                    key_str = str(k).strip()
                    val_str = str(v).strip()
                    clean_row[key_str] = val_str
                    
                    # Also normalize keys
                    norm_k = re.sub(r'[^a-zA-Z0-9]', '', key_str)
                    norm_lower = norm_k.lower()
                    
                    if "email" in norm_lower:
                        clean_row["email"] = val_str
                    elif "first" in norm_lower or norm_lower == "fname":
                        clean_row["firstName"] = val_str
                    elif "last" in norm_lower or norm_lower == "lname":
                        clean_row["lastName"] = val_str
                    elif "name" in norm_lower and "firstName" not in clean_row:
                        clean_row["name"] = val_str
                        clean_row["firstName"] = val_str.split()[0] if val_str else "there"
                    elif "company" in norm_lower or "org" in norm_lower:
                        clean_row["company"] = val_str
                    elif "role" in norm_lower or "title" in norm_lower or "position" in norm_lower:
                        clean_row["role"] = val_str
                    elif "dept" in norm_lower or "department" in norm_lower:
                        clean_row["department"] = val_str

                # Fill default fallbacks
                if "email" not in clean_row:
                    for val in clean_row.values():
                        if isinstance(val, str) and "@" in val and "." in val:
                            clean_row["email"] = val
                            break
                
                if not clean_row.get("firstName"):
                    clean_row["firstName"] = clean_row.get("name", "there").split()[0] if clean_row.get("name") else "there"
                if not clean_row.get("company"):
                    clean_row["company"] = "your organization"
                if not clean_row.get("role"):
                    clean_row["role"] = "Hiring Manager"

                if clean_row.get("email"):
                    records.append(clean_row)

        return records

    @staticmethod
    def extract_template_tags(template_text: str) -> List[str]:
        """Extract all {tag} placeholder variables from a template string."""
        return list(set(re.findall(r'\{([a-zA-Z0-9_]+)\}', template_text)))

    @staticmethod
    def render_template(template_str: str, recipient: Dict[str, Any], global_tags: Optional[Dict[str, Any]] = None) -> str:
        """
        Substitute {tag} placeholders with recipient-specific and global variables.
        Gracefully replaces unknown tags with neutral defaults.
        """
        combined = {}
        if global_tags:
            combined.update(global_tags)
        combined.update(recipient)

        def replace_match(match):
            tag = match.group(1)
            # Case-insensitive lookup
            for k, v in combined.items():
                if k.lower() == tag.lower() and str(v).strip():
                    return str(v).strip()
            # Default fallbacks
            defaults = {
                "firstname": "there",
                "name": "there",
                "company": "your organization",
                "role": "Team Lead",
                "candidatename": "Job Seeker",
                "keyskills": "relevant domain skills",
                "keyachievement": "delivered high-impact technical results",
                "portfoliourl": ""
            }
            return defaults.get(tag.lower(), f"[{tag}]")

        return re.sub(r'\{([a-zA-Z0-9_]+)\}', replace_match, template_str)

    @staticmethod
    def build_multi_stage_sequence(
        target_role: str,
        target_company: str,
        candidate_name: str,
        key_skills: str,
        key_achievement: str = "",
        portfolio_url: str = ""
    ) -> List[Dict[str, str]]:
        """
        Generate a high-converting 4-stage cold outreach sequence.
        """
        achieve_text = key_achievement if key_achievement else "scaled distributed infrastructure and reduced latency by 35%"
        
        sequence = [
            {
                "stage": "Stage 1 — Initial Pitch (Day 1)",
                "delay_days": 0,
                "subject": f"Quick question regarding {target_role} at {target_company} — {candidate_name}",
                "body": (
                    f"Hi {{firstName}},\n\n"
                    f"I noticed {target_company} is growing its engineering team and actively seeking a {target_role}.\n\n"
                    f"With my background in {key_skills}, where I recently {achieve_text}, I believe I could make an immediate contribution to your upcoming technical roadmap.\n\n"
                    f"Would you be open to a brief 10-minute conversation next Tuesday or Wednesday to explore if my background aligns with your current priorities?\n\n"
                    f"Best regards,\n{candidate_name}\n{portfolio_url}"
                )
            },
            {
                "stage": "Stage 2 — Value-Add Case Study (Day 4)",
                "delay_days": 3,
                "subject": f"Thought on {target_company}'s engineering stack — {candidate_name}",
                "body": (
                    f"Hi {{firstName}},\n\n"
                    f"Following up on my previous message. I've been researching {target_company}'s recent technical initiatives in the {key_skills} space.\n\n"
                    f"In my past work, I tackled a similar challenge by streamlining core deployment pipelines and optimizing database throughput. I've documented some actionable architectural insights here: {portfolio_url or '[Portfolio Link]'}.\n\n"
                    f"Happy to share how these learnings could benefit {target_company} if you have 5 minutes this week.\n\n"
                    f"Best,\n{candidate_name}"
                )
            },
            {
                "stage": "Stage 3 — Soft Nudge (Day 8)",
                "delay_days": 7,
                "subject": f"Re: {target_role} at {target_company}",
                "body": (
                    f"Hi {{firstName}},\n\n"
                    f"I know how busy you must be managing priorities at {target_company}.\n\n"
                    f"Just wanted to check if hiring for the {target_role} position is still an active focus this quarter. I'd love to connect briefly whenever convenient.\n\n"
                    f"Best,\n{candidate_name}"
                )
            },
            {
                "stage": "Stage 4 — Graceful Breakup (Day 14)",
                "delay_days": 14,
                "subject": f"Permission to close the loop — {candidate_name}",
                "body": (
                    f"Hi {{firstName}},\n\n"
                    f"I assume timing might not be right currently for the {target_role} role at {target_company}, so I won't follow up further on this thread.\n\n"
                    f"If priorities shift down the road, please feel free to reach out anytime. Wishing you and {target_company} continued success!\n\n"
                    f"Warmly,\n{candidate_name}\n{portfolio_url}"
                )
            }
        ]
        return sequence

    @staticmethod
    def save_campaign_record(campaign_data: Dict[str, Any]) -> None:
        """Save campaign delivery record to persistent JSON store."""
        try:
            with _campaign_log_lock:
                records = []
                if CAMPAIGNS_LOG_FILE.exists():
                    with open(CAMPAIGNS_LOG_FILE, "r", encoding="utf-8") as f:
                        records = json.load(f)
                if not isinstance(records, list):
                    raise ValueError("Campaign history must be a JSON list.")
                records.append(campaign_data)
                temporary_path = CAMPAIGNS_LOG_FILE.with_suffix(".tmp")
                with open(temporary_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                temporary_path.replace(CAMPAIGNS_LOG_FILE)
        except Exception as e:
            logger.error(f"Failed to save campaign record: {e}")

    @staticmethod
    def get_campaign_history() -> List[Dict[str, Any]]:
        """Retrieve all recorded campaigns."""
        if not CAMPAIGNS_LOG_FILE.exists():
            return []
        try:
            with open(CAMPAIGNS_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
