"""
Personal Profile and Long-Term Memory Manager for Auto-JARVIS.
Persists user preferences, working habits, project knowledge, and custom instructions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import ASSISTANT_DIR, DEFAULT_USER_NAME, DEFAULT_ASSISTANT_NAME

logger = logging.getLogger(__name__)

PROFILE_FILE = ASSISTANT_DIR / "profile.json"
MEMORY_FILE = ASSISTANT_DIR / "long_term_memory.json"

DEFAULT_PROFILE: Dict[str, Any] = {
    "user_name": DEFAULT_USER_NAME,
    "assistant_name": DEFAULT_ASSISTANT_NAME,
    "role_description": "Visionary Creator & Lead Engineer",
    "preferred_style": "Concise, highly structured, executive-ready, proactive",
    "auto_execute_safe_code": True,
    "default_workspace": "workspace",
    "custom_instructions": "Always deliver actionable outputs. When writing reports or datasets, format cleanly and save to workspace."
}

class ProfileManager:
    """Manages user profile configuration and persistent memory."""

    @staticmethod
    def load_profile() -> Dict[str, Any]:
        """Load user profile from disk or initialize with defaults."""
        if not PROFILE_FILE.exists():
            ProfileManager.save_profile(DEFAULT_PROFILE)
            return DEFAULT_PROFILE.copy()

        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge with defaults for any missing keys
            profile = DEFAULT_PROFILE.copy()
            profile.update(data)
            return profile
        except Exception as e:
            logger.error(f"Error loading profile: {str(e)}")
            return DEFAULT_PROFILE.copy()

    @staticmethod
    def save_profile(profile_data: Dict[str, Any]) -> bool:
        """Save user profile to disk."""
        try:
            with open(PROFILE_FILE, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            return False

    @staticmethod
    def load_memories() -> List[Dict[str, Any]]:
        """Load persistent long-term memories."""
        if not MEMORY_FILE.exists():
            return []
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            return []

    @staticmethod
    def add_memory(fact: str, category: str = "general") -> bool:
        """Add a persistent memory fact about the user or projects."""
        memories = ProfileManager.load_memories()
        entry = {
            "id": f"mem_{int(time.time()*1000)}",
            "fact": fact.strip(),
            "category": category,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        memories.append(entry)
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memories, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
            return False

    @staticmethod
    def clear_memories() -> bool:
        """Clear all long-term memories."""
        try:
            if MEMORY_FILE.exists():
                MEMORY_FILE.unlink()
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False

    @staticmethod
    def get_assistant_system_context() -> str:
        """Construct the dynamic personal assistant system prompt injection."""
        profile = ProfileManager.load_profile()
        memories = ProfileManager.load_memories()

        memory_bullets = ""
        if memories:
            memory_bullets = "\n".join([f"- {m['fact']} (logged {m['timestamp']})" for m in memories[-10:]])
        else:
            memory_bullets = "No prior long-term memories logged yet."

        return (
            f"\n\n[PERSONAL ASSISTANT PROTOCOL - ASSIGNED TO {profile.get('user_name', 'USER').upper()}]:\n"
            f"- User Name: {profile.get('user_name', 'Boss')}\n"
            f"- User Role: {profile.get('role_description', 'Creator')}\n"
            f"- Preferred Working Style: {profile.get('preferred_style', 'Executive')}\n"
            f"- Custom Directives: {profile.get('custom_instructions', 'Deliver complete work.')}\n\n"
            f"LONG-TERM MEMORY CONTEXT:\n{memory_bullets}\n"
            f"Always operate with extreme proactivity. If a goal requires multiple steps or files, "
            f"execute them completely and save all artifacts in the workspace directory."
        )
