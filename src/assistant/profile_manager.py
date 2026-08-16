"""
Personal Profile and Long-Term Memory Manager for Auto-JARVIS.
Persists user preferences, working habits, project knowledge, and custom instructions.

Memory Model:
Each memory entry contains:
- id:         Unique identifier (mem_<timestamp_ms>)
- fact:       The stored information
- category:   Classification (general, preference, project, skill, etc.)
- source:     Origin of the memory (conversation, user_explicit, agent_inferred)
- confidence: Reliability score 0.0-1.0 (1.0 = user-stated fact, 0.5 = agent-inferred)
- timestamp:  When the memory was created
- updated_at: When the memory was last updated (if ever)
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
    """Manages user profile configuration and persistent memory with full lifecycle support."""

    # --- Profile Management ---

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

    # --- Memory Management ---

    @staticmethod
    def _save_memories(memories: List[Dict[str, Any]]) -> bool:
        """Internal: persist the full memory list to disk."""
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memories, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
            return False

    @staticmethod
    def load_memories() -> List[Dict[str, Any]]:
        """Load persistent long-term memories."""
        if not MEMORY_FILE.exists():
            return []
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                memories = json.load(f)
            # Migrate old entries that lack new fields
            for mem in memories:
                if "source" not in mem:
                    mem["source"] = "conversation"
                if "confidence" not in mem:
                    mem["confidence"] = 1.0
                if "updated_at" not in mem:
                    mem["updated_at"] = None
            return memories
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            return []

    @staticmethod
    def add_memory(
        fact: str,
        category: str = "general",
        source: str = "conversation",
        confidence: float = 1.0
    ) -> bool:
        """
        Add a persistent memory fact about the user or projects.
        
        Args:
            fact: The information to remember.
            category: Classification (general, preference, project, skill, etc.)
            source: Origin of the memory (conversation, user_explicit, agent_inferred).
            confidence: Reliability score 0.0-1.0 (1.0 = user-stated fact).
        """
        memories = ProfileManager.load_memories()
        entry = {
            "id": f"mem_{int(time.time()*1000)}",
            "fact": fact.strip(),
            "category": category,
            "source": source,
            "confidence": max(0.0, min(1.0, confidence)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": None
        }
        memories.append(entry)
        return ProfileManager._save_memories(memories)

    @staticmethod
    def delete_memory(memory_id: str) -> bool:
        """
        Delete a specific memory by its ID.
        
        Args:
            memory_id: The unique ID of the memory to delete (e.g., "mem_1234567890").
            
        Returns:
            True if the memory was found and deleted, False otherwise.
        """
        memories = ProfileManager.load_memories()
        original_count = len(memories)
        memories = [m for m in memories if m.get("id") != memory_id]

        if len(memories) == original_count:
            logger.warning(f"Memory '{memory_id}' not found for deletion.")
            return False

        return ProfileManager._save_memories(memories)

    @staticmethod
    def update_memory(
        memory_id: str,
        new_fact: Optional[str] = None,
        new_category: Optional[str] = None,
        new_confidence: Optional[float] = None
    ) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            memory_id: The unique ID of the memory to update.
            new_fact: Updated fact text (None = keep existing).
            new_category: Updated category (None = keep existing).
            new_confidence: Updated confidence score (None = keep existing).
            
        Returns:
            True if the memory was found and updated, False otherwise.
        """
        memories = ProfileManager.load_memories()
        found = False

        for mem in memories:
            if mem.get("id") == memory_id:
                if new_fact is not None:
                    mem["fact"] = new_fact.strip()
                if new_category is not None:
                    mem["category"] = new_category
                if new_confidence is not None:
                    mem["confidence"] = max(0.0, min(1.0, new_confidence))
                mem["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                found = True
                break

        if not found:
            logger.warning(f"Memory '{memory_id}' not found for update.")
            return False

        return ProfileManager._save_memories(memories)

    @staticmethod
    def clear_memories() -> bool:
        """Clear all long-term memories safely across platforms."""
        try:
            return ProfileManager._save_memories([])
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False

    # --- Context Generation ---

    @staticmethod
    def get_assistant_system_context() -> str:
        """Construct the dynamic personal assistant system prompt injection."""
        profile = ProfileManager.load_profile()
        memories = ProfileManager.load_memories()

        if memories:
            # Show most recent 10 memories, sorted by confidence (high first)
            sorted_mems = sorted(memories, key=lambda m: m.get("confidence", 1.0), reverse=True)
            recent = sorted_mems[-10:]
            memory_bullets = "\n".join([
                f"- [{m.get('category', 'general')}] {m['fact']} "
                f"(confidence: {m.get('confidence', 1.0):.1f}, logged {m['timestamp']})"
                for m in recent
            ])
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

    format_context_for_prompt = get_assistant_system_context
