"""
Session Management and Conversational Memory Persistence for Jarvis.
"""

import json
import logging
import time
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path("logs/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_file(session_id: str) -> Path:
    """Return a safe session file path; session IDs are never treated as paths."""
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise ValueError("Invalid session ID.")
    return SESSIONS_DIR / f"{session_id}.json"

class SessionManager:
    """Manages chat sessions, history persistence, and exports."""
    
    @staticmethod
    def list_sessions() -> List[str]:
        """List all saved session IDs."""
        try:
            files = list(SESSIONS_DIR.glob("*.json"))
            return sorted([f.stem for f in files], reverse=True)
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []

    @staticmethod
    def generate_session_id() -> str:
        """Create a new timestamp-based session ID."""
        return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def save_session(session_id: str, messages: List[BaseMessage], persona: str = "JARVIS Supreme"):
        """Save session messages and metadata to disk."""
        try:
            serializable_msgs = []
            for msg in messages:
                role = "human" if isinstance(msg, HumanMessage) else "ai"
                serializable_msgs.append({
                    "role": role,
                    "content": msg.content
                })
            
            data = {
                "session_id": session_id,
                "persona": persona,
                "timestamp": time.time(),
                "messages": serializable_msgs
            }
            
            file_path = _session_file(session_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")

    @staticmethod
    def load_session(session_id: str) -> Tuple[List[BaseMessage], str]:
        """Load session messages and persona from disk."""
        try:
            file_path = _session_file(session_id)
        except ValueError:
            return [], "JARVIS Supreme"
        if not file_path.exists():
            return [], "JARVIS Supreme"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            messages: List[BaseMessage] = []
            for m in data.get("messages", []):
                if m.get("role") == "human":
                    messages.append(HumanMessage(content=m.get("content", "")))
                else:
                    messages.append(AIMessage(content=m.get("content", "")))
                    
            persona = data.get("persona", "JARVIS Supreme")
            return messages, persona
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return [], "JARVIS Supreme"

    @staticmethod
    def export_as_markdown(session_id: str, messages: List[BaseMessage], persona: str) -> str:
        """Export session as a formatted Markdown transcript."""
        lines = [
            f"# J.A.R.V.I.S. Intelligence Briefing — {session_id}",
            f"**Persona**: {persona}",
            f"**Exported At**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "---",
            ""
        ]
        for msg in messages:
            role = "USER" if isinstance(msg, HumanMessage) else "JARVIS"
            lines.append(f"### {role}")
            lines.append(msg.content)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def prune_context_window(
        messages: List[BaseMessage],
        max_messages: int = 20,
        max_chars: int = 16000
    ) -> List[BaseMessage]:
        """
        Prune conversational history using a sliding window to fit within LLM token/context budgets.
        
        Rules:
        - If messages count <= max_messages and total length <= max_chars, returns unmodified list.
        - Keeps the first message if it establishes primary user intent / system role.
        - Retains the most recent (max_messages - 1) messages.
        - Truncates oversized message bodies to prevent single-turn token exhaustion.
        """
        if not messages:
            return []

        # 1. Message Count Pruning: Keep first message (if exists) + last (max_messages - 1)
        if len(messages) > max_messages:
            if max_messages <= 1:
                pruned = messages[-1:]
            else:
                pruned = [messages[0]] + list(messages[-(max_messages - 1):])
        else:
            pruned = list(messages)

        # 2. Character Budget Guard: Calculate total length and truncate from older turns if needed
        total_chars = sum(len(str(getattr(m, "content", ""))) for m in pruned)
        
        if total_chars > max_chars:
            while len(pruned) > 2 and total_chars > max_chars:
                removed = pruned.pop(1)
                total_chars -= len(str(getattr(removed, "content", "")))

        # 3. Individual Message Body Cap (max 4000 chars per individual historical turn)
        bounded: List[BaseMessage] = []
        for msg in pruned:
            content = str(getattr(msg, "content", ""))
            if len(content) > 4000:
                truncated_content = content[:3900] + "\n... [Context truncated for token efficiency]"
                if isinstance(msg, HumanMessage):
                    bounded.append(HumanMessage(content=truncated_content))
                else:
                    bounded.append(AIMessage(content=truncated_content))
            else:
                bounded.append(msg)

        return bounded
