"""
Session Management and Conversational Memory Persistence for Jarvis.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path("logs/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

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
        return f"session_{time.strftime('%Y%m%d_%H%M%S')}"

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
            
            file_path = SESSIONS_DIR / f"{session_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")

    @staticmethod
    def load_session(session_id: str) -> Tuple[List[BaseMessage], str]:
        """Load session messages and persona from disk."""
        file_path = SESSIONS_DIR / f"{session_id}.json"
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
