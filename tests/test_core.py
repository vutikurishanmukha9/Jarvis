import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.config import PROVIDERS, PERSONAS
from src.core.session_manager import SessionManager

def test_config_and_personas():
    """Verify that all configured LLM providers and personas are present."""
    assert len(PROVIDERS) >= 3
    assert "OpenRouter" in PROVIDERS
    assert "OpenAI" in PROVIDERS
    assert "Custom" in PROVIDERS

    assert len(PERSONAS) >= 5
    assert "JARVIS Supreme" in PERSONAS
    assert "Career & Talent Strategist" in PERSONAS
    assert "HR & Executive Outreach Specialist" in PERSONAS

def test_session_manager_persistence():
    """Verify session save, list, load, and transcript export functionality."""
    session_id = "test_session_core_123"
    messages = [
        HumanMessage(content="Generate a quarterly analysis."),
        AIMessage(content="Here is the quarterly summary.")
    ]

    # Save session
    SessionManager.save_session(session_id, messages, persona="JARVIS Supreme")

    # List sessions
    sessions = SessionManager.list_sessions()
    assert session_id in sessions

    # Load session
    loaded_msgs, loaded_persona = SessionManager.load_session(session_id)
    assert len(loaded_msgs) == 2
    assert loaded_msgs[0].content == "Generate a quarterly analysis."
    assert loaded_persona == "JARVIS Supreme"

    # Export to markdown
    md_export = SessionManager.export_as_markdown(session_id, loaded_msgs, loaded_persona)
    assert "J.A.R.V.I.S. Intelligence Briefing" in md_export
    assert "Generate a quarterly analysis." in md_export
