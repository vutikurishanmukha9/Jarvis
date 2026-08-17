"""
Tests for multi-session persistence, JSON serialization, and Markdown export.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.core.session_manager import SESSIONS_DIR, SessionManager


def test_session_creation_and_loading():
    """Verify creating a session, saving messages, and reloading it from disk."""
    session_id = "test_sess_001"
    messages = [
        HumanMessage(content="Analyze technical architecture."),
        AIMessage(content="The architecture is modular."),
    ]
    persona = "JARVIS Supreme"

    SessionManager.save_session(session_id, messages, persona)

    loaded_msgs, loaded_persona = SessionManager.load_session(session_id)
    assert len(loaded_msgs) == 2
    assert loaded_persona == persona
    assert loaded_msgs[0].content == "Analyze technical architecture."
    assert loaded_msgs[1].content == "The architecture is modular."


def test_session_listing_and_deletion():
    """Verify listing all active sessions and cleaning up session file."""
    session_id = "test_sess_002"
    SessionManager.save_session(session_id, [HumanMessage(content="Hello")], "JARVIS Supreme")

    sessions = SessionManager.list_sessions()
    assert session_id in sessions

    # Clean up file
    sess_file = SESSIONS_DIR / f"{session_id}.json"
    if sess_file.exists():
        sess_file.unlink()

    assert session_id not in SessionManager.list_sessions()


def test_session_markdown_transcript_export():
    """Verify formatting session history into a readable Markdown document."""
    session_id = "test_sess_export"
    messages = [
        HumanMessage(content="Run calculation for Q3."),
        AIMessage(content="Here is the projected Q3 revenue table."),
    ]
    persona = "Data & Vision Scientist"

    md_text = SessionManager.export_as_markdown(session_id, messages, persona)
    assert f"# J.A.R.V.I.S. Intelligence Briefing — {session_id}" in md_text
    assert f"**Persona**: {persona}" in md_text
    assert "### USER" in md_text
    assert "Run calculation for Q3." in md_text
    assert "### JARVIS" in md_text
    assert "Here is the projected Q3 revenue table." in md_text


def test_session_manager_nonexistent_handling():
    """Verify graceful handling when loading non-existent session IDs."""
    msgs, persona = SessionManager.load_session("non_existent_session_9999")
    assert msgs == []
    assert persona == "JARVIS Supreme"


def test_session_manager_rejects_path_traversal():
    """Session IDs must never be accepted as filesystem paths."""
    messages, persona = SessionManager.load_session("../../outside")
    assert messages == []
    assert persona == "JARVIS Supreme"
