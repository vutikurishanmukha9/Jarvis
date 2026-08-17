"""
Tests for Sliding-Window Context Pruning and Token Budget Management.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.core.session_manager import SessionManager


def test_prune_context_window_under_limits():
    """Verify messages under limits are returned unmodified."""
    msgs = [
        HumanMessage(content="Hello Jarvis"),
        AIMessage(content="Hello! How may I assist you today?"),
        HumanMessage(content="Explain quantum computing in one sentence."),
    ]
    pruned = SessionManager.prune_context_window(msgs, max_messages=10, max_chars=5000)
    assert len(pruned) == 3
    assert pruned[0].content == "Hello Jarvis"
    assert pruned[2].content == "Explain quantum computing in one sentence."


def test_prune_context_window_message_count_truncation():
    """Verify that message count is pruned to keep first message + last (max_messages - 1)."""
    msgs = [HumanMessage(content=f"Message {i}") for i in range(25)]
    # Limit to 6 messages: should keep Message 0 and Message 20..24
    pruned = SessionManager.prune_context_window(msgs, max_messages=6, max_chars=50000)
    assert len(pruned) == 6
    assert pruned[0].content == "Message 0"
    assert pruned[-1].content == "Message 24"
    assert pruned[1].content == "Message 20"


def test_prune_context_window_character_budget_truncation():
    """Verify that character budget prunes older intermediate messages."""
    msgs = [
        HumanMessage(content="Primary instruction: Act as a senior software architect."),
        AIMessage(content="A" * 2000),
        HumanMessage(content="B" * 2000),
        AIMessage(content="C" * 2000),
        HumanMessage(content="Latest user request: summarize our progress."),
    ]
    # Total chars = ~6000. Limit to max_chars=3000
    pruned = SessionManager.prune_context_window(msgs, max_messages=20, max_chars=3000)

    total_pruned_chars = sum(len(m.content) for m in pruned)
    assert total_pruned_chars <= 3000
    assert pruned[0].content.startswith("Primary instruction")
    assert pruned[-1].content.startswith("Latest user request")


def test_prune_context_window_individual_message_capping():
    """Verify single massive message bodies (>4000 chars) are capped with a truncation marker."""
    massive_content = "X" * 6000
    msgs = [
        HumanMessage(content="Short intro"),
        AIMessage(content=massive_content),
        HumanMessage(content="Short follow-up"),
    ]
    pruned = SessionManager.prune_context_window(msgs, max_messages=10, max_chars=20000)
    assert len(pruned) == 3
    assert len(pruned[1].content) <= 4000
    assert "... [Context truncated for token efficiency]" in pruned[1].content


def test_prune_context_window_empty_or_single_message():
    """Verify handling of empty lists and single messages."""
    assert SessionManager.prune_context_window([]) == []

    single = [HumanMessage(content="Single query")]
    pruned_single = SessionManager.prune_context_window(single, max_messages=1)
    assert len(pruned_single) == 1
    assert pruned_single[0].content == "Single query"
