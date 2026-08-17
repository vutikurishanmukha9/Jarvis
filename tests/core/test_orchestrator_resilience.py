"""
Unit tests for JarvisOrchestrator error resilience and fallback behavior.
"""

from unittest.mock import MagicMock, patch

from src.core.orchestrator import JarvisOrchestrator


def test_orchestrator_handles_agent_runtime_exceptions() -> None:
    """Verify that exceptions raised during agent invocation do not crash the orchestrator."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = TimeoutError("LLM Provider Timeout")
        mock_create_agent.return_value = mock_graph

        orchestrator = JarvisOrchestrator(api_key="test_key")
        result = orchestrator.run("Summarize quantum physics", [])

        assert "An error occurred during agent processing: LLM Provider Timeout" in result["output"]
        assert len(result["steps"]) >= 1
        assert result["steps"][-1]["type"] == "error"


def test_orchestrator_handles_empty_response() -> None:
    """Verify graceful handling when graph returns empty message list."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"messages": []}
        mock_create_agent.return_value = mock_graph

        orchestrator = JarvisOrchestrator(api_key="test_key")
        result = orchestrator.run("Test prompt", [])

        assert result["output"] == "I processed your request, but generated an empty response."
