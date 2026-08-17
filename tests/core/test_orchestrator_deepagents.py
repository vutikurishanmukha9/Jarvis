"""
Unit tests for JarvisOrchestrator Deep Agents integration.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.core.orchestrator import JarvisOrchestrator


def test_orchestrator_initialization() -> None:
    """Verify that JarvisOrchestrator initializes LLM, tools, subagents, and graph."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_graph = MagicMock()
        mock_create_agent.return_value = mock_graph

        orchestrator = JarvisOrchestrator(
            api_provider="OpenRouter",
            api_key="test_key",
            model_name="openai/gpt-4o",
            persona="JARVIS Supreme",
            deep_research_mode=False,
        )

        assert orchestrator.api_provider == "OpenRouter"
        assert orchestrator.model_name == "openai/gpt-4o"
        assert len(orchestrator.tools) >= 4  # python, web, vision, workspace
        assert len(orchestrator.subagents) >= 3  # career, outreach, vision
        assert orchestrator.agent_executor == mock_graph
        mock_create_agent.assert_called_once()


def test_orchestrator_deep_research_mode_prompt() -> None:
    """Verify deep research mode extends system prompt with structured instructions."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_create_agent.return_value = MagicMock()

        orchestrator = JarvisOrchestrator(
            api_key="test_key",
            deep_research_mode=True,
        )
        prompt = orchestrator._assemble_system_prompt()
        assert "[DEEP RESEARCH MODE ACTIVATED]" in prompt
        assert "Break the user's query into 2-4 critical research sub-questions" in prompt


def test_orchestrator_run_with_deep_agents_graph() -> None:
    """Verify orchestrator run() correctly passes messages to compiled graph and extracts AIMessage."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="Hello Jarvis"),
                AIMessage(content="Greetings! How may I assist you today?"),
            ]
        }
        mock_create_agent.return_value = mock_graph

        orchestrator = JarvisOrchestrator(api_key="test_key")
        result = orchestrator.run("Hello Jarvis", [])

        assert result["output"] == "Greetings! How may I assist you today?"
        assert isinstance(result["steps"], list)
        assert isinstance(result["figures"], list)
        assert isinstance(result["annotated_images"], list)
