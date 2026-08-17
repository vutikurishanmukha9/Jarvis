"""
Unit tests for domain sub-agents registration and routing within JarvisOrchestrator.
"""

from unittest.mock import MagicMock, patch

from src.core.orchestrator import JarvisOrchestrator


def test_subagents_registration_with_all_tools() -> None:
    """Verify that all domain sub-agents (career, outreach, vision, doc) are registered."""
    mock_doc_tool = MagicMock()
    mock_doc_tool.name = "document_rag_search"

    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_create_agent.return_value = MagicMock()

        orchestrator = JarvisOrchestrator(
            api_key="test_key",
            document_tool=mock_doc_tool,
        )

        subagent_names = [sub.get("name") if isinstance(sub, dict) else sub.name for sub in orchestrator.subagents]
        assert "career_specialist" in subagent_names
        assert "outreach_specialist" in subagent_names
        assert "vision_analyst" in subagent_names
        assert "document_researcher" in subagent_names


def test_subagents_have_valid_descriptions_and_prompts() -> None:
    """Ensure every subagent has a non-empty description and specialized system prompt."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_create_agent:
        mock_create_agent.return_value = MagicMock()

        orchestrator = JarvisOrchestrator(api_key="test_key")

        for sub in orchestrator.subagents:
            name = sub.get("name") if isinstance(sub, dict) else sub.name
            desc = sub.get("description") if isinstance(sub, dict) else sub.description
            prompt = sub.get("system_prompt") if isinstance(sub, dict) else sub.system_prompt
            tools = sub.get("tools") if isinstance(sub, dict) else sub.tools
            assert len(name) > 0
            assert len(desc) > 10
            assert len(prompt) > 10
            assert len(tools) > 0
