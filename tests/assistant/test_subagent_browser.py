"""
Unit tests validating the autonomous Web Navigation specialist sub-agent (`browser_specialist`).
"""

from unittest.mock import MagicMock, patch

from src.core.orchestrator import JarvisOrchestrator
from src.tools.browser_tools import get_browser_tools


def test_browser_specialist_subagent_registered() -> None:
    """Test browser_specialist subagent is constructed with all 5 browser capabilities."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_agent:
        mock_agent.return_value = MagicMock()
        orchestrator = JarvisOrchestrator(api_key="test_key")

        sub_names = [s.name if hasattr(s, "name") else s["name"] for s in orchestrator.subagents]
        assert "browser_specialist" in sub_names

        # Find browser specialist subagent
        browser_subagent = next(
            s for s in orchestrator.subagents if (s.name if hasattr(s, "name") else s["name"]) == "browser_specialist"
        )
        tools = browser_subagent.tools if hasattr(browser_subagent, "tools") else browser_subagent["tools"]
        tool_names = [t.name for t in tools]

        assert "navigate_to_url" in tool_names
        assert "browser_click_element" in tool_names
        assert "browser_fill_form" in tool_names
        assert "browser_scrape_page" in tool_names
        assert "browser_scroll" in tool_names


def test_browser_tools_included_in_orchestrator_tools() -> None:
    """Test browser tools are present in primary orchestrator tool list."""
    orchestrator = JarvisOrchestrator(api_key="test_key")
    tool_names = [t.name for t in orchestrator.tools]
    for b_tool in get_browser_tools():
        assert b_tool.name in tool_names
