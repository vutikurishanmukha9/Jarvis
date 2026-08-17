"""
Unit tests validating the 6 Deep Agents architectural pillars in Jarvis:
1. Planning (Goal & subtask DAG execution)
2. Subagents (Specialist subagent registration & isolation)
3. Context (Window pruning & message token budgeting)
4. Skills (Dynamic skill capabilities)
5. Filesystem (Workspace sandbox file operations)
6. Tool Orchestration (Multi-modal tool routing & telemetry)
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.assistant.goal_planner import topological_sort
from src.assistant.workspace_tools import get_workspace_tools
from src.core.orchestrator import JarvisOrchestrator, ThoughtStepTracer
from src.core.session_manager import SessionManager


def test_pillar_1_planning_dag() -> None:
    """Validate Pillar 1: Planning with topological sorting of dependencies."""
    tasks = [
        {"id": "t2", "title": "Analyze Data", "depends_on": ["t1"]},
        {"id": "t1", "title": "Collect Input", "depends_on": []},
    ]
    sorted_tasks = topological_sort(tasks)
    assert [t["id"] for t in sorted_tasks] == ["t1", "t2"]


def test_pillar_2_subagents_registration() -> None:
    """Validate Pillar 2: Subagents modular registration."""
    with patch("src.core.orchestrator.create_deep_agent") as mock_agent:
        mock_agent.return_value = MagicMock()
        orchestrator = JarvisOrchestrator(api_key="test_key")
        sub_names = [s.get("name") if isinstance(s, dict) else s.name for s in orchestrator.subagents]
        assert "career_specialist" in sub_names
        assert "outreach_specialist" in sub_names
        assert "vision_analyst" in sub_names


def test_pillar_3_context_compaction() -> None:
    """Validate Pillar 3: Context pruning and sliding window management."""
    chat_history = [
        HumanMessage(content=f"Message {i}" * 50) for i in range(20)
    ]
    pruned = SessionManager.prune_context_window(chat_history, max_messages=6, max_chars=2000)
    assert len(pruned) <= 6


def test_pillar_4_skills_integration() -> None:
    """Validate Pillar 4: Skills discovery and extraction tools."""
    orchestrator = JarvisOrchestrator(api_key="test_key")
    tool_names = [t.name for t in orchestrator.tools]
    assert "extract_candidate_skills" in tool_names
    assert "predict_career_salary_and_role" in tool_names


def test_pillar_5_filesystem_sandbox() -> None:
    """Validate Pillar 5: Sandboxed filesystem and artifact generation tools."""
    ws_tools = get_workspace_tools()
    ws_names = [t.name for t in ws_tools]
    assert "write_workspace_file" in ws_names
    assert "read_workspace_file" in ws_names
    assert "generate_excel_spreadsheet" in ws_names
    assert "generate_word_document" in ws_names


def test_pillar_6_tool_orchestration_telemetry() -> None:
    """Validate Pillar 6: Tool orchestration and ThoughtStep telemetry."""
    tracer = ThoughtStepTracer()
    tracer.on_tool_start({"name": "web_search"}, "Quantum computing")
    tracer.on_tool_end("Search results returned 5 links")
    assert len(tracer.steps) == 2
    assert tracer.steps[0]["type"] == "tool_start"
    assert tracer.steps[1]["type"] == "tool_end"
