"""
Unit tests for Goal Planner DAG decomposition and topological sorting.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from src.assistant.goal_planner import GoalPlanner, topological_sort


def test_topological_sort_linear_dependencies() -> None:
    """Test ordering tasks with linear dependencies: task_1 -> task_2 -> task_3."""
    tasks = [
        {"id": "task_3", "title": "Generate Report", "depends_on": ["task_2"]},
        {"id": "task_1", "title": "Fetch Data", "depends_on": []},
        {"id": "task_2", "title": "Process Stats", "depends_on": ["task_1"]},
    ]

    ordered = topological_sort(tasks)
    ordered_ids = [t["id"] for t in ordered]
    assert ordered_ids == ["task_1", "task_2", "task_3"]


def test_topological_sort_handles_cycles() -> None:
    """Test fallback when cyclical dependency is detected (task_1 <-> task_2)."""
    tasks = [
        {"id": "task_1", "title": "Task A", "depends_on": ["task_2"]},
        {"id": "task_2", "title": "Task B", "depends_on": ["task_1"]},
    ]

    ordered = topological_sort(tasks)
    assert len(ordered) == 2


def test_decompose_plan_from_json() -> None:
    """Test extracting JSON goal plan from LLM response."""
    planner = GoalPlanner(api_key="test_key")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="""```json
{
  "goal_summary": "Analyze AI market",
  "estimated_steps": 2,
  "tasks": [
    {"id": "task_1", "title": "Scrape Data", "instruction": "Fetch news", "depends_on": []},
    {"id": "task_2", "title": "Write Brief", "instruction": "Synthesize", "depends_on": ["task_1"]}
  ]
}
```"""
    )
    planner.llm = mock_llm

    plan = planner.plan_goal("Analyze AI market trends")
    assert plan["goal_summary"] == "Analyze AI market"
    assert len(plan["tasks"]) == 2
    assert plan["tasks"][0]["id"] == "task_1"
    assert plan["tasks"][1]["id"] == "task_2"
