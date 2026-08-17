"""
Tests for GoalPlanner schema validation, task bounding, and fallback plan generation.
"""

from src.assistant.goal_planner import PLANNING_SYSTEM_PROMPT, GoalPlanner
from src.config import MAX_AUTONOMOUS_SUBTASKS


def test_planning_prompt_schema_contract():
    """Verify system planning prompt defines JSON schema with depends_on."""
    assert "depends_on" in PLANNING_SYSTEM_PROMPT
    assert "expected_deliverable" in PLANNING_SYSTEM_PROMPT
    assert "tasks" in PLANNING_SYSTEM_PROMPT
    assert "goal_summary" in PLANNING_SYSTEM_PROMPT


def test_goal_planner_fallback_structure():
    """Verify fallback plan generation when LLM client is unconfigured or errors."""
    planner = GoalPlanner(api_provider="Custom", api_key="invalid_key", base_url="http://invalid-url")
    plan = planner.plan_goal("Create a market research report and dataset")

    assert plan["status"] == "planned"
    assert "tasks" in plan
    assert len(plan["tasks"]) == 2
    assert plan["tasks"][0]["id"] == "task_1"
    assert plan["tasks"][1]["id"] == "task_2"
    assert plan["tasks"][1]["depends_on"] == ["task_1"]
    assert plan["tasks"][0]["status"] == "pending"
    assert plan["tasks"][1]["status"] == "pending"


def test_goal_planner_initializes_required_fields():
    """Verify task fields (attempts, status, result) are initialized."""
    planner = GoalPlanner(api_provider="Custom", api_key="dummy")
    plan = planner.plan_goal("Build an analysis pipeline")

    for task in plan["tasks"]:
        assert task["status"] == "pending"
        assert task["attempts"] == 0
        assert task["result"] == ""
        assert "expected_deliverable" in task
        assert "instruction" in task


def test_goal_planner_task_count_bound():
    """Verify tasks do not exceed MAX_AUTONOMOUS_SUBTASKS."""
    planner = GoalPlanner(api_provider="Custom", api_key="dummy")
    plan = planner.plan_goal("Exhaustive multi-step project")
    assert len(plan["tasks"]) <= MAX_AUTONOMOUS_SUBTASKS


def test_goal_planner_context_injection():
    """Verify goal planner embeds context data into plan prompt without error."""
    planner = GoalPlanner(api_provider="Custom", api_key="dummy")
    context_data = "Competitor list: Alpha Inc, Beta LLC, Gamma Corp."
    plan = planner.plan_goal("Analyze competitor list", context=context_data)
    assert plan is not None
    assert len(plan["tasks"]) >= 1
