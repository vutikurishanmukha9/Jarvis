"""
Tests for Autonomous Mission Governor: timeouts, retry budgets, and graceful degradation.
"""

from src.assistant.autonomous_runner import AutonomousRunner


def test_autonomous_governor_respects_max_mission_duration():
    """Verify that a plan exceeding max_mission_duration_seconds halts execution gracefully."""
    runner = AutonomousRunner(orchestrator=None)  # Simulated execution mode

    # Create a 4-task plan
    plan = {
        "tasks": [
            {"id": "t1", "title": "Subtask 1", "instruction": "Step 1", "depends_on": []},
            {"id": "t2", "title": "Subtask 2", "instruction": "Step 2", "depends_on": ["t1"]},
            {"id": "t3", "title": "Subtask 3", "instruction": "Step 3", "depends_on": ["t2"]},
            {"id": "t4", "title": "Subtask 4", "instruction": "Step 4", "depends_on": ["t3"]},
        ]
    }

    # Execute with duration budget = -1.0 (immediate timeout trigger)
    result = runner.execute_plan(
        plan=plan, initial_goal="Test timeout governance", max_mission_duration_seconds=-0.1, max_cumulative_retries=6
    )

    # Verify tasks are marked as skipped due to timeout
    assert result["plan"]["status"] == "partially_completed"
    assert result["plan"]["tasks"][0]["status"] == "skipped_due_to_timeout"
    assert "exceeded budget" in result["plan"]["tasks"][0]["result"]


def test_autonomous_governor_respects_max_cumulative_retries():
    """Verify that cumulative retries exceeding the limit halts execution gracefully."""
    runner = AutonomousRunner(orchestrator=None)

    plan = {
        "tasks": [
            {"id": "t1", "title": "Task 1", "instruction": "Execute 1"},
            {"id": "t2", "title": "Task 2", "instruction": "Execute 2"},
        ]
    }

    # max_cumulative_retries = 0 triggers immediate cutoff
    result = runner.execute_plan(
        plan=plan,
        initial_goal="Test retry limit governance",
        max_mission_duration_seconds=300.0,
        max_cumulative_retries=0,
    )

    assert result["plan"]["status"] == "partially_completed"
    assert result["plan"]["tasks"][0]["status"] == "skipped_due_to_timeout"


def test_autonomous_governor_normal_completion():
    """Verify normal execution runs to completion when within budget."""
    runner = AutonomousRunner(orchestrator=None)

    plan = {
        "tasks": [
            {"id": "t1", "title": "Extract requirements", "instruction": "List requirements"},
            {"id": "t2", "title": "Generate report", "instruction": "Summarize", "depends_on": ["t1"]},
        ]
    }

    result = runner.execute_plan(
        plan=plan,
        initial_goal="Normal mission within budget",
        max_mission_duration_seconds=300.0,
        max_cumulative_retries=6,
    )

    assert result["plan"]["status"] == "completed"
    assert result["plan"]["tasks"][0]["status"] == "completed"
    assert result["plan"]["tasks"][1]["status"] == "completed"
    assert "t1" in runner._artifact_store
    assert "t2" in runner._artifact_store
