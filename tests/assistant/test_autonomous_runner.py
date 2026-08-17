"""
Tests for AutonomousRunner execution loop, artifact store scoping, and mission summary generation.
"""

from src.assistant.autonomous_runner import AutonomousRunner


def test_autonomous_runner_simulated_execution():
    """Verify runner completes tasks in simulated mode without an orchestrator."""
    runner = AutonomousRunner(orchestrator=None)
    plan = {
        "goal": "Build Financial Model",
        "status": "planned",
        "tasks": [
            {"id": "task_1", "title": "Scrape Revenue Metrics", "instruction": "Fetch data", "depends_on": []},
            {"id": "task_2", "title": "Compile Excel Model", "instruction": "Write .xlsx", "depends_on": ["task_1"]},
        ],
    }
    result = runner.execute_plan(plan, "Build Financial Model")

    assert result["plan"]["status"] == "completed"
    assert result["plan"]["tasks"][0]["status"] == "completed"
    assert result["plan"]["tasks"][1]["status"] == "completed"
    assert "Simulated autonomous completion" in result["plan"]["tasks"][0]["result"]


def test_autonomous_runner_artifact_store_population():
    """Verify output deliverables are registered in _artifact_store by task_id."""
    runner = AutonomousRunner(orchestrator=None)
    plan = {
        "tasks": [
            {"id": "task_alpha", "title": "Data Gathering", "instruction": "Get data", "depends_on": []},
            {"id": "task_beta", "title": "Report Writing", "instruction": "Write doc", "depends_on": ["task_alpha"]},
        ]
    }
    runner.execute_plan(plan, "Test Goal")

    assert "task_alpha" in runner._artifact_store
    assert "task_beta" in runner._artifact_store


def test_autonomous_runner_dependency_context_scoping():
    """Verify task context contains only outputs from declared depends_on tasks."""
    runner = AutonomousRunner(orchestrator=None)
    runner._artifact_store = {
        "task_1": "Dataset Alpha: 100 rows",
        "task_2": "Competitor Beta: $50M revenue",
        "task_3": "Unrelated Gamma Data",
    }

    # Task depends only on task_1 and task_2
    task = {"id": "task_4", "depends_on": ["task_1", "task_2"]}
    context = runner._build_dependency_context(task)

    assert "Dataset Alpha" in context
    assert "Competitor Beta" in context
    assert "Unrelated Gamma Data" not in context


def test_autonomous_runner_empty_dependency_context():
    """Verify tasks with no dependencies produce an empty context string."""
    runner = AutonomousRunner(orchestrator=None)
    runner._artifact_store = {"task_1": "Some data"}

    task = {"id": "task_2", "depends_on": []}
    context = runner._build_dependency_context(task)
    assert context == ""


def test_autonomous_runner_missing_dependency_key_resilience():
    """Verify missing dependency keys in artifact store do not raise exceptions."""
    runner = AutonomousRunner(orchestrator=None)
    runner._artifact_store = {}

    task = {"id": "task_2", "depends_on": ["nonexistent_task_id"]}
    context = runner._build_dependency_context(task)
    assert context == ""


def test_autonomous_runner_step_callback_telemetry():
    """Verify step callbacks receive progress events during execution."""
    events = []

    def record_step(p, tid, msg):
        events.append((tid, msg))

    runner = AutonomousRunner(orchestrator=None, step_callback=record_step)
    plan = {
        "tasks": [
            {"id": "t1", "title": "Step One", "instruction": "Do step 1", "depends_on": []},
            {"id": "t2", "title": "Step Two", "instruction": "Do step 2", "depends_on": ["t1"]},
        ]
    }
    runner.execute_plan(plan, "Telemetry Test")

    assert len(events) >= 4  # Start and finish for each of the 2 tasks
    assert any("Step One" in e[1] for e in events)
    assert any("COMPLETED" in e[1] for e in events)


def test_autonomous_runner_mission_summary_format():
    """Verify the generated final executive summary structure."""
    runner = AutonomousRunner(orchestrator=None)
    plan = {
        "tasks": [
            {"id": "t1", "title": "Extract Data", "status": "completed", "result": "Extracted 50 items."},
            {"id": "t2", "title": "Build Model", "status": "completed", "result": "Model saved."},
        ]
    }
    summary = runner._generate_mission_summary("Build Pipeline", plan)
    assert "# Autonomous Mission Complete" in summary
    assert "**Assigned Goal**: Build Pipeline" in summary
    assert "Extracted 50 items." in summary
    assert "Model saved." in summary
