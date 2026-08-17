"""
Unit tests for AutonomousRunner multi-step mission execution.
"""

from unittest.mock import MagicMock

from src.assistant.autonomous_runner import AutonomousRunner


def test_build_dependency_context_retrieval() -> None:
    """Test retrieving prior task deliverables from the artifact store."""
    runner = AutonomousRunner()
    runner._artifact_store["task_1"] = "Revenue grew by 24% YoY in 2025."
    runner._artifact_store["task_2"] = "Customer churn decreased to 1.2%."

    task_3 = {
        "id": "task_3",
        "title": "Synthesize Annual Report",
        "depends_on": ["task_1", "task_2"],
    }

    dep_context = runner._build_dependency_context(task_3)
    assert "Revenue grew by 24% YoY" in dep_context
    assert "Customer churn decreased" in dep_context


def test_build_dependency_context_empty() -> None:
    """Test task with no dependencies returns empty context string."""
    runner = AutonomousRunner()
    task_1 = {"id": "task_1", "title": "Fetch Data", "depends_on": []}
    assert runner._build_dependency_context(task_1) == ""


def test_autonomous_runner_execution_flow() -> None:
    """Test running a multi-step plan with mocked orchestrator."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.run.side_effect = [
        # Task execution return
        {
            "output": "Audit logs verified: 0 anomalies detected.",
            "steps": [],
            "figures": [],
            "annotated_images": [],
        },
        # Verification return (PASS)
        {
            "output": "PASS\nOutput fulfills requirements.",
            "steps": [],
            "figures": [],
            "annotated_images": [],
        },
        # Final synthesis return
        {
            "output": "Executive Summary: Security audit concluded cleanly.",
            "steps": [],
            "figures": [],
            "annotated_images": [],
        },
    ]

    runner = AutonomousRunner(orchestrator=mock_orchestrator)
    plan = {
        "goal_summary": "Perform audit",
        "tasks": [
            {
                "id": "t1",
                "title": "Audit Logs",
                "instruction": "Check security logs",
                "expected_deliverable": "Log summary",
                "depends_on": [],
            }
        ],
    }

    result = runner.execute_plan(plan=plan, initial_goal="Audit security logs")
    assert result["plan"]["status"] == "completed"
    assert len(result["plan"]["tasks"]) == 1
    assert "t1" in runner._artifact_store
