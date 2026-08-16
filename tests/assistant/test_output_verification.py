"""
Tests for semantic output verification logic in AutonomousRunner.
"""

import pytest
from src.assistant.autonomous_runner import AutonomousRunner

def test_output_verification_no_orchestrator():
    """Verify output verification passes gracefully when no orchestrator is attached."""
    runner = AutonomousRunner(orchestrator=None)
    res = runner._verify_output("Any result", "instruction", "expected")
    assert res["passed"] is True

def test_output_verification_rejects_empty_output():
    """Verify verification rejects empty or sub-10 character outputs."""
    runner = AutonomousRunner(orchestrator=None)
    # Even without LLM, structural check catches empty/too short output
    runner.orchestrator = True  # Mock presence
    res_empty = runner._verify_output("", "instruction", "deliverable")
    assert res_empty["passed"] is False
    assert "empty" in res_empty["reason"].lower() or "short" in res_empty["reason"].lower()

    res_short = runner._verify_output("ok", "instruction", "deliverable")
    assert res_short["passed"] is False

def test_output_verification_detects_error_markers():
    """Verify verification flags common error strings."""
    runner = AutonomousRunner(orchestrator=None)
    runner.orchestrator = True

    markers = [
        "An error occurred during agent processing: Rate limit exceeded",
        "Python Execution Error: ZeroDivisionError: division by zero",
        "Security Restriction: Import of 'os' is blocked",
        "Execution Timeout: Code did not complete within 30 seconds",
        "Failed to retrieve web page from 'https://example.com'"
    ]

    for err in markers:
        res = runner._verify_output(err, "Do task", "Deliverable")
        assert res["passed"] is False, f"Marker '{err}' was not flagged as failure"
        assert "error marker" in res["reason"].lower()

def test_output_verification_pass_case():
    """Verify clean outputs without error markers are sent for evaluation."""
    class MockOrchestrator:
        def run(self, user_input, chat_history):
            return {"output": "PASS\nThe deliverable is complete and meets all requirements."}

    runner = AutonomousRunner(orchestrator=MockOrchestrator())
    result = runner._verify_output(
        "Successfully compiled competitive benchmark table with 10 companies.",
        "Compile table",
        "Table of competitors"
    )
    assert result["passed"] is True
    assert "complete" in result["reason"].lower()

def test_output_verification_fail_case():
    """Verify model FAIL evaluations trigger failure return."""
    class MockOrchestratorFail:
        def run(self, user_input, chat_history):
            return {"output": "FAIL\nThe output is missing the required financial metrics column."}

    runner = AutonomousRunner(orchestrator=MockOrchestratorFail())
    result = runner._verify_output(
        "Found 2 companies but didn't extract revenue.",
        "Compile revenue table",
        "Table with revenue"
    )
    assert result["passed"] is False
    assert "financial metrics" in result["reason"]
