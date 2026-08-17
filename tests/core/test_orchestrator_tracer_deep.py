"""
Unit tests for ThoughtStepTracer in JarvisOrchestrator.
"""

from src.core.orchestrator import ThoughtStepTracer


def test_thought_step_tracer_lifecycle() -> None:
    """Verify tool start, end with truncation, and error tracking."""
    tracer = ThoughtStepTracer()
    assert tracer.steps == []

    # 1. Start Tool
    tracer.on_tool_start({"name": "web_search"}, "quantum computing")
    assert len(tracer.steps) == 1
    assert tracer.steps[0]["type"] == "tool_start"
    assert tracer.steps[0]["tool"] == "web_search"
    assert tracer.steps[0]["input"] == "quantum computing"

    # 2. End Tool with Long Output Truncation (> 800 chars)
    long_output = "A" * 1000
    tracer.on_tool_end(long_output)
    assert len(tracer.steps) == 2
    assert tracer.steps[1]["type"] == "tool_end"
    assert len(tracer.steps[1]["output"]) == 803  # 800 chars + "..."
    assert tracer.steps[1]["output"].endswith("...")

    # 3. Tool Error
    tracer.on_tool_error(RuntimeError("API Connection Failed"))
    assert len(tracer.steps) == 3
    assert tracer.steps[2]["type"] == "tool_error"
    assert "API Connection Failed" in tracer.steps[2]["error"]
