"""
Tests for ThoughtStepTracer callback handling, tool tracking, and telemetry formatting.
"""

from src.core.orchestrator import ThoughtStepTracer


def test_thought_step_tracer_tool_lifecycle():
    """Verify tracer records tool start, end, and parameter inputs."""
    tracer = ThoughtStepTracer()
    assert tracer.steps == []

    # 1. Simulate on_tool_start
    tracer.on_tool_start({"name": "duckduckgo_search"}, "quantum computing breakthroughs 2026")
    assert len(tracer.steps) == 1
    start_step = tracer.steps[0]
    assert start_step["type"] == "tool_start"
    assert start_step["tool"] == "duckduckgo_search"
    assert "quantum computing" in start_step["input"]
    assert "timestamp" in start_step

    # 2. Simulate on_tool_end
    tracer.on_tool_end("Found 5 articles detailing topological qubit stability.")
    assert len(tracer.steps) == 2
    end_step = tracer.steps[1]
    assert end_step["type"] == "tool_end"
    assert "topological qubit" in end_step["output"]


def test_thought_step_tracer_error_interception():
    """Verify tracer records tool exceptions gracefully."""
    tracer = ThoughtStepTracer()
    tracer.on_tool_start({"name": "python_interpreter"}, "import undefined_library")
    tracer.on_tool_error(ImportError("No module named 'undefined_library'"))

    assert len(tracer.steps) == 2
    err_step = tracer.steps[1]
    assert err_step["type"] == "tool_error"
    assert "undefined_library" in err_step["error"]


def test_thought_step_tracer_output_truncation():
    """Verify that very long tool outputs are truncated to 800 chars in thought steps."""
    tracer = ThoughtStepTracer()
    long_output = "X" * 2000
    tracer.on_tool_end(long_output)

    assert len(tracer.steps) == 1
    assert len(tracer.steps[0]["output"]) <= 805
    assert tracer.steps[0]["output"].endswith("...")
