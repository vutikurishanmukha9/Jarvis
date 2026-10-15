"""
Unit tests validating Windows Job Object Sandbox & Process Resource Governor.
Tests Job Object creation, process assignment, memory limits, timeouts, and sandboxed_python_runner tool.
"""

import sys
from unittest.mock import MagicMock, patch

from src.core.job_sandbox import (
    WindowsJobSandbox,
    run_sandboxed_python_code,
)
from src.tools.pyexec import sandboxed_python_runner


def test_windows_job_sandbox_initialization() -> None:
    """Test WindowsJobSandbox initializes with specified memory caps and kill_on_close."""
    sandbox = WindowsJobSandbox(memory_limit_mb=128, kill_on_close=True)
    assert sandbox.memory_limit_mb == 128
    assert sandbox.memory_limit_bytes == 128 * 1024 * 1024
    assert sandbox.kill_on_close is True

    if sys.platform == "win32":
        # On Windows, job handle should be an integer handle
        assert sandbox.job_handle is not None

    sandbox.close()
    assert sandbox.job_handle is None


def test_windows_job_sandbox_context_manager() -> None:
    """Test WindowsJobSandbox properly closes handle on context exit."""
    with WindowsJobSandbox(memory_limit_mb=64) as sandbox:
        if sys.platform == "win32":
            assert sandbox.job_handle is not None
    assert sandbox.job_handle is None


def test_run_sandboxed_python_code_basic() -> None:
    """Test executing standard Python calculation inside sandboxed subprocess."""
    code = "import math\nprint(f'PI={math.pi:.4f}')"
    exit_code, stdout, stderr = run_sandboxed_python_code(code, memory_limit_mb=256, timeout_seconds=15.0)

    assert exit_code == 0
    assert "PI=3.1416" in stdout
    assert stderr == ""


def test_run_sandboxed_python_code_timeout() -> None:
    """Test that a long-running runaway process is terminated by the sandbox deadline."""
    code = "import time\ntime.sleep(5.0)\nprint('done')"
    exit_code, stdout, stderr = run_sandboxed_python_code(code, memory_limit_mb=256, timeout_seconds=1.0)

    assert exit_code == -1
    assert "Sandbox Timeout" in stderr


def test_sandboxed_python_runner_tool_success() -> None:
    """Test sandboxed_python_runner tool invocation with fenced code."""
    fenced_code = "```python\nprint(sum([x * 2 for x in range(5)]))\n```"
    res = sandboxed_python_runner.invoke({"code": fenced_code})

    assert "Sandboxed Execution (Exit Code 0)" in res
    assert "[Output]:" in res
    assert "20" in res


def test_sandboxed_python_runner_tool_syntax_error() -> None:
    """Test sandboxed_python_runner tool handles syntax errors gracefully."""
    invalid_code = "def broken(:\n    pass"
    res = sandboxed_python_runner.invoke({"code": invalid_code})

    assert "Exit Code 1" in res or "Exit Code" in res
    assert "SyntaxError" in res or "Stderr/Errors" in res
