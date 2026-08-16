"""
Tests for Python Executor Security: import blocklist, restricted builtins, execution timeout, and output limits.
"""

import pytest
from src.tools.python_executor import python_interpreter, BLOCKED_MODULES

def test_python_executor_blocked_modules_list():
    """Verify dangerous operating system and network modules are registered in blocklist."""
    assert "os" in BLOCKED_MODULES
    assert "subprocess" in BLOCKED_MODULES
    assert "shutil" in BLOCKED_MODULES
    assert "socket" in BLOCKED_MODULES
    assert "ctypes" in BLOCKED_MODULES
    assert "importlib" in BLOCKED_MODULES

def test_python_executor_blocks_direct_os_import():
    """Verify attempting to import os raises a security restriction."""
    code = "import os\nprint(os.getcwd())"
    output = python_interpreter.invoke({"code": code})
    assert "Security Restriction" in output or "blocked" in output.lower()

def test_python_executor_blocks_direct_subprocess_import():
    """Verify attempting to import subprocess is blocked."""
    code = "import subprocess\nsubprocess.run(['dir'])"
    output = python_interpreter.invoke({"code": code})
    assert "Security Restriction" in output or "blocked" in output.lower()

def test_python_executor_blocks_socket_and_shutil():
    """Verify socket and shutil imports are blocked."""
    code1 = "import socket\ns = socket.socket()"
    out1 = python_interpreter.invoke({"code": code1})
    assert "Security Restriction" in out1 or "blocked" in out1.lower()

    code2 = "import shutil\nshutil.rmtree('test')"
    out2 = python_interpreter.invoke({"code": code2})
    assert "Security Restriction" in out2 or "blocked" in out2.lower()

def test_python_executor_blocks_builtins_bypass():
    """Verify dangerous builtin functions like open and eval are blocked or restricted."""
    code_eval = "eval('2 + 2')"
    out_eval = python_interpreter.invoke({"code": code_eval})
    assert "Error" in out_eval or "Security" in out_eval or "not defined" in out_eval

    code_open = "f = open('test.txt', 'w')"
    out_open = python_interpreter.invoke({"code": code_open})
    assert "Error" in out_open or "Security" in out_open or "not defined" in out_open

def test_python_executor_output_truncation_cap():
    """Verify that very large stdout strings (>50KB) are truncated to prevent memory exhaustion."""
    code = "print('Z' * 80000)"
    output = python_interpreter.invoke({"code": code})
    assert "truncated" in output.lower()
    assert len(output) < 70000
