"""
Unit tests for Python REPL AST Pre-Scan Security Guard.
Validates blocking of dangerous function calls, attribute access (__subclasses__, __globals__),
blocked module imports, and safe mathematical execution.
"""

from src.tools.python_executor import _validate_python_ast, python_interpreter


def test_ast_guard_blocks_import_of_blocked_modules():
    """Verify that importing blocked modules (os, subprocess, pickle, sys) is rejected at AST level."""
    blocked_snippets = [
        "import os; os.system('whoami')",
        "import subprocess\nsubprocess.run(['dir'])",
        "import pickle\npickle.loads(b'...')",
        "import sys\nsys.exit(0)",
        "from os import path",
        "from subprocess import Popen",
    ]

    for snippet in blocked_snippets:
        ast_err = _validate_python_ast(snippet)
        assert ast_err is not None
        assert "Security Restriction" in ast_err
        # Verify python_interpreter returns the security restriction without executing
        result = python_interpreter.invoke({"code": snippet})
        assert "Security Restriction" in result


def test_ast_guard_blocks_forbidden_function_calls():
    """Verify that direct calls to open, eval, exec, __import__, getattr, globals, vars are blocked."""
    forbidden_calls = [
        "open('C:/Windows/System32/drivers/etc/hosts', 'r')",
        "eval('2 + 2')",
        "exec('x = 5')",
        "getattr(math, 'sin')(1)",
        "globals()",
        "locals()",
        "vars()",
        "__import__('os')",
    ]

    for snippet in forbidden_calls:
        ast_err = _validate_python_ast(snippet)
        assert ast_err is not None
        assert "Security Restriction" in ast_err
        result = python_interpreter.invoke({"code": snippet})
        assert "Security Restriction" in result


def test_ast_guard_blocks_dunder_attribute_traversal():
    """Verify that sandbox escape attempts via __subclasses__, __globals__, __bases__ are blocked."""
    escape_attempts = [
        "().__class__.__bases__[0].__subclasses__()",
        "def foo(): pass\nfoo.__globals__['__builtins__']",
        "object.__subclasses__()",
    ]

    for snippet in escape_attempts:
        ast_err = _validate_python_ast(snippet)
        assert ast_err is not None
        assert "Security Restriction" in ast_err
        result = python_interpreter.invoke({"code": snippet})
        assert "Security Restriction" in result


def test_ast_guard_allows_safe_code_execution():
    """Verify that legitimate mathematical, statistical, and plotting computations pass cleanly."""
    safe_code = """
import numpy as np
import pandas as pd

data = {'Revenue': [100, 200, 300], 'Cost': [60, 110, 150]}
df = pd.DataFrame(data)
df['Profit'] = df['Revenue'] - df['Cost']
print(f"Total Profit: {df['Profit'].sum()}")
"""
    ast_err = _validate_python_ast(safe_code)
    assert ast_err is None

    result = python_interpreter.invoke({"code": safe_code})
    assert "[Output]:" in result
    assert "Total Profit: 280" in result
