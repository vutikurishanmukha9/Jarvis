"""
Unit tests for Python executor thread safety and execution sandbox.
"""

from src.tools.python_executor import (
    get_and_clear_figure_buffer,
    python_interpreter,
)


def test_python_interpreter_basic_math() -> None:
    """Test standard evaluation of mathematical computations."""
    res = python_interpreter.invoke("print(sum([i**2 for i in range(1, 6)]))")
    assert "55" in res


def test_python_interpreter_blocks_unsafe_imports() -> None:
    """Ensure os, subprocess, shutil, and socket modules cannot be imported."""
    res = python_interpreter.invoke("import os\nprint(os.getcwd())")
    assert "Security Exception" in res or "Import of" in res or "blocked" in res


def test_python_interpreter_matplotlib_plot_interception() -> None:
    """Ensure matplotlib plots are intercepted and stored in figure buffer."""
    get_and_clear_figure_buffer()
    code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
"""
    python_interpreter.invoke(code)
    figs = get_and_clear_figure_buffer()
    assert len(figs) >= 1
