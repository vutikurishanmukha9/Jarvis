"""
Tests for Controlled Python Executor: math, pandas/numpy, matplotlib figure capture, and syntax handling.
"""

import matplotlib.figure

from src.tools.python_executor import get_and_clear_figure_buffer, python_interpreter


def test_python_executor_arithmetic():
    """Verify stdout print capture for mathematical computations."""
    code = "a = 25\nb = 4\nprint(f'Result: {a * b}')"
    output = python_interpreter.invoke({"code": code})
    assert "Result: 100" in output


def test_python_executor_numpy_and_pandas():
    """Verify pre-imported NumPy and Pandas capabilities."""
    code = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'scores': [88, 92, 95, 78]})\n"
        "print(f'Mean Score: {df[\"scores\"].mean():.2f}')\n"
    )
    output = python_interpreter.invoke({"code": code})
    assert "Mean Score: 88.25" in output


def test_python_executor_matplotlib_figure_capture():
    """Verify generated matplotlib figures are captured into _FIGURE_BUFFER."""
    code = (
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(6, 4))\n"
        "plt.plot([1, 2, 3], [10, 20, 30], label='Growth')\n"
        "plt.title('Test Chart')\n"
    )
    output = python_interpreter.invoke({"code": code})
    figs = get_and_clear_figure_buffer()
    assert len(figs) > 0
    assert isinstance(figs[0], matplotlib.figure.Figure)
    assert "Chart Generated" in output


def test_python_executor_markdown_codeblock_cleaning():
    """Verify code wrapped in markdown ```python fences is stripped and executed cleanly."""
    fenced_code = "```python\nx = 42\nprint(f'Value: {x}')\n```"
    output = python_interpreter.invoke({"code": fenced_code})
    assert "Value: 42" in output


def test_python_executor_syntax_error_handling():
    """Verify syntax errors return clean error messages rather than crashing."""
    broken_code = "def broken_func(\nprint('missing paren')"
    output = python_interpreter.invoke({"code": broken_code})
    assert "Python Execution Error" in output or "SyntaxError" in output
