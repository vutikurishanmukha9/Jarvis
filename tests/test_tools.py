import pytest
import io
import pandas as pd
import matplotlib.figure
from pathlib import Path

from src.tools.document_tools import extract_text_from_file, get_files_hash
from src.tools.python_executor import python_interpreter, get_and_clear_figure_buffer
from src.tools.web_tools import get_web_tools, _validate_url

class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
    def getvalue(self) -> bytes:
        return self._data
    def seek(self, pos: int):
        pass

def test_universal_document_parser():
    """Verify parsing of text, CSV, and tabular data."""
    # 1. Test Text file
    txt_bytes = "Hello JARVIS Universal Document Parser".encode("utf-8")
    txt_file = MockUploadedFile("sample.txt", txt_bytes)
    text, meta = extract_text_from_file(txt_file)
    assert "Hello JARVIS Universal Document Parser" in text
    assert meta["type"] == ".txt"

    # 2. Test CSV file
    csv_bytes = "Name,Version\nJarvis,2.0\nUltron,1.0".encode("utf-8")
    csv_file = MockUploadedFile("data.csv", csv_bytes)
    text_csv, meta_csv = extract_text_from_file(csv_file)
    assert "Jarvis" in text_csv
    assert meta_csv["type"] == ".csv"
    assert meta_csv["size"] > 0

def test_file_hash_change_detection():
    """Verify that file hash changes when file content is modified."""
    f1 = MockUploadedFile("doc1.txt", b"Initial text")
    hash_v1 = get_files_hash([f1])

    f2 = MockUploadedFile("doc1.txt", b"Modified text for hash change")
    hash_v2 = get_files_hash([f2])

    assert hash_v1 != hash_v2

def test_python_interpreter_tool():
    """Verify controlled Python interpreter executes and captures output and figures."""
    # Test stdout capture
    code = "x = [1, 2, 3]\nprint(f'Sum: {sum(x)}')"
    output = python_interpreter.invoke({"code": code})
    assert "Sum: 6" in output

    # Test Matplotlib figure capture
    plot_code = "import matplotlib.pyplot as plt\nplt.figure()\nplt.plot([1, 2], [3, 4])\nplt.title('Test Plot')"
    output_plot = python_interpreter.invoke({"code": plot_code})
    figs = get_and_clear_figure_buffer()
    assert len(figs) > 0
    assert isinstance(figs[0], matplotlib.figure.Figure)

def test_web_tools():
    """Verify web search, Wikipedia, and page reader tools are registered."""
    tools = get_web_tools()
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert "duckduckgo_search" in tool_names
    assert "wikipedia_lookup" in tool_names
    assert "read_webpage_content" in tool_names


# ==================== NEW SECURITY TESTS ====================

def test_python_executor_blocks_dangerous_imports():
    """Verify that os, subprocess, shutil, and socket imports are blocked in the controlled environment."""
    dangerous_modules = ["os", "subprocess", "shutil", "socket"]

    for module_name in dangerous_modules:
        code = f"import {module_name}\nprint({module_name}.__name__)"
        output = python_interpreter.invoke({"code": code})
        assert "Security Restriction" in output or "blocked" in output.lower(), (
            f"Import of '{module_name}' was not blocked. Output: {output}"
        )

def test_python_executor_blocks_nested_imports():
    """Verify that nested/indirect dangerous imports are also blocked."""
    # Test __import__ bypass attempt
    code = "m = __import__('os')\nprint(m.getcwd())"
    output = python_interpreter.invoke({"code": code})
    # __import__ is removed from builtins, so this should fail
    assert "Error" in output or "blocked" in output.lower() or "Security" in output, (
        f"__import__('os') bypass was not blocked. Output: {output}"
    )

def test_python_executor_output_size_limit():
    """Verify that output is truncated when exceeding the size limit."""
    # Generate output larger than 50KB
    code = "print('A' * 100000)"
    output = python_interpreter.invoke({"code": code})
    assert "truncated" in output.lower(), f"Large output was not truncated. Length: {len(output)}"

def test_web_tools_blocks_private_ips():
    """Verify that URLs resolving to private/internal IPs are blocked."""
    private_urls = [
        "http://127.0.0.1",
        "http://127.0.0.1:8080/secret",
        "http://10.0.0.1/admin",
        "http://192.168.1.1/config",
        "http://[::1]/internal",
    ]

    for url in private_urls:
        error = _validate_url(url)
        assert error, f"Private IP URL was not blocked: {url}"
        assert "private" in error.lower() or "internal" in error.lower() or "blocked" in error.lower(), (
            f"Error message for '{url}' didn't indicate IP blocking: {error}"
        )

def test_web_tools_rejects_non_http_schemes():
    """Verify that file://, ftp://, javascript:, and data: schemes are rejected."""
    bad_urls = [
        "file:///etc/passwd",
        "ftp://evil.com/malware",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "gopher://evil.com",
    ]

    for url in bad_urls:
        error = _validate_url(url)
        assert error, f"Non-HTTP scheme was not rejected: {url}"
        assert "scheme" in error.lower() or "blocked" in error.lower(), (
            f"Error message for '{url}' didn't indicate scheme rejection: {error}"
        )

def test_web_tools_allows_valid_urls():
    """Verify that valid HTTP/HTTPS URLs pass validation (DNS may fail for nonexistent domains)."""
    # Test scheme validation passes for valid schemes
    valid_url = "https://www.example.com"
    error = _validate_url(valid_url)
    # This should either pass (empty string) or fail only on DNS, not on scheme/IP
    if error:
        assert "scheme" not in error.lower(), f"Valid HTTPS URL was rejected for scheme: {error}"
