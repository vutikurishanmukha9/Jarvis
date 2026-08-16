import pytest
import io
import pandas as pd
import matplotlib.figure
from pathlib import Path

from src.tools.document_tools import extract_text_from_file, get_files_hash
from src.tools.python_executor import python_interpreter, get_and_clear_figure_buffer
from src.tools.web_tools import get_web_tools

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
    """Verify sandboxed Python interpreter executes and captures output and figures."""
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
