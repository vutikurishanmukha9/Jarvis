"""
Tests for workspace file operations, path confinement, Excel/Word generation, and directory listing.
"""

import json

import pytest

from src.assistant.workspace_tools import (
    _resolve_workspace_path,
    generate_excel_spreadsheet,
    generate_word_document,
    get_workspace_tools,
    list_workspace_files,
    read_workspace_file,
    save_personal_memory,
    write_workspace_file,
)
from src.config import WORKSPACE_DIR


def test_resolve_workspace_path_confinement():
    """Verify that path traversal attempts are confined strictly within WORKSPACE_DIR."""
    # 1. Normal relative path
    p1 = _resolve_workspace_path("test.txt")
    assert p1.is_relative_to(WORKSPACE_DIR.resolve()) or str(p1).startswith(str(WORKSPACE_DIR.resolve()))

    # 2. Traversal and absolute paths must be rejected rather than silently rewritten.
    with pytest.raises(ValueError):
        _resolve_workspace_path("../workspace_evil/secret.txt")
    with pytest.raises(ValueError):
        _resolve_workspace_path("/etc/passwd")


def test_write_and_read_workspace_file():
    """Verify creating and reading text and script files in workspace."""
    filename = "test_script.py"
    code = "import math\nprint(math.sqrt(144))\n"

    write_res = write_workspace_file.invoke({"filename": filename, "content": code})
    assert "Successfully created workspace file" in write_res

    read_res = read_workspace_file.invoke({"filename": filename})
    assert "math.sqrt(144)" in read_res


def test_read_nonexistent_file():
    """Verify reading a non-existent file returns a friendly error message."""
    res = read_workspace_file.invoke({"filename": "ghost_file_9999.txt"})
    assert "does not exist" in res.lower()


def test_list_workspace_files():
    """Verify listing workspace files reports file names and sizes."""
    write_workspace_file.invoke({"filename": "file_alpha.txt", "content": "alpha content"})
    listing = list_workspace_files.invoke({})

    assert "Workspace Files" in listing
    assert "file_alpha.txt" in listing


def test_generate_excel_spreadsheet():
    """Verify generating .xlsx files from JSON table data."""
    table_data = json.dumps(
        [{"Quarter": "Q1", "Revenue": 150000, "Margin": 0.22}, {"Quarter": "Q2", "Revenue": 180000, "Margin": 0.25}]
    )
    res = generate_excel_spreadsheet.invoke(
        {"filename": "quarterly_financials.xlsx", "json_table_data": table_data, "sheet_name": "Q1_Q2_Summary"}
    )
    assert "Successfully generated Excel" in res
    assert (WORKSPACE_DIR / "quarterly_financials.xlsx").exists()


def test_generate_excel_invalid_json():
    """Verify error handling when malformed JSON is passed to Excel generator."""
    res = generate_excel_spreadsheet.invoke(
        {"filename": "bad_table.xlsx", "json_table_data": "not valid json {", "sheet_name": "Sheet1"}
    )
    assert "Error" in res or "Invalid JSON" in res


def test_generate_word_document():
    """Verify generating Word (.docx) documents with headings and lists."""
    md_text = (
        "# Executive Mission Briefing\n\n"
        "## Key Findings\n"
        "- Finding 1: System throughput increased by 30%.\n"
        "- Finding 2: Latency dropped below 50ms.\n\n"
        "### Conclusion\n"
        "The model is ready for production deployment."
    )
    res = generate_word_document.invoke(
        {"filename": "executive_briefing.docx", "title": "Mission Briefing", "markdown_content": md_text}
    )
    assert "Successfully generated Word Document" in res or "Saved document as Markdown" in res


def test_save_personal_memory_tool():
    """Verify tool wrapper for saving long-term memory facts."""
    res = save_personal_memory.invoke({"fact": "Target launch date is October 15th.", "category": "milestones"})
    assert "persistent memory" in res.lower()


def test_workspace_tools_suite():
    """Verify all 6 workspace tools are exposed in the module registry."""
    tools = get_workspace_tools()
    assert len(tools) == 6
    names = [t.name for t in tools]
    assert "write_workspace_file" in names
    assert "read_workspace_file" in names
    assert "list_workspace_files" in names
    assert "generate_excel_spreadsheet" in names
    assert "generate_word_document" in names
    assert "save_personal_memory" in names
