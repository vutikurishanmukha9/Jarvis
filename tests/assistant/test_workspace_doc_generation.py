"""
Unit tests for workspace document, markdown, and spreadsheet generation.
"""

from pathlib import Path

from src.assistant.workspace_tools import (
    generate_excel_spreadsheet,
    generate_word_document,
    read_workspace_file,
    write_workspace_file,
)
from src.config import WORKSPACE_DIR


def test_write_and_read_workspace_file(tmp_path: Path) -> None:
    """Test writing and reading a file within the workspace sandbox."""
    res_write = write_workspace_file.invoke(
        {"filename": "test_report.md", "content": "# Test Heading\nThis is test content."}
    )
    assert "Successfully wrote" in res_write or "test_report.md" in res_write

    content = read_workspace_file.invoke({"filename": "test_report.md"})
    assert "# Test Heading" in content
    assert "This is test content." in content


def test_generate_excel_spreadsheet_from_json() -> None:
    """Test generating a multi-row .xlsx spreadsheet."""
    json_data = '[{"Quarter": "Q1", "Revenue": 150000}, {"Quarter": "Q2", "Revenue": 185000}]'
    res = generate_excel_spreadsheet.invoke(
        {"filename": "finance.xlsx", "json_table_data": json_data, "sheet_name": "Q1_Q2_Revenue"}
    )
    assert "Successfully generated Excel" in res or "finance.xlsx" in res

    target = WORKSPACE_DIR / "finance.xlsx"
    assert target.exists()


def test_generate_word_document_from_markdown() -> None:
    """Test generating a .docx Word document from structured markdown."""
    md_content = "# Title\n## Section 1\n- Item 1\n- Item 2\n1. Numbered Item"
    res = generate_word_document.invoke(
        {"filename": "summary.docx", "title": "Summary Report", "markdown_content": md_content}
    )
    assert "Successfully generated Word" in res or "summary.docx" in res

    target = WORKSPACE_DIR / "summary.docx"
    assert target.exists()
