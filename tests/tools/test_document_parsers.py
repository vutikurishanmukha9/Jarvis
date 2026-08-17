"""
Tests for Universal Document Parser across TXT, CSV, JSON, Python, and unsupported formats.
"""

import json

from src.tools.document_tools import extract_text_from_file


def test_document_parser_txt(sample_text_file):
    """Verify plaintext extraction and metadata dictionary."""
    text, meta = extract_text_from_file(sample_text_file)
    assert "J.A.R.V.I.S. Core Intelligence Engine" in text
    assert meta["type"] == ".txt"
    assert meta["filename"] == "test_doc.txt"
    assert meta["size"] > 0


def test_document_parser_csv(sample_csv_file):
    """Verify CSV tabular parsing, row counts, and summary representation."""
    text, meta = extract_text_from_file(sample_csv_file)
    assert "CSV Dataset" in text or "Alice" in text
    assert "Alice" in text
    assert "Bob" in text
    assert meta["type"] == ".csv"
    assert meta["rows"] == 2
    assert len(meta["columns"]) == 3


def test_document_parser_json():
    """Verify JSON structure parsing and pretty-printed string extraction."""
    from tests.conftest import MockUploadedFile

    json_data = json.dumps({"project": "Jarvis", "version": "2.0", "modules": ["core", "vision", "career"]})
    json_file = MockUploadedFile("config.json", json_data.encode("utf-8"))

    text, meta = extract_text_from_file(json_file)
    assert "JSON Document" in text
    assert "Jarvis" in text
    assert "modules" in text
    assert meta["type"] == ".json"


def test_document_parser_python_script():
    """Verify Python code file ingestion and comment/function retention."""
    from tests.conftest import MockUploadedFile

    py_code = "def calculate_orbit():\n    return 42 * 3.14\n"
    py_file = MockUploadedFile("orbit.py", py_code.encode("utf-8"))

    text, meta = extract_text_from_file(py_file)
    assert "calculate_orbit" in text
    assert meta["type"] == ".py"


def test_document_parser_markdown():
    """Verify Markdown file text extraction."""
    from tests.conftest import MockUploadedFile

    md_content = "# Overview\n\nThis is a test overview document."
    md_file = MockUploadedFile("notes.md", md_content.encode("utf-8"))

    text, meta = extract_text_from_file(md_file)
    assert "Overview" in text
    assert meta["type"] == ".md"


def test_document_parser_unsupported_extension():
    """Verify unsupported binary formats return a descriptive notice."""
    from tests.conftest import MockUploadedFile

    bin_file = MockUploadedFile("archive.zip", b"PK\x03\x04randomdata")

    text, meta = extract_text_from_file(bin_file)
    assert meta["type"] == ".zip"
