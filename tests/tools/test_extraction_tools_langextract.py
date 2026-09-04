"""
Unit and integration tests for Google LangExtract grounded entity extraction in J.A.R.V.I.S.
Verifies grounded entity extraction, character span normalization, HTML visualizer generation,
and safe workspace sandbox persistence.
"""

from unittest.mock import MagicMock, patch

from src.tools.document_tools import extract_entities_from_document
from src.tools.extraction_tools import (
    extract_grounded_entities,
    extract_structured_entities_tool,
    get_extraction_tools,
    is_langextract_available,
    save_grounded_visualization,
    visualize_extractions_tool,
)


class MockUploadedFile:
    """Mock file simulating Streamlit UploadedFile."""

    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content

    def seek(self, pos: int) -> None:
        pass


def test_is_langextract_available() -> None:
    """Validate that is_langextract_available detects the library correctly."""
    available = is_langextract_available()
    assert isinstance(available, bool)
    assert available is True


def test_get_extraction_tools_registration() -> None:
    """Verify tool collection registration and schema contracts."""
    tools = get_extraction_tools()
    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "extract_structured_entities_tool" in tool_names
    assert "visualize_extractions_tool" in tool_names


def test_extract_grounded_entities_empty_text() -> None:
    """Verify empty or whitespace text returns an empty extraction list immediately."""
    res = extract_grounded_entities(text="   ", prompt_description="Extract names")
    assert res["success"] is True
    assert res["count"] == 0
    assert res["extractions"] == []


def test_extract_grounded_entities_mocked_success() -> None:
    """Verify grounded extraction pipeline normalizes character offsets and attributes."""
    mock_interval = MagicMock()
    mock_interval.start_pos = 12
    mock_interval.end_pos = 28

    mock_extraction = MagicMock()
    mock_extraction.extraction_class = "technology"
    mock_extraction.extraction_text = "Google LangExtract"
    mock_extraction.char_interval = mock_interval
    mock_extraction.attributes = {"domain": "NLP", "verified": True}
    mock_extraction.alignment_status = "match_exact"

    mock_doc = MagicMock()
    mock_doc.extractions = [mock_extraction]

    with patch("langextract.extract", return_value=mock_doc):
        text = "Engineered with Google LangExtract for high recall."
        res = extract_grounded_entities(
            text=text,
            prompt_description="Extract technology entities",
            model_id="gpt-4o-mini",
        )

        assert res["success"] is True
        assert res["count"] == 1
        entity = res["extractions"][0]
        assert entity["extraction_class"] == "technology"
        assert entity["extraction_text"] == "Google LangExtract"
        assert entity["char_interval"]["start_pos"] == 12
        assert entity["char_interval"]["end_pos"] == 28
        assert entity["attributes"]["domain"] == "NLP"


def test_extract_grounded_entities_exception_handling() -> None:
    """Verify that exceptions raised by LangExtract are caught and returned safely."""
    with patch("langextract.extract", side_effect=RuntimeError("Provider API timeout")):
        res = extract_grounded_entities(
            text="Valid document content",
            prompt_description="Extract skills",
        )
        assert res["success"] is False
        assert "Provider API timeout" in res["error"]
        assert res["extractions"] == []


def test_save_grounded_visualization_in_workspace(tmp_path) -> None:
    """Verify HTML visualization generation and persistence inside the workspace."""
    mock_html = "<html><body><h1>Entity Grounding Report</h1></body></html>"

    with patch("langextract.visualize", return_value=mock_html):
        mock_doc = MagicMock()
        filename = save_grounded_visualization(mock_doc, output_filename="test_grounding.html")
        assert filename == "test_grounding.html"


def test_extract_structured_entities_tool_invocation() -> None:
    """Validate invoking extract_structured_entities_tool returns valid JSON."""
    mock_interval = MagicMock()
    mock_interval.start_pos = 0
    mock_interval.end_pos = 9

    mock_extraction = MagicMock()
    mock_extraction.extraction_class = "framework"
    mock_extraction.extraction_text = "J.A.R.V.I.S."
    mock_extraction.char_interval = mock_interval
    mock_extraction.attributes = {"type": "agentic"}
    mock_extraction.alignment_status = "match_exact"

    mock_doc = MagicMock()
    mock_doc.extractions = [mock_extraction]

    with patch("langextract.extract", return_value=mock_doc):
        output = extract_structured_entities_tool.invoke(
            {
                "text": "J.A.R.V.I.S. is an autonomous intelligence platform.",
                "extraction_goal": "Extract framework names",
            }
        )
        assert "J.A.R.V.I.S." in output
        assert "success" in output


def test_visualize_extractions_tool_invocation() -> None:
    """Validate visualize_extractions_tool returns workspace location status."""
    output = visualize_extractions_tool.invoke({"filename": "non_existent_file.html"})
    assert "has not been generated yet" in output or "workspace" in output


def test_extract_entities_from_document_integration() -> None:
    """Validate end-to-end extraction connecting Docling and LangExtract."""
    mock_doc_content = b"# Architecture\n\nJ.A.R.V.I.S. integrates Docling and LangExtract."
    mock_file = MockUploadedFile("system.md", mock_doc_content)

    mock_interval = MagicMock()
    mock_interval.start_pos = 16
    mock_interval.end_pos = 23

    mock_extraction = MagicMock()
    mock_extraction.extraction_class = "module"
    mock_extraction.extraction_text = "Docling"
    mock_extraction.char_interval = mock_interval
    mock_extraction.attributes = {"role": "document_converter"}
    mock_extraction.alignment_status = "match_exact"

    mock_doc = MagicMock()
    mock_doc.extractions = [mock_extraction]

    with patch("langextract.extract", return_value=mock_doc):
        res = extract_entities_from_document(mock_file, prompt_description="Extract module names")
        assert res["success"] is True
        assert res["count"] == 1
        assert "document_metadata" in res
        assert res["document_metadata"]["filename"] == "system.md"
