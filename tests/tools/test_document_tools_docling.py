"""
Unit and integration tests for Docling document intelligence engine in J.A.R.V.I.S.
Verifies DocumentConverter initialization, structured extraction, table parsing,
and resilient fallback mechanisms.
"""

from unittest.mock import MagicMock, patch

from src.tools.document_tools import (
    convert_document_with_docling,
    extract_text_from_file,
    get_docling_converter,
    is_docling_available,
    process_documents_and_build_vector_store,
)


class MockUploadedFile:
    """Mock file object simulating Streamlit UploadedFile."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content

    def seek(self, pos: int) -> None:
        pass


def test_is_docling_available() -> None:
    """Validate that is_docling_available returns a valid boolean."""
    avail = is_docling_available()
    assert isinstance(avail, bool)


def test_get_docling_converter_singleton() -> None:
    """Validate that get_docling_converter is idempotent and thread-safe."""
    conv1 = get_docling_converter()
    conv2 = get_docling_converter()
    assert conv1 is conv2


def test_convert_document_with_docling_html() -> None:
    """Validate that Docling parses HTML with embedded tables into Markdown."""
    if not is_docling_available():
        return

    html_bytes = b"""
    <html>
    <body>
    <h1>Executive Financial Overview</h1>
    <p>Q3 metrics summary.</p>
    <table>
      <tr><th>Quarter</th><th>Revenue</th></tr>
      <tr><td>Q1</td><td>$10M</td></tr>
      <tr><td>Q2</td><td>$15M</td></tr>
    </table>
    </body>
    </html>
    """
    result = convert_document_with_docling("summary.html", html_bytes)
    assert result is not None
    text, meta = result
    assert "Executive Financial Overview" in text
    assert "Q1" in text
    assert meta["engine"] == "docling"
    assert meta["filename"] == "summary.html"


def test_convert_document_with_docling_markdown() -> None:
    """Validate that Docling converts markdown files preserving structure."""
    if not is_docling_available():
        return

    md_bytes = b"# System Architecture\n\nJ.A.R.V.I.S. multi-agent specification."
    result = convert_document_with_docling("spec.md", md_bytes)
    assert result is not None
    text, meta = result
    assert "System Architecture" in text
    assert meta["engine"] == "docling"


def test_extract_text_from_file_uses_docling() -> None:
    """Verify that extract_text_from_file invokes Docling when available."""
    html_bytes = b"<html><body><h2>Service Reliability</h2><p>99.99% uptime achieved.</p></body></html>"
    mock_file = MockUploadedFile("status.html", html_bytes)

    text, meta = extract_text_from_file(mock_file)
    assert "Service Reliability" in text
    if is_docling_available():
        assert meta.get("engine") == "docling"


def test_extract_text_from_file_docling_fallback_on_error() -> None:
    """Verify that extract_text_from_file gracefully falls back when Docling encounters an error."""
    with patch("src.tools.document_tools.convert_document_with_docling") as mock_convert:
        mock_convert.return_value = None  # Simulate Docling skipping or failing

        raw_md = b"# Native Fallback Title\n\nFallback content line."
        mock_file = MockUploadedFile("fallback.md", raw_md)

        text, meta = extract_text_from_file(mock_file)
        assert "Native Fallback Title" in text
        assert meta["filename"] == "fallback.md"


def test_process_documents_and_build_vector_store_with_docling() -> None:
    """Validate building a FAISS vector store with Docling-extracted structured documents."""
    doc_bytes = b"""
    # Technical Documentation
    J.A.R.V.I.S. provides autonomous goal planning using Kahn's topological sort.
    """
    mock_file = MockUploadedFile("docs.md", doc_bytes)

    with (
        patch("src.tools.document_tools.HuggingFaceEmbeddings"),
        patch("src.tools.document_tools.FAISS.from_texts") as mock_faiss,
    ):
        mock_faiss.return_value = MagicMock()
        vs, summaries, msg = process_documents_and_build_vector_store([mock_file], api_provider="HuggingFace")
        assert vs is not None
        assert len(summaries) == 1
        assert "Successfully processed" in msg
        assert mock_faiss.called
