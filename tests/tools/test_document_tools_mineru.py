"""
Unit and integration tests for MinerU document intelligence engine in B.H.A.I.R.A.V.A.
Verifies LaTeX formula extraction, table recognition, multi-tier fallback mechanisms,
and LangChain tool integration.
"""

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

from src.tools.document_tools import (
    convert_document_with_mineru,
    create_mineru_document_tool,
    extract_text_from_file,
    is_mineru_available,
    parse_document_with_mineru,
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


def test_is_mineru_available() -> None:
    """Validate that is_mineru_available returns a valid boolean."""
    avail = is_mineru_available()
    assert isinstance(avail, bool)


def test_convert_document_with_mineru_unsupported_suffix() -> None:
    """Validate that convert_document_with_mineru rejects unsupported extensions."""
    res = convert_document_with_mineru("archive.zip", b"dummy zip bytes")
    assert res is None


def test_convert_document_with_mineru_with_mock_runner() -> None:
    """Validate that convert_document_with_mineru processes formulas and tables via runner."""
    sample_md = (
        "# Quantum Physics Paper\n\n"
        "Energy-momentum relation is given by:\n\n"
        "$$\\mathcal{H}\\psi = E\\psi$$\n\n"
        "And mass equivalence $E = mc^2$ applies."
    )
    sample_meta = {
        "filename": "quantum.pdf",
        "type": ".pdf",
        "size": 1024,
        "engine": "mineru",
        "pages": 1,
        "formulas": 2,
        "tables": 0,
    }

    def mock_runner(file_name: str, content_bytes: bytes, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        return sample_md, sample_meta

    with patch.object(convert_document_with_mineru, "_runner", mock_runner, create=True):
        res = convert_document_with_mineru("quantum.pdf", b"%PDF-1.4 mock content")
        assert res is not None
        md, meta = res
        assert "\\mathcal{H}\\psi = E\\psi" in md
        assert "$E = mc^2$" in md
        assert meta["engine"] == "mineru"
        assert meta["formulas"] == 2
        assert meta["pages"] == 1


def test_convert_document_with_mineru_exception_resilience() -> None:
    """Validate that convert_document_with_mineru catches exceptions and returns None."""

    def broken_runner(file_name: str, content_bytes: bytes, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        raise RuntimeError("Model checkpoint corrupted or unavailable")

    with patch.object(convert_document_with_mineru, "_runner", broken_runner, create=True):
        res = convert_document_with_mineru("broken.pdf", b"corrupted bytes")
        assert res is None


def test_extract_text_from_file_routes_to_mineru() -> None:
    """Validate that extract_text_from_file leverages MinerU when prefer_engine='mineru'."""
    expected_md = "## Advanced Linear Algebra\n$$\\det(A) = \\prod_{i} \\lambda_i$$"
    expected_meta = {
        "filename": "algebra.pdf",
        "type": ".pdf",
        "size": 512,
        "engine": "mineru",
        "pages": 1,
        "formulas": 1,
        "tables": 0,
    }

    def mock_runner(file_name: str, content_bytes: bytes, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        return expected_md, expected_meta

    with patch.object(convert_document_with_mineru, "_runner", mock_runner, create=True):
        mock_file = MockUploadedFile("algebra.pdf", b"%PDF-1.4 test")
        text, meta = extract_text_from_file(mock_file, prefer_engine="mineru")
        assert "\\det(A)" in text
        assert meta["engine"] == "mineru"
        assert meta["formulas"] == 1


def test_extract_text_from_file_falls_back_when_mineru_returns_none() -> None:
    """Validate that extract_text_from_file falls back to native/Docling when MinerU returns None."""
    with patch("src.tools.document_tools.convert_document_with_mineru", return_value=None):
        raw_text = "Standard plain text document content without neural formatting."
        mock_file = MockUploadedFile("fallback.txt", raw_text.encode("utf-8"))
        text, meta = extract_text_from_file(mock_file)
        assert "Standard plain text document content" in text
        assert meta["filename"] == "fallback.txt"


def test_parse_document_with_mineru_success() -> None:
    """Validate parse_document_with_mineru structured output on success."""
    sample_md = "# Machine Learning Foundations\nCross entropy: $L = -\\sum y \\log(\\hat{y})$"
    sample_meta = {
        "filename": "ml.pdf",
        "type": ".pdf",
        "size": 256,
        "engine": "mineru",
        "pages": 1,
        "formulas": 1,
        "tables": 0,
    }

    def mock_runner(file_name: str, content_bytes: bytes, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        return sample_md, sample_meta

    with patch.object(convert_document_with_mineru, "_runner", mock_runner, create=True):
        mock_file = MockUploadedFile("ml.pdf", b"%PDF-mock")
        res = parse_document_with_mineru(mock_file)
        assert res["status"] == "success"
        assert res["engine"] == "mineru"
        assert "Cross entropy" in res["text"]
        assert res["metadata"]["formulas"] == 1


def test_parse_document_with_mineru_fallback_status() -> None:
    """Validate parse_document_with_mineru status is 'fallback' when MinerU returns None."""
    with patch("src.tools.document_tools.convert_document_with_mineru", return_value=None):
        mock_file = MockUploadedFile("notes.txt", b"Notes on graph theory.")
        res = parse_document_with_mineru(mock_file)
        assert res["status"] == "fallback"
        assert "Notes on graph theory" in res["text"]


def test_create_mineru_document_tool(tmp_path: Path) -> None:
    """Validate that create_mineru_document_tool generates an invokable LangChain tool."""
    tool = create_mineru_document_tool()
    assert tool.name == "parse_scientific_document"

    test_file = tmp_path / "paper.pdf"
    test_file.write_bytes(b"%PDF-1.4 test math paper")

    sample_md = "# General Relativity\nEinstein field equations: $$G_{\\mu\\nu} + \\Lambda g_{\\mu\\nu} = \\kappa T_{\\mu\\nu}$$"
    sample_meta = {
        "filename": "paper.pdf",
        "type": ".pdf",
        "size": test_file.stat().st_size,
        "engine": "mineru",
        "pages": 1,
        "formulas": 1,
        "tables": 0,
    }

    def mock_runner(file_name: str, content_bytes: bytes, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        return sample_md, sample_meta

    with patch.object(convert_document_with_mineru, "_runner", mock_runner, create=True):
        output = tool.invoke(str(test_file))
        assert "=== MinerU Document Analysis: paper.pdf" in output
        assert "Einstein field equations" in output
        assert "G_{\\mu\\nu}" in output


def test_create_mineru_document_tool_missing_file() -> None:
    """Validate tool behavior when file does not exist."""
    tool = create_mineru_document_tool()
    output = tool.invoke("non_existent_file_path_12345.pdf")
    assert "Error: File not found" in output


def test_mineru_concurrency_thread_safety() -> None:
    """Validate thread safety under concurrent invocation of convert_document_with_mineru."""
    results = []

    def mock_runner(file_name: str, content_bytes: bytes, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        return f"# Parsed {file_name}", {"filename": file_name, "engine": "mineru"}

    with patch.object(convert_document_with_mineru, "_runner", mock_runner, create=True):

        def worker(idx: int) -> None:
            res = convert_document_with_mineru(f"doc_{idx}.pdf", b"data")
            if res:
                results.append(res[0])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
