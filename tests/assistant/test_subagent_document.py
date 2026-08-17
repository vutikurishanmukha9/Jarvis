"""
Unit tests for Document Ingestion & RAG Sub-Agent tool.
"""

import io
from unittest.mock import MagicMock, patch

from src.tools.document_tools import (
    create_document_retriever_tool,
    extract_text_from_file,
    get_files_hash,
    process_documents_and_build_vector_store,
)


class MockUploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content
        self._buffer = io.BytesIO(content)

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._buffer.read()

    def seek(self, pos: int) -> None:
        self._buffer.seek(pos)


def test_extract_text_from_txt_file() -> None:
    """Test extracting plaintext content from mock file."""
    mock_file = MockUploadedFile("sample.txt", b"Hello, this is a knowledge base document.")
    text, meta = extract_text_from_file(mock_file)
    assert "Hello, this is a knowledge base document." in text
    assert meta["filename"] == "sample.txt"


def test_extract_text_from_json_file() -> None:
    """Test extracting structured JSON content."""
    json_bytes = b'{"project": "Jarvis", "version": "2.0", "status": "active"}'
    mock_file = MockUploadedFile("data.json", json_bytes)
    text, meta = extract_text_from_file(mock_file)
    assert "Jarvis" in text
    assert "active" in text


def test_get_files_hash_consistency() -> None:
    """Test SHA-256 hash consistency for identical file payloads."""
    file1 = MockUploadedFile("a.txt", b"abc")
    file2 = MockUploadedFile("a.txt", b"abc")

    hash1 = get_files_hash([file1])
    hash2 = get_files_hash([file2])

    assert len(hash1) == 64
    assert hash1 == hash2


def test_process_documents_and_build_vector_store() -> None:
    """Test vector store creation and retriever tool packaging."""
    mock_file = MockUploadedFile("guide.txt", b"Quantum computing uses qubits instead of bits.")

    with patch("src.tools.document_tools.FAISS") as mock_faiss, \
         patch("src.tools.document_tools.HuggingFaceEmbeddings"):
        mock_vectorstore = MagicMock()
        mock_faiss.from_texts.return_value = mock_vectorstore

        vs, summaries, msg = process_documents_and_build_vector_store(
            uploaded_files=[mock_file],
            api_provider="OpenRouter",
        )
        assert vs is not None
        assert len(summaries) == 1
        assert "file(s)" in msg

        tool = create_document_retriever_tool(vs)
        assert tool is not None
        assert tool.name == "document_search"
