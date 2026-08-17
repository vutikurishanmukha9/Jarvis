"""
Unit tests for document hash caching and deduplication in document tools.
"""

import io

from src.tools.document_tools import get_files_hash


class InMemoryFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data
        self._io = io.BytesIO(data)

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._io.read()

    def seek(self, offset: int) -> None:
        self._io.seek(offset)


def test_hash_different_for_different_files() -> None:
    """Ensure different files produce distinct SHA-256 digests."""
    f1 = InMemoryFile("doc1.pdf", b"Content 1")
    f2 = InMemoryFile("doc2.pdf", b"Content 2")

    h1 = get_files_hash([f1])
    h2 = get_files_hash([f2])

    assert h1 != h2
    assert len(h1) == 64
    assert len(h2) == 64


def test_hash_empty_list() -> None:
    """Ensure empty file list produces consistent SHA-256 string."""
    h = get_files_hash([])
    assert len(h) == 64
