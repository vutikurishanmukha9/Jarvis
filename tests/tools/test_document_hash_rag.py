"""
Tests for document hashing, cache invalidation, and content change sensitivity.
"""

import pytest
from tests.conftest import MockUploadedFile
from src.tools.document_tools import get_files_hash

def test_file_hash_deterministic():
    """Verify identical files always produce identical MD5 hashes."""
    f1 = MockUploadedFile("doc.txt", b"Exact content")
    f2 = MockUploadedFile("doc.txt", b"Exact content")

    hash1 = get_files_hash([f1])
    hash2 = get_files_hash([f2])
    assert hash1 == hash2
    assert len(hash1) == 32  # Standard MD5 hex digest length

def test_file_hash_content_mutation_sensitivity():
    """Verify changing 1 character in content modifies the resulting hash."""
    f_orig = MockUploadedFile("doc.txt", b"Version 1.0.0")
    f_mut = MockUploadedFile("doc.txt", b"Version 1.0.1")

    hash_orig = get_files_hash([f_orig])
    hash_mut = get_files_hash([f_mut])
    assert hash_orig != hash_mut

def test_file_hash_name_mutation_sensitivity():
    """Verify changing filename modifies the hash even if contents match."""
    f_a = MockUploadedFile("alpha.txt", b"Common body text")
    f_b = MockUploadedFile("beta.txt", b"Common body text")

    assert get_files_hash([f_a]) != get_files_hash([f_b])

def test_file_hash_empty_list():
    """Verify empty list of files produces an MD5 hash gracefully."""
    empty_hash = get_files_hash([])
    assert len(empty_hash) == 32
