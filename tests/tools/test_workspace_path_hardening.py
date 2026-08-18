"""
Unit tests for Workspace Path Jail and Sandbox Hardening (VULN-04).
Validates rejection of directory traversal attempts, absolute paths, Windows drive letters,
UNC paths, null bytes, and path length limits.
"""

import pytest

from src.assistant.workspace_tools import _resolve_workspace_path, read_workspace_file, write_workspace_file
from src.config import WORKSPACE_DIR


def test_resolve_safe_relative_path():
    """Verify that legitimate workspace-relative paths resolve safely inside WORKSPACE_DIR."""
    resolved = _resolve_workspace_path("reports/summary.md")
    assert resolved.is_relative_to(WORKSPACE_DIR.resolve())
    assert resolved.name == "summary.md"


def test_rejects_parent_directory_traversal():
    """Verify that ../ attempts escaping the sandbox raise ValueError."""
    traversal_paths = [
        "../secret.txt",
        "../../etc/passwd",
        "nested/../../outside.env",
        "....//....//windows/win.ini",
        "../../../../windows/win.ini",
        ".../attack.py",
    ]
    for p in traversal_paths:
        with pytest.raises(ValueError, match="Path must remain inside the workspace."):
            _resolve_workspace_path(p)


def test_rejects_windows_drive_letters():
    """Verify that Windows drive letters (C:, D:) are strictly blocked."""
    drive_paths = [
        "C:/Windows/System32/drivers/etc/hosts",
        "D:\\Secrets\\passwords.txt",
        "c:test.txt",
    ]
    for p in drive_paths:
        with pytest.raises(ValueError, match="Drive letters and UNC network paths are not permitted"):
            _resolve_workspace_path(p)


def test_rejects_unc_network_paths():
    """Verify that UNC network shares (\\\\server\\share) are strictly blocked."""
    unc_paths = [
        "\\\\192.168.1.1\\share\\data.txt",
        "//attacker.com/payload.py",
    ]
    for p in unc_paths:
        with pytest.raises(ValueError, match="Drive letters and UNC network paths are not permitted"):
            _resolve_workspace_path(p)


def test_rejects_null_byte_injection():
    """Verify that null byte injection attempts are blocked."""
    null_byte_path = "legit.txt\x00.evil.exe"
    with pytest.raises(ValueError, match="Null byte injection detected"):
        _resolve_workspace_path(null_byte_path)


def test_rejects_empty_or_excessive_path_length():
    """Verify that empty paths and oversized paths are rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_workspace_path("   ")

    oversized_path = "a" * 260 + ".txt"
    with pytest.raises(ValueError, match="exceeds maximum permitted length"):
        _resolve_workspace_path(oversized_path)


def test_write_and_read_workspace_file_within_sandbox(tmp_path):
    """Verify end-to-end safe write and read operations within workspace."""
    test_filename = "security_audit_test.txt"
    content = "J.A.R.V.I.S. Secure Sandbox Deliverable"

    write_result = write_workspace_file.invoke({"filename": test_filename, "content": content})
    assert "Successfully created workspace file" in write_result

    read_result = read_workspace_file.invoke({"filename": test_filename})
    assert content in read_result
