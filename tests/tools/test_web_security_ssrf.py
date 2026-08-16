"""
Tests for Web URL validation and SSRF defensive guards: schemes, private IPs, loopbacks.
"""

import pytest
from src.tools.web_tools import _validate_url

def test_web_url_validation_allows_valid_https():
    """Verify standard public web domains pass scheme and structure validation."""
    err = _validate_url("https://www.example.com")
    # Should either pass (empty error) or fail only on external DNS lookup
    if err:
        assert "scheme" not in err.lower()

def test_web_url_validation_blocks_bad_schemes():
    """Verify non-HTTP schemes (file://, ftp://, gopher://, javascript:, data:) are rejected."""
    bad_schemes = [
        "file:///etc/shadow",
        "ftp://internal.vault/data",
        "gopher://ancient.service",
        "javascript:alert(document.cookie)",
        "data:text/plain;base64,SGVsbG8="
    ]
    for url in bad_schemes:
        err = _validate_url(url)
        assert err, f"Scheme in '{url}' was not rejected"
        assert "scheme" in err.lower() or "blocked" in err.lower()

def test_web_url_validation_blocks_ipv4_loopbacks():
    """Verify loopback addresses (127.0.0.1, 127.0.1.1) are blocked."""
    loopbacks = [
        "http://127.0.0.1:8080/admin",
        "http://127.0.1.1/secret"
    ]
    for url in loopbacks:
        err = _validate_url(url)
        assert err, f"Loopback '{url}' was not blocked"
        assert "private" in err.lower() or "internal" in err.lower() or "blocked" in err.lower()

def test_web_url_validation_blocks_rfc1918_private_ranges():
    """Verify RFC1918 private IP addresses (10.x, 172.16.x, 192.168.x) are blocked."""
    private_ips = [
        "http://10.0.0.1/router",
        "http://172.16.0.5/database",
        "http://192.168.1.1/gateway",
    ]
    for url in private_ips:
        err = _validate_url(url)
        assert err, f"Private IP '{url}' was not blocked"
        assert "private" in err.lower() or "internal" in err.lower()

def test_web_url_validation_blocks_cloud_metadata_and_ipv6_loopback():
    """Verify cloud metadata endpoint (169.254.169.254) and IPv6 loopback (::1) are blocked."""
    special_ips = [
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:8000/internal"
    ]
    for url in special_ips:
        err = _validate_url(url)
        assert err, f"Special IP '{url}' was not blocked"
        assert "private" in err.lower() or "internal" in err.lower() or "blocked" in err.lower()

def test_web_url_validation_malformed_url():
    """Verify completely malformed URL strings are rejected."""
    assert _validate_url("not_a_valid_url_at_all") != ""
