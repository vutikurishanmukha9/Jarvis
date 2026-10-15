"""
Unit tests validating Browser Permission Policy Gate against permission.site Web API benchmark.
Tests zero-trust policy decisions for 25+ browser permissions across ALLOW, MOCK,
REQUIRE_HUMAN_APPROVAL, and DENY classifications.
"""

from src.core.permission_gate import (
    PermissionPolicy,
    get_permission_gate,
)
from src.tools.browser import check_browser_permission, get_browser_tools


def test_permission_gate_allow_policies() -> None:
    """Verify safe, non-destructive permissions are automatically allowed."""
    gate = get_permission_gate()
    gate.reset_policies()

    for perm in [
        "notifications",
        "clipboard",
        "clipboard-read",
        "clipboard-write",
        "read-text",
        "write-text",
        "copy",
        "fullscreen",
        "eme",
    ]:
        res = gate.evaluate_permission(perm)
        assert res.decision == PermissionPolicy.ALLOW, f"Expected ALLOW for {perm}, got {res.decision}"
        assert not res.requires_user_prompt
        assert res.mock_payload is None


def test_permission_gate_mock_policies() -> None:
    """Verify privacy-sensitive telemetry permissions return synthetic mock data without leaking host coordinates."""
    gate = get_permission_gate()
    gate.reset_policies()

    # Location / Geolocation
    loc_res = gate.evaluate_permission("location")
    assert loc_res.decision == PermissionPolicy.MOCK
    assert loc_res.mock_payload is not None
    assert "latitude" in loc_res.mock_payload
    assert "longitude" in loc_res.mock_payload

    geo_res = gate.evaluate_permission("geolocation")
    assert geo_res.decision == PermissionPolicy.MOCK
    assert "latitude" in geo_res.mock_payload

    approx_res = gate.evaluate_permission("approximate-location")
    assert approx_res.decision == PermissionPolicy.MOCK

    # Idle Detection
    idle_res = gate.evaluate_permission("idle-detection")
    assert idle_res.decision == PermissionPolicy.MOCK
    assert idle_res.mock_payload.get("user") == "active"

    # Orientation & Motion
    orient_res = gate.evaluate_permission("orientation")
    assert orient_res.decision == PermissionPolicy.MOCK
    assert "alpha" in orient_res.mock_payload

    motion_res = gate.evaluate_permission("motion")
    assert motion_res.decision == PermissionPolicy.MOCK
    assert "acceleration" in motion_res.mock_payload


def test_permission_gate_human_approval_policies() -> None:
    """Verify high-risk audiovisual sensors require explicit operator confirmation."""
    gate = get_permission_gate()
    gate.reset_policies()

    sensitive_perms = [
        "camera",
        "microphone",
        "camera+microphone",
        "pan-tilt-zoom",
        "pan-tilt-zoom+microphone",
        "screenshare",
        "display-capture",
        "webauthn-attestation",
    ]
    for perm in sensitive_perms:
        res = gate.evaluate_permission(perm)
        assert res.decision == PermissionPolicy.REQUIRE_HUMAN_APPROVAL, f"Expected REQUIRE_HUMAN_APPROVAL for {perm}"
        assert res.requires_user_prompt is True


def test_permission_gate_prohibited_deny_policies() -> None:
    """Verify critical attack surface Web APIs are strictly DENIED."""
    gate = get_permission_gate()
    gate.reset_policies()

    prohibited_perms = [
        "usb",
        "web-usb",
        "serial",
        "web-serial",
        "hid",
        "web-hid",
        "bluetooth",
        "web-bluetooth",
        "nfc",
        "show-open-file-picker",
        "show-directory-picker",
        "file-system-access",
        "keyboardlock",
        "download",
        "auto-download",
        "protocol-handler",
        "persistent-storage",
        "keygen",
        "vr",
        "ar",
    ]
    for perm in prohibited_perms:
        res = gate.evaluate_permission(perm)
        assert res.decision == PermissionPolicy.DENY, f"Expected DENY for {perm}, got {res.decision}"
        assert not res.requires_user_prompt


def test_permission_gate_unknown_defaults_to_deny() -> None:
    """Verify unclassified or unknown permissions are blocked by zero-trust fallback."""
    gate = get_permission_gate()
    res = gate.evaluate_permission("exotic-unknown-hardware-api")
    assert res.decision == PermissionPolicy.DENY
    assert "zero-trust default" in res.reason


def test_permission_gate_custom_override() -> None:
    """Test setting and resetting custom runtime policies."""
    gate = get_permission_gate()
    gate.reset_policies()

    # Initially camera requires human approval
    assert gate.evaluate_permission("camera").decision == PermissionPolicy.REQUIRE_HUMAN_APPROVAL

    # Override to DENY in strict airgapped mode
    gate.set_custom_policy("camera", PermissionPolicy.DENY, reason="Airgapped mission mode")
    res = gate.evaluate_permission("camera")
    assert res.decision == PermissionPolicy.DENY
    assert "Airgapped mission mode" in res.reason

    # Reset
    gate.reset_policies()
    assert gate.evaluate_permission("camera").decision == PermissionPolicy.REQUIRE_HUMAN_APPROVAL


def test_check_browser_permission_tool() -> None:
    """Test check_browser_permission LangChain tool output formatting."""
    res_usb = check_browser_permission.invoke({"permission_name": "usb"})
    assert "DENY" in res_usb
    assert "WebUSB access strictly denied" in res_usb

    res_loc = check_browser_permission.invoke({"permission_name": "location"})
    assert "MOCK" in res_loc
    assert "Synthetic Mock Data" in res_loc

    res_cam = check_browser_permission.invoke({"permission_name": "camera"})
    assert "REQUIRE_HUMAN_APPROVAL" in res_cam
    assert "Human-In-The-Loop: Required" in res_cam


def test_check_browser_permission_in_tools_suite() -> None:
    """Verify check_browser_permission is registered in get_browser_tools."""
    tools = get_browser_tools()
    tool_names = [t.name for t in tools]
    assert "check_browser_permission" in tool_names
