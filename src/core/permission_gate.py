"""
Browser Permission Policy Gate for J.A.R.V.I.S.
Inspired by permission.site Web API security benchmark.

Enforces zero-trust automated browser navigation policies across 25+ browser permissions:
- ALLOW: Safe, read-only or local non-destructive features (Notifications, Clipboard read/write).
- MOCK: Privacy-sensitive features that return synthetic payloads to avoid fingerprinting / location leak (Geolocation, Motion, Idle).
- REQUIRE_HUMAN_APPROVAL: High-risk hardware audiovisual sensors (Camera, Microphone, Screen Sharing, WebAuthn).
- DENY: Critical attack surface APIs strictly prohibited for automated agents (USB, Serial, HID, Bluetooth, NFC, File System Access).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PermissionPolicy(str, Enum):
    """Zero-Trust Policy Verdict for Web API Permissions."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    MOCK = "MOCK"
    REQUIRE_HUMAN_APPROVAL = "REQUIRE_HUMAN_APPROVAL"


@dataclass
class PermissionEvaluationResult:
    """Standardized evaluation result for a browser permission request."""

    permission: str
    decision: PermissionPolicy
    reason: str
    mock_payload: Optional[Dict[str, Any]] = None
    requires_user_prompt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission": self.permission,
            "decision": self.decision.value,
            "reason": self.reason,
            "mock_payload": self.mock_payload,
            "requires_user_prompt": self.requires_user_prompt,
        }


# Canonical permission mappings based on permission.site specification
_DEFAULT_POLICY_MAP: Dict[str, Tuple[PermissionPolicy, str, Optional[Dict[str, Any]]]] = {
    # 1. Safe / Autonomous Permissions (ALLOW)
    "notifications": (
        PermissionPolicy.ALLOW,
        "System notifications permitted for asynchronous task updates.",
        None,
    ),
    "clipboard": (
        PermissionPolicy.ALLOW,
        "Clipboard read/write permitted for agent text extraction and copying.",
        None,
    ),
    "clipboard-read": (
        PermissionPolicy.ALLOW,
        "Clipboard reading permitted for autonomous data extraction.",
        None,
    ),
    "clipboard-write": (
        PermissionPolicy.ALLOW,
        "Clipboard writing permitted for synthesized deliverables.",
        None,
    ),
    "read-text": (
        PermissionPolicy.ALLOW,
        "Async clipboard readText permitted for agent inspection.",
        None,
    ),
    "write-text": (
        PermissionPolicy.ALLOW,
        "Async clipboard writeText permitted for agent clipboard population.",
        None,
    ),
    "copy": (
        PermissionPolicy.ALLOW,
        "Standard DOM copy event permitted.",
        None,
    ),
    "fullscreen": (
        PermissionPolicy.ALLOW,
        "Display mode change permitted for page inspection.",
        None,
    ),
    "pointerlock": (
        PermissionPolicy.ALLOW,
        "Pointer lock permitted for navigation simulation.",
        None,
    ),
    "eme": (
        PermissionPolicy.ALLOW,
        "Encrypted Media Extensions permitted for media playback testing.",
        None,
    ),

    # 2. Synthetic Privacy Defenses (MOCK)
    "location": (
        PermissionPolicy.MOCK,
        "Physical GPS coordinates masked. Synthetic coordinates returned to prevent geolocation leakage.",
        {"latitude": 51.5074, "longitude": -0.1278, "accuracy": 15.0},
    ),
    "geolocation": (
        PermissionPolicy.MOCK,
        "Physical GPS coordinates masked. Synthetic coordinates returned to prevent geolocation leakage.",
        {"latitude": 51.5074, "longitude": -0.1278, "accuracy": 15.0},
    ),
    "approximate-location": (
        PermissionPolicy.MOCK,
        "Approximate location mocked to prevent IP/location correlation.",
        {"latitude": 51.5074, "longitude": -0.1278, "accuracy": 1000.0},
    ),
    "idle-detection": (
        PermissionPolicy.MOCK,
        "Idle state mocked to maintain active agent execution.",
        {"user": "active", "screen": "unlocked"},
    ),
    "orientation": (
        PermissionPolicy.MOCK,
        "Device orientation mocked to prevent physical device telemetry fingerprinting.",
        {"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
    ),
    "device-orientation": (
        PermissionPolicy.MOCK,
        "Device orientation mocked to prevent physical device telemetry fingerprinting.",
        {"alpha": 0.0, "beta": 0.0, "gamma": 0.0},
    ),
    "motion": (
        PermissionPolicy.MOCK,
        "Device motion sensors mocked to prevent physical vibration fingerprinting.",
        {"acceleration": {"x": 0.0, "y": 0.0, "z": 9.8}, "rotationRate": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}},
    ),
    "device-motion": (
        PermissionPolicy.MOCK,
        "Device motion sensors mocked to prevent physical vibration fingerprinting.",
        {"acceleration": {"x": 0.0, "y": 0.0, "z": 9.8}, "rotationRate": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}},
    ),

    # 3. High-Risk Hardware Sensors (REQUIRE_HUMAN_APPROVAL)
    "camera": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Hardware camera optical sensor access requires explicit operator confirmation.",
        None,
    ),
    "microphone": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Hardware microphone audio capture requires explicit operator confirmation.",
        None,
    ),
    "camera+microphone": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Simultaneous camera and microphone access requires explicit operator confirmation.",
        None,
    ),
    "pan-tilt-zoom": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Motorized camera controls require operator confirmation.",
        None,
    ),
    "pan-tilt-zoom+microphone": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Motorized camera and microphone access require operator confirmation.",
        None,
    ),
    "screenshare": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Screen capture / display surface sharing exposes sensitive desktop data. Operator confirmation required.",
        None,
    ),
    "display-capture": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Display capture exposes sensitive desktop data. Operator confirmation required.",
        None,
    ),
    "webauthn-attestation": (
        PermissionPolicy.REQUIRE_HUMAN_APPROVAL,
        "Hardware security token authentication requires explicit human user interaction.",
        None,
    ),

    # 4. Strictly Prohibited Peripheral & Local APIs (DENY)
    "usb": (
        PermissionPolicy.DENY,
        "WebUSB access strictly denied to prevent hardware peripheral compromise.",
        None,
    ),
    "web-usb": (
        PermissionPolicy.DENY,
        "WebUSB access strictly denied to prevent hardware peripheral compromise.",
        None,
    ),
    "serial": (
        PermissionPolicy.DENY,
        "WebSerial access strictly denied to prevent direct COM/UART port manipulation.",
        None,
    ),
    "web-serial": (
        PermissionPolicy.DENY,
        "WebSerial access strictly denied to prevent direct COM/UART port manipulation.",
        None,
    ),
    "hid": (
        PermissionPolicy.DENY,
        "WebHID access strictly denied to prevent keystroke and peripheral tampering.",
        None,
    ),
    "web-hid": (
        PermissionPolicy.DENY,
        "WebHID access strictly denied to prevent keystroke and peripheral tampering.",
        None,
    ),
    "bluetooth": (
        PermissionPolicy.DENY,
        "WebBluetooth access strictly denied to isolate local radio interfaces.",
        None,
    ),
    "web-bluetooth": (
        PermissionPolicy.DENY,
        "WebBluetooth access strictly denied to isolate local radio interfaces.",
        None,
    ),
    "nfc": (
        PermissionPolicy.DENY,
        "WebNFC access strictly denied to protect RFID / contactless hardware.",
        None,
    ),
    "show-open-file-picker": (
        PermissionPolicy.DENY,
        "Direct local file picker access denied to enforce sandbox containment.",
        None,
    ),
    "show-directory-picker": (
        PermissionPolicy.DENY,
        "Direct directory traversal picker denied to enforce sandbox containment.",
        None,
    ),
    "file-system-access": (
        PermissionPolicy.DENY,
        "FileSystemAccess API strictly prohibited for automated web sessions.",
        None,
    ),
    "keyboardlock": (
        PermissionPolicy.DENY,
        "Keyboard lock strictly prohibited to prevent agent window trapping.",
        None,
    ),
    "download": (
        PermissionPolicy.DENY,
        "Unprompted drive-by automatic file downloads blocked by security policy.",
        None,
    ),
    "auto-download": (
        PermissionPolicy.DENY,
        "Unprompted drive-by automatic file downloads blocked by security policy.",
        None,
    ),
    "protocol-handler": (
        PermissionPolicy.DENY,
        "Custom protocol handler registration blocked to prevent OS URL scheme hijacks.",
        None,
    ),
    "persistent-storage": (
        PermissionPolicy.DENY,
        "Persistent disk storage allocation denied to prevent disk exhaustion.",
        None,
    ),
    "keygen": (
        PermissionPolicy.DENY,
        "Deprecated HTML keygen tag blocked to prevent RSA key generation exploits.",
        None,
    ),
    "vr": (
        PermissionPolicy.DENY,
        "Virtual Reality hardware access denied.",
        None,
    ),
    "ar": (
        PermissionPolicy.DENY,
        "Augmented Reality hardware access denied.",
        None,
    ),
}


class BrowserPermissionGate:
    """
    Evaluates and enforces browser permission policies for automated web sessions.
    Protects user hardware, disk, and identity from untrusted web pages.
    """

    def __init__(self) -> None:
        self._custom_policies: Dict[str, Tuple[PermissionPolicy, str, Optional[Dict[str, Any]]]] = {}

    def _normalize_name(self, permission: str) -> str:
        """Normalize permission name to canonical lower-case hyphenated token."""
        return permission.strip().lower().replace("_", "-").replace(" ", "-")

    def evaluate_permission(self, permission: str, origin: str = "") -> PermissionEvaluationResult:
        """
        Evaluate a browser permission request against zero-trust policy matrix.

        Args:
            permission: The permission name (e.g., 'camera', 'usb', 'location', 'clipboard').
            origin: Optional web origin requesting the permission.

        Returns:
            PermissionEvaluationResult with decision, reason, and mock payload if applicable.
        """
        norm = self._normalize_name(permission)

        # 1. Check custom override policies
        if norm in self._custom_policies:
            policy, reason, mock_payload = self._custom_policies[norm]
            return PermissionEvaluationResult(
                permission=norm,
                decision=policy,
                reason=reason,
                mock_payload=mock_payload,
                requires_user_prompt=(policy == PermissionPolicy.REQUIRE_HUMAN_APPROVAL),
            )

        # 2. Check default baseline policies
        if norm in _DEFAULT_POLICY_MAP:
            policy, reason, mock_payload = _DEFAULT_POLICY_MAP[norm]
            return PermissionEvaluationResult(
                permission=norm,
                decision=policy,
                reason=reason,
                mock_payload=mock_payload,
                requires_user_prompt=(policy == PermissionPolicy.REQUIRE_HUMAN_APPROVAL),
            )

        # 3. Default fallback: Unknown permissions are DENIED for security
        logger.warning("Unknown browser permission '%s' requested by origin '%s' - DENYING.", permission, origin)
        return PermissionEvaluationResult(
            permission=norm,
            decision=PermissionPolicy.DENY,
            reason=f"Unknown or unclassified browser permission '{permission}' denied by zero-trust default.",
            mock_payload=None,
            requires_user_prompt=False,
        )

    def set_custom_policy(
        self,
        permission: str,
        policy: PermissionPolicy,
        reason: str = "Custom operator override",
        mock_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a runtime custom policy override for a specific permission."""
        norm = self._normalize_name(permission)
        self._custom_policies[norm] = (policy, reason, mock_payload)

    def reset_policies(self) -> None:
        """Reset all runtime custom policies back to default baseline."""
        self._custom_policies.clear()


# Global active permission gate singleton
_ACTIVE_PERMISSION_GATE = BrowserPermissionGate()


def get_permission_gate() -> BrowserPermissionGate:
    """Retrieve active singleton BrowserPermissionGate."""
    return _ACTIVE_PERMISSION_GATE
