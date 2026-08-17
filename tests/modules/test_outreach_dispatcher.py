"""
Tests for EmailDispatcher: validation, simulated campaign delivery, and Excel audit export.
"""

import pytest
from pathlib import Path
from src.modules.outreach.email_dispatcher import EmailDispatcher
from src.config import WORKSPACE_DIR

def test_email_dispatcher_empty_recipients():
    """Verify dispatcher returns error status on empty recipient list."""
    res = EmailDispatcher.dispatch(
        subject_template="Hello",
        body_template="World",
        recipients=[],
        simulated=True
    )
    assert res["status"] == "error"
    assert res["total"] == 0

def test_email_dispatcher_invalid_email_handling():
    """Verify invalid emails (no @) are flagged as failed in audit logs."""
    recipients = [
        {"email": "not_an_email_address", "firstName": "Bad", "company": "Co"},
        {"email": "valid@example.com", "firstName": "Good", "company": "Co"}
    ]
    res = EmailDispatcher.dispatch(
        subject_template="Hello {firstName}",
        body_template="Connecting with {company}",
        recipients=recipients,
        simulated=True
    )
    assert res["status"] == "success"
    assert res["total"] == 2
    assert res["sent"] == 1
    assert res["failed"] == 1

def test_email_dispatcher_simulated_audit_file_creation():
    """Verify dispatch writes an audit spreadsheet into WORKSPACE_DIR."""
    recipients = [
        {"email": "lead1@tech.io", "firstName": "Alice", "company": "TechCorp"},
        {"email": "lead2@tech.io", "firstName": "Bob", "company": "CloudInc"}
    ]
    res = EmailDispatcher.dispatch(
        subject_template="Question for {firstName}",
        body_template="Hi {firstName} at {company}",
        recipients=recipients,
        simulated=True
    )
    assert res["sent"] == 2
    assert res["audit_file"] is not None
    assert Path(res["audit_file"]).exists()


def test_email_dispatcher_campaign_ids_are_unique():
    recipients = [{"email": "lead@tech.io", "firstName": "Ada", "company": "Tech"}]
    first = EmailDispatcher.dispatch("Hello", "Hi", recipients, simulated=True, delay_seconds=0)
    second = EmailDispatcher.dispatch("Hello", "Hi", recipients, simulated=True, delay_seconds=0)
    assert first["campaign_id"] != second["campaign_id"]

def test_email_dispatcher_live_smtp_failure_without_credentials():
    """Verify live SMTP mode fails gracefully when host is unreachable."""
    recipients = [{"email": "target@example.com", "firstName": "Test", "company": "TestOrg"}]
    res = EmailDispatcher.dispatch(
        subject_template="Subject",
        body_template="Body",
        recipients=recipients,
        smtp_config={"host": "127.0.0.1", "port": 9999, "user": "", "password": ""},
        simulated=False
    )
    assert res["status"] == "error"
    assert "SMTP" in res["message"]
