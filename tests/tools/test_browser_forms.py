"""
Unit tests validating Browser Capability 3: Forms.
Tests form field filling, placeholder identification, and form payload submission.
"""

from src.tools.browser_tools import (
    browser_fill_form,
    browser_submit_form,
    get_browser_session,
)


def test_fill_form_no_page_loaded() -> None:
    """Test filling form with no active page loaded."""
    session = get_browser_session()
    session.reset()
    res = browser_fill_form.invoke({"field_name": "email", "value": "test@example.com"})
    assert "Error: No active web page loaded" in res


def test_fill_form_by_input_name() -> None:
    """Test filling an input field identified by its name attribute."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/apply"
    session.page_content = "<html><body><form><input name='applicant_name' type='text' /></form></body></html>"

    res = browser_fill_form.invoke({"field_name": "applicant_name", "value": "Shanmukh"})
    assert "Set form field 'applicant_name'" in res
    assert "Shanmukh" in res
    assert session.form_data["applicant_name"] == "Shanmukh"


def test_fill_form_by_placeholder() -> None:
    """Test filling an input field identified by placeholder text."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/login"
    session.page_content = "<html><body><input placeholder='Enter your work email' /></body></html>"

    res = browser_fill_form.invoke({"field_name": "Enter your work email", "value": "lead@company.ai"})
    assert "Set form field 'Enter your work email'" in res
    assert session.form_data["Enter your work email"] == "lead@company.ai"


def test_submit_form_with_payload() -> None:
    """Test submitting the active form with buffered fields."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/signup"
    session.form_data = {"username": "tony_stark", "email": "tony@stark.com"}

    res = browser_submit_form.invoke({})
    assert "Form submitted successfully" in res
    assert "username: tony_stark" in res
    assert "email: tony@stark.com" in res
    assert len(session.form_data) == 0  # Buffer cleared


def test_submit_form_empty_warning() -> None:
    """Test submitting a form with no accumulated data."""
    session = get_browser_session()
    session.reset()
    session.current_url = "https://example.com/empty"
    session.form_data.clear()

    res = browser_submit_form.invoke({})
    assert "Warning: Form submitted with empty form data" in res
