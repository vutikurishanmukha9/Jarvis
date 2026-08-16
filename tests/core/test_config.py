"""
Tests for system configurations, providers, models, personas, and directory definitions.
"""

import pytest
from pathlib import Path
from src.config import (
    PROVIDERS,
    PERSONAS,
    SUPPORTED_DOC_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_ALL_EXTENSIONS,
    WORKSPACE_DIR,
    ASSISTANT_DIR,
    OUTREACH_DIR,
    DEFAULT_ASSISTANT_NAME,
    DEFAULT_USER_NAME
)

def test_provider_definitions():
    """Verify all supported AI providers have base URLs and default model lists."""
    assert "OpenRouter" in PROVIDERS
    assert "OpenAI" in PROVIDERS
    assert "Custom" in PROVIDERS

    openrouter = PROVIDERS["OpenRouter"]
    assert openrouter["base_url"] == "https://openrouter.ai/api/v1"
    assert len(openrouter["default_models"]) >= 4
    assert "openai/gpt-4o" in openrouter["default_models"]

    openai_prov = PROVIDERS["OpenAI"]
    assert "gpt-4o" in openai_prov["default_models"]
    assert openai_prov["default_model"] == "gpt-4o"

def test_system_personas_integrity():
    """Verify all 6 system personas contain prompt templates and taglines."""
    expected_personas = [
        "JARVIS Supreme",
        "Deep Research Analyst",
        "Data & Vision Scientist",
        "Code Architect & Engineer",
        "Career & Talent Strategist",
        "HR & Executive Outreach Specialist"
    ]
    for persona_name in expected_personas:
        assert persona_name in PERSONAS, f"Missing persona: {persona_name}"
        persona = PERSONAS[persona_name]
        assert "tagline" in persona and len(persona["tagline"]) > 5
        assert "prompt" in persona and len(persona["prompt"]) > 50

def test_system_directories_and_extensions():
    """Verify workspace/log paths exist and supported file extension whitelists are valid."""
    assert isinstance(WORKSPACE_DIR, Path)
    assert isinstance(ASSISTANT_DIR, Path)
    assert isinstance(OUTREACH_DIR, Path)

    assert ".pdf" in SUPPORTED_DOC_EXTENSIONS
    assert ".docx" in SUPPORTED_DOC_EXTENSIONS
    assert ".csv" in SUPPORTED_DOC_EXTENSIONS
    assert ".xlsx" in SUPPORTED_DOC_EXTENSIONS

    assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
    assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS

    # Extensions without dots for easy lookup
    assert "pdf" in SUPPORTED_ALL_EXTENSIONS
    assert "png" in SUPPORTED_ALL_EXTENSIONS
    assert DEFAULT_ASSISTANT_NAME == "Jarvis"
    assert DEFAULT_USER_NAME == "Boss"
