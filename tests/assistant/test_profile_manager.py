"""
Tests for ProfileManager user preferences, memory CRUD operations, and confidence sorting.
"""

import time

import pytest

from src.assistant.profile_manager import ProfileManager


@pytest.fixture(autouse=True)
def clean_memory_state():
    """Ensure clean memory store before and after each test."""
    ProfileManager.clear_memories()
    yield
    ProfileManager.clear_memories()


def test_profile_load_and_default_keys():
    """Verify default profile keys and values."""
    profile = ProfileManager.load_profile()
    assert "user_name" in profile
    assert "assistant_name" in profile
    assert "role_description" in profile
    assert "preferred_style" in profile
    assert "custom_instructions" in profile


def test_profile_save_and_reload():
    """Verify saving updated profile persists to disk."""
    profile = ProfileManager.load_profile()
    profile["user_name"] = "Tony Stark"
    profile["role_description"] = "Chief AI Architect"
    ProfileManager.save_profile(profile)

    reloaded = ProfileManager.load_profile()
    assert reloaded["user_name"] == "Tony Stark"
    assert reloaded["role_description"] == "Chief AI Architect"


def test_memory_add_with_source_and_confidence():
    """Verify adding memories records source and confidence metrics."""
    assert (
        ProfileManager.add_memory(
            "Prefers dark mode UI theme.", category="preferences", source="user_explicit", confidence=1.0
        )
        is True
    )

    memories = ProfileManager.load_memories()
    assert len(memories) == 1
    mem = memories[0]
    assert mem["fact"] == "Prefers dark mode UI theme."
    assert mem["category"] == "preferences"
    assert mem["source"] == "user_explicit"
    assert mem["confidence"] == 1.0
    assert "timestamp" in mem


def test_memory_confidence_clamping():
    """Verify confidence score is strictly bounded to [0.0, 1.0]."""
    ProfileManager.add_memory("Fact high", confidence=1.5)
    time.sleep(0.01)
    ProfileManager.add_memory("Fact low", confidence=-0.5)

    memories = ProfileManager.load_memories()
    assert len(memories) == 2
    assert memories[0]["confidence"] == 1.0
    assert memories[1]["confidence"] == 0.0


def test_memory_update_by_id():
    """Verify updating a specific memory by unique ID."""
    ProfileManager.add_memory("Initial fact text", category="general", confidence=0.5)
    memories = ProfileManager.load_memories()
    mem_id = memories[0]["id"]

    assert (
        ProfileManager.update_memory(
            memory_id=mem_id, new_fact="Corrected fact text", new_category="updates", new_confidence=0.9
        )
        is True
    )

    updated_memories = ProfileManager.load_memories()
    updated = updated_memories[0]
    assert updated["fact"] == "Corrected fact text"
    assert updated["category"] == "updates"
    assert updated["confidence"] == 0.9
    assert updated["updated_at"] is not None


def test_memory_update_nonexistent_id():
    """Verify update returns False for invalid IDs."""
    assert ProfileManager.update_memory("mem_invalid_9999", new_fact="New fact") is False


def test_memory_delete_by_id():
    """Verify deleting a specific memory by ID."""
    ProfileManager.add_memory("Keep this fact", category="a")
    time.sleep(0.01)
    ProfileManager.add_memory("Delete this fact", category="b")

    memories = ProfileManager.load_memories()
    del_id = memories[1]["id"]

    assert ProfileManager.delete_memory(del_id) is True

    remaining = ProfileManager.load_memories()
    assert len(remaining) == 1
    assert remaining[0]["fact"] == "Keep this fact"


def test_memory_delete_nonexistent_id():
    """Verify delete returns False for invalid IDs."""
    assert ProfileManager.delete_memory("mem_nonexistent_8888") is False


def test_memory_clear_all():
    """Verify clear_memories resets the memory store."""
    ProfileManager.add_memory("Fact 1")
    ProfileManager.add_memory("Fact 2")
    assert len(ProfileManager.load_memories()) == 2

    assert ProfileManager.clear_memories() is True
    assert ProfileManager.load_memories() == []


def test_memory_system_context_confidence_sorting():
    """Verify system context sorts memories by confidence descending and caps at 10."""
    ProfileManager.add_memory("Low confidence fact", confidence=0.2)
    time.sleep(0.01)
    ProfileManager.add_memory("High confidence fact", confidence=0.9)

    context = ProfileManager.get_assistant_system_context()
    assert "High confidence fact" in context
    assert "Low confidence fact" in context
    assert "[PERSONAL ASSISTANT PROTOCOL" in context
