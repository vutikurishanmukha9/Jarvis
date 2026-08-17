"""
Tests for Pydantic V2 Schemas: Subtasks, Goal Plans, Memory Records, User Profiles, and ATS Reports.
"""

import pytest
from pydantic import ValidationError

from src.core.schemas import (
    ATSReportModel,
    GoalPlanModel,
    MemoryEntryModel,
    OutreachRecipientModel,
    SubTaskModel,
    UserProfileModel,
)


def test_subtask_model_validation():
    """Verify subtask field validation and status normalization."""
    task = SubTaskModel(
        id="task_01",
        title="Parse Financial Dataset",
        instruction="Read the Excel spreadsheet and extract quarterly net revenues.",
        tool="python_interpreter",
        deliverable="Cleaned DataFrame",
        depends_on=[],
        status="PENDING",
    )
    assert task.id == "task_01"
    assert task.status == "pending"


def test_subtask_model_empty_id_rejected():
    """Verify validation error when subtask ID is whitespace or empty."""
    with pytest.raises(ValidationError):
        SubTaskModel(id="   ", title="Valid Title", instruction="Valid instruction long enough")


def test_goal_plan_dependency_filtering():
    """Verify GoalPlanModel filters non-existent dependency IDs."""
    plan = GoalPlanModel(
        mission_title="Quarterly Review Analysis",
        mission_objective="Analyze Q1 to Q4 sales data and produce executive Word report.",
        estimated_steps=2,
        subtasks=[
            SubTaskModel(id="t1", title="Extract Data", instruction="Extract all CSV data into workspace."),
            SubTaskModel(
                id="t2",
                title="Generate Report",
                instruction="Write Word report.",
                depends_on=["t1", "non_existent_t99"],
            ),
        ],
    )
    assert plan.subtasks[1].depends_on == ["t1"]


def test_memory_entry_confidence_clamping():
    """Verify memory confidence is strictly clamped to [0.0, 1.0]."""
    mem_high = MemoryEntryModel(fact="User prefers dark theme", confidence=1.5)
    assert mem_high.confidence == 1.0

    mem_low = MemoryEntryModel(fact="User likes coffee", confidence=-0.5)
    assert mem_low.confidence == 0.0


def test_user_profile_defaults():
    """Verify user profile initializes with sensible executive defaults."""
    profile = UserProfileModel(user_name="Alex", role="CTO")
    assert profile.user_name == "Alex"
    assert profile.role == "CTO"
    assert "Direct" in profile.preferred_tone


def test_ats_report_clamping():
    """Verify ATS report score clamping to [0, 100]."""
    report = ATSReportModel(
        ats_score=110.0, sub_scores={"keywords": 95.0, "skills": 80.0}, suggestions=["Add more action verbs"]
    )
    assert report.ats_score == 100.0


def test_outreach_recipient_email_validation():
    """Verify RFC email validation passes valid and rejects malformed addresses."""
    valid_rec = OutreachRecipientModel(
        first_name="Jane", company="Stark Industries", role="VP Engineering", email="jane.doe@stark.com"
    )
    assert valid_rec.email == "jane.doe@stark.com"

    with pytest.raises(ValidationError):
        OutreachRecipientModel(
            first_name="Jane", company="Stark Industries", role="VP", email="invalid-email-address-no-at"
        )
