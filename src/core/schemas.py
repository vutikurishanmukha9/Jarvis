"""
Pydantic V2 Schemas for J.A.R.V.I.S. Core Data Models and Agent Contracts.
Provides strict type validation, field constraints, serialization, and schema enforcement.
"""

import re
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# 1. Autonomous Planning Schemas
# ==============================================================================

class SubTaskModel(BaseModel):
    """Schema for an individual autonomous subtask in an execution DAG."""
    id: str = Field(..., description="Unique identifier for the subtask (e.g., 'task_1')")
    title: str = Field(..., min_length=3, description="Concise human-readable title of the subtask")
    instruction: str = Field(..., min_length=5, description="Detailed instruction for the agent executor")
    tool: str = Field(default="general_assistant", description="Primary tool recommended for this task")
    deliverable: str = Field(default="", description="Expected deliverable or artifact produced")
    depends_on: List[str] = Field(default_factory=list, description="IDs of prerequisite subtasks")
    status: str = Field(default="pending", description="Task execution status: pending, in_progress, completed, failed")
    result: Optional[str] = Field(default=None, description="Output or artifact summary produced by execution")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Subtask ID cannot be empty")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"pending", "in_progress", "completed", "failed", "skipped"}
        if v.lower() not in valid_statuses:
            return "pending"
        return v.lower()


class GoalPlanModel(BaseModel):
    """Schema for a decomposed autonomous mission plan with topological dependencies."""
    mission_title: str = Field(..., min_length=3, description="High-level title for the autonomous goal")
    mission_objective: str = Field(..., min_length=5, description="Comprehensive objective statement")
    estimated_steps: int = Field(default=1, ge=1, le=10, description="Estimated number of execution steps")
    subtasks: List[SubTaskModel] = Field(..., min_length=1, max_length=10, description="Ordered list of subtasks")

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> "GoalPlanModel":
        task_ids = {t.id for t in self.subtasks}
        for task in self.subtasks:
            # Filter out any dependency references that don't exist in the plan
            task.depends_on = [dep for dep in task.depends_on if dep in task_ids and dep != task.id]
        return self


# ==============================================================================
# 2. Long-Term Memory & User Profile Schemas
# ==============================================================================

class MemoryEntryModel(BaseModel):
    """Schema for a persistent memory fact with provenance and confidence."""
    id: str = Field(default_factory=lambda: f"mem_{int(time.time()*1000)}", description="Unique memory ID")
    fact: str = Field(..., min_length=3, description="Fact or directive to remember")
    category: str = Field(default="preference", description="Memory category: preference, project, personal, technical, system")
    timestamp: float = Field(default_factory=time.time, description="Unix epoch timestamp when created")
    source: str = Field(default="conversation", description="Origin: user_explicit, conversation, agent_inferred")
    confidence: float = Field(default=1.0, description="Confidence score from 0.0 to 1.0")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, round(val, 2)))
        except (ValueError, TypeError):
            return 1.0


class UserProfileModel(BaseModel):
    """Schema for user profile configuration and executive directives."""
    user_name: str = Field(default="Executive Leader", min_length=1)
    role: str = Field(default="Technology Leader", min_length=1)
    preferred_tone: str = Field(default="Direct, precise, highly analytical, proactive.")
    custom_directives: str = Field(default="")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ==============================================================================
# 3. Career & ATS Scoring Schemas
# ==============================================================================

class ATSInterpretationModel(BaseModel):
    """Schema for ATS badge and executive summary."""
    badge: str = Field(default="Good Match")
    color: str = Field(default="green")
    message: str = Field(default="Resume satisfies core requirements.")


class ATSReportModel(BaseModel):
    """Schema for comprehensive 5-pillar ATS compatibility score output."""
    ats_score: float = Field(default=0.0, description="Overall ATS score clamped strictly to 0-100")
    sub_scores: Dict[str, float] = Field(default_factory=dict, description="Pillar scores (0-100)")
    missing_keywords: Dict[str, List[str]] = Field(
        default_factory=lambda: {"critical": [], "important": [], "optional": []},
        description="Missing keywords by priority"
    )
    suggestions: List[str] = Field(default_factory=list, description="Actionable improvement recommendations")
    interpretation: ATSInterpretationModel = Field(default_factory=ATSInterpretationModel)

    @field_validator("ats_score", mode="before")
    @classmethod
    def clamp_overall_score(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(100.0, round(val, 2)))
        except (ValueError, TypeError):
            return 0.0


# ==============================================================================
# 4. HR Outreach & Cold Email Schemas
# ==============================================================================

class OutreachRecipientModel(BaseModel):
    """Schema for outreach campaign recipient with RFC-compliant email verification."""
    first_name: str = Field(default="Colleague", description="Recipient's first name")
    company: str = Field(default="Target Organization", description="Target company or organization")
    role: str = Field(default="Hiring Manager", description="Recipient's job title or functional role")
    email: str = Field(..., description="Target email address")

    @field_validator("email")
    @classmethod
    def validate_email_syntax(cls, v: str) -> str:
        v = v.strip()
        # Basic RFC 5322 regex validation
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email address syntax: '{v}'")
        return v
