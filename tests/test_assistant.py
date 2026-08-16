import pytest
import os
import json
from pathlib import Path

from src.assistant.profile_manager import ProfileManager
from src.assistant.workspace_tools import (
    generate_excel_spreadsheet,
    generate_word_document,
    write_workspace_file,
    read_workspace_file,
    list_workspace_files,
    save_personal_memory,
    get_workspace_tools
)
from src.assistant.autonomous_runner import AutonomousRunner

def test_personal_profile_and_memory():
    """Verify user profile metadata and long-term memory fact recording."""
    # Check default profile loading
    profile = ProfileManager.load_profile()
    assert "user_name" in profile
    assert "role_description" in profile

    # Update profile
    profile["role_description"] = "Lead AI Architect"
    ProfileManager.save_profile(profile)
    updated = ProfileManager.load_profile()
    assert updated["role_description"] == "Lead AI Architect"

    # Add memories
    ProfileManager.add_memory("Project Alpha launched in Q3.", category="projects")
    ProfileManager.add_memory("Prefers Apple UI aesthetic and dark mode.", category="preferences")

    memories = ProfileManager.load_memories()
    assert len(memories) >= 2
    facts = [m["fact"] for m in memories]
    assert any("Project Alpha" in f for f in facts)

    # Render prompt injection context
    prompt_ctx = ProfileManager.format_context_for_prompt()
    assert "Project Alpha" in prompt_ctx

def test_workspace_file_operations():
    """Verify Excel, Word, and script creation in workspace."""
    # 1. Generate Excel
    table_sample = json.dumps([
        {"Company": "AlphaCorp", "Valuation": "10B", "Industry": "Robotics"},
        {"Company": "BetaScale", "Valuation": "5B", "Industry": "AI"}
    ])
    excel_res = generate_excel_spreadsheet.invoke({
        "filename": "test_companies.xlsx",
        "json_table_data": table_sample,
        "sheet_name": "Market Leaders"
    })
    assert "Successfully generated Excel" in excel_res

    # 2. Generate Word Document
    md_sample = "# Executive Summary\n\nThis is a formal test briefing generated for the user."
    doc_res = generate_word_document.invoke({
        "filename": "test_briefing.docx",
        "title": "Autonomous Executive Summary",
        "markdown_content": md_sample
    })
    assert "Successfully generated Word Document" in doc_res or "Saved document as Markdown" in doc_res

    # 3. Write Workspace File
    script_res = write_workspace_file.invoke({
        "filename": "compute_test.py",
        "content": "print('Computed 84')"
    })
    assert "Successfully created workspace file" in script_res

    # 4. Read & List Workspace Files
    content = read_workspace_file.invoke({"filename": "compute_test.py"})
    assert "Computed 84" in content

    file_list = list_workspace_files.invoke({})
    assert "compute_test.py" in file_list

    # 5. Tool suite
    tools = get_workspace_tools()
    assert len(tools) == 6
    names = [t.name for t in tools]
    assert "write_workspace_file" in names
    assert "generate_excel_spreadsheet" in names
    assert "generate_word_document" in names
    assert "save_personal_memory" in names

def test_autonomous_runner_mock_execution():
    """Verify autonomous runner step execution and plan status progression."""
    runner = AutonomousRunner(orchestrator=None)
    mock_plan = {
        "goal": "Perform Market Analysis",
        "status": "pending",
        "tasks": [
            {"id": "task_1", "title": "Gather raw competitor metrics", "instruction": "Search metrics", "status": "pending"},
            {"id": "task_2", "title": "Synthesize summary into Excel", "instruction": "Export Excel", "status": "pending"}
        ]
    }

    result = runner.execute_plan(mock_plan, "Perform Market Analysis")
    assert result["plan"]["status"] == "completed"
    assert result["plan"]["tasks"][0]["status"] == "completed"
    assert result["plan"]["tasks"][1]["status"] == "completed"
    assert "Autonomous Mission Complete" in result["final_summary"]
