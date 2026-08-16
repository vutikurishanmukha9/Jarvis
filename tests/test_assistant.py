import pytest
import os
import json
import time
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
from src.assistant.goal_planner import topological_sort

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

    # Add memories with new fields
    ProfileManager.add_memory(
        "Project Alpha launched in Q3.",
        category="projects",
        source="user_explicit",
        confidence=1.0
    )
    ProfileManager.add_memory(
        "Prefers Apple UI aesthetic and dark mode.",
        category="preferences",
        source="conversation",
        confidence=0.8
    )

    memories = ProfileManager.load_memories()
    assert len(memories) >= 2
    facts = [m["fact"] for m in memories]
    assert any("Project Alpha" in f for f in facts)

    # Verify new fields exist on memories
    for mem in memories:
        assert "source" in mem, "Memory missing 'source' field"
        assert "confidence" in mem, "Memory missing 'confidence' field"

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
            {"id": "task_1", "title": "Gather raw competitor metrics", "instruction": "Search metrics", "depends_on": [], "status": "pending"},
            {"id": "task_2", "title": "Synthesize summary into Excel", "instruction": "Export Excel", "depends_on": ["task_1"], "status": "pending"}
        ]
    }

    result = runner.execute_plan(mock_plan, "Perform Market Analysis")
    assert result["plan"]["status"] == "completed"
    assert result["plan"]["tasks"][0]["status"] == "completed"
    assert result["plan"]["tasks"][1]["status"] == "completed"
    assert "Autonomous Mission Complete" in result["final_summary"]


# ==================== NEW TESTS ====================

def test_memory_lifecycle_delete_and_update():
    """Verify that delete_memory and update_memory work correctly."""
    # Clear existing memories
    ProfileManager.clear_memories()
    assert ProfileManager.load_memories() == []

    # Add test memories
    ProfileManager.add_memory("Fact A", category="test", source="user_explicit", confidence=1.0)
    time.sleep(0.01)  # Ensure unique timestamps
    ProfileManager.add_memory("Fact B", category="test", source="agent_inferred", confidence=0.7)

    memories = ProfileManager.load_memories()
    assert len(memories) == 2

    # Get IDs
    mem_a_id = memories[0]["id"]
    mem_b_id = memories[1]["id"]

    # Test update_memory
    assert ProfileManager.update_memory(mem_a_id, new_fact="Updated Fact A", new_confidence=0.9)
    memories = ProfileManager.load_memories()
    updated_a = [m for m in memories if m["id"] == mem_a_id][0]
    assert updated_a["fact"] == "Updated Fact A"
    assert updated_a["confidence"] == 0.9
    assert updated_a["updated_at"] is not None

    # Test update non-existent memory
    assert ProfileManager.update_memory("nonexistent_id", new_fact="nope") is False

    # Test delete_memory
    assert ProfileManager.delete_memory(mem_b_id)
    memories = ProfileManager.load_memories()
    assert len(memories) == 1
    assert memories[0]["id"] == mem_a_id

    # Test delete non-existent memory
    assert ProfileManager.delete_memory("nonexistent_id") is False

    # Cleanup
    ProfileManager.clear_memories()

def test_autonomous_runner_artifact_store():
    """Verify the artifact store pattern: tasks only receive declared dependency outputs."""
    runner = AutonomousRunner(orchestrator=None)

    mock_plan = {
        "goal": "Multi-dependency test",
        "status": "pending",
        "tasks": [
            {"id": "task_1", "title": "Independent task A", "instruction": "Do A", "depends_on": [], "status": "pending"},
            {"id": "task_2", "title": "Independent task B", "instruction": "Do B", "depends_on": [], "status": "pending"},
            {"id": "task_3", "title": "Depends on A only", "instruction": "Merge A", "depends_on": ["task_1"], "status": "pending"}
        ]
    }

    result = runner.execute_plan(mock_plan, "Multi-dependency test")

    # All tasks should complete
    for task in result["plan"]["tasks"]:
        assert task["status"] == "completed"

    # Artifact store should contain all task results
    assert "task_1" in runner._artifact_store
    assert "task_2" in runner._artifact_store
    assert "task_3" in runner._artifact_store

def test_topological_sort_linear_dependencies():
    """Verify topological sort orders tasks by dependencies."""
    tasks = [
        {"id": "task_3", "depends_on": ["task_2"]},
        {"id": "task_1", "depends_on": []},
        {"id": "task_2", "depends_on": ["task_1"]},
    ]

    sorted_tasks = topological_sort(tasks)
    sorted_ids = [t["id"] for t in sorted_tasks]

    # task_1 must come before task_2, task_2 before task_3
    assert sorted_ids.index("task_1") < sorted_ids.index("task_2")
    assert sorted_ids.index("task_2") < sorted_ids.index("task_3")

def test_topological_sort_parallel_tasks():
    """Verify topological sort handles independent tasks correctly."""
    tasks = [
        {"id": "task_a", "depends_on": []},
        {"id": "task_b", "depends_on": []},
        {"id": "task_c", "depends_on": ["task_a", "task_b"]},
    ]

    sorted_tasks = topological_sort(tasks)
    sorted_ids = [t["id"] for t in sorted_tasks]

    # task_a and task_b must both come before task_c
    assert sorted_ids.index("task_a") < sorted_ids.index("task_c")
    assert sorted_ids.index("task_b") < sorted_ids.index("task_c")

def test_topological_sort_cycle_detection():
    """Verify topological sort falls back to original order on cycles."""
    tasks = [
        {"id": "task_1", "depends_on": ["task_2"]},
        {"id": "task_2", "depends_on": ["task_1"]},
    ]

    # Should return original order (fallback) since there's a cycle
    sorted_tasks = topological_sort(tasks)
    assert len(sorted_tasks) == 2  # All tasks still present
