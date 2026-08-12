"""
Automated E2E Test Suite for Jarvis Super-Intelligence & Autonomous Personal Assistant.
Tests UI initialization, Universal Document RAG, Python Sandbox, Vision Bridge,
Web Research Tools, Session Persistence, Workspace File Operations, and Personal Memory.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Import core modules
from src.config import PROVIDERS, PERSONAS, WORKSPACE_DIR
from src.tools.document_tools import extract_text_from_file, get_files_hash
from src.tools.python_executor import python_interpreter, get_and_clear_figure_buffer
from src.tools.web_tools import wikipedia_lookup, read_webpage_content
from src.memory.session_manager import SessionManager, SESSIONS_DIR
from src.vision.vision_bridge import (
    register_uploaded_image, clear_active_images, analyze_image_deep, _ACTIVE_IMAGES
)
from src.assistant.profile_manager import ProfileManager, PROFILE_FILE, MEMORY_FILE
from src.assistant.workspace_tools import (
    write_workspace_file, read_workspace_file, list_workspace_files,
    generate_excel_spreadsheet, generate_word_document, save_personal_memory
)
from src.assistant.goal_planner import GoalPlanner
from src.assistant.autonomous_runner import AutonomousRunner

# ----------------- Fixtures ----------------- #

@pytest.fixture
def sample_pdf():
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 500 800] /Contents 5 0 R >>\nendobj\n4 0 obj\n<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>\nendobj\n5 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Jarvis Super Intelligence) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000222 00000 n \n0000000305 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n398\n%%EOF"
    class DummyFile:
        def __init__(self, content, name="test_doc.pdf"):
            self._content = content
            self.name = name
        def getvalue(self):
            return self._content
        def read(self):
            return self._content
        def seek(self, pos):
            pass
    return DummyFile(pdf_content)

@pytest.fixture
def sample_csv():
    csv_content = b"name,age,salary\nAlice,30,95000\nBob,35,120000\nCharlie,28,80000"
    class DummyCSV:
        def __init__(self, content, name="employees.csv"):
            self._content = content
            self.name = name
        def getvalue(self):
            return self._content
        def read(self):
            return self._content
        def seek(self, pos):
            pass
    return DummyCSV(csv_content)

@pytest.fixture
def sample_text():
    txt_content = b"Quantum computing uses qubits and superposition to process complex algorithms."
    class DummyTXT:
        def __init__(self, content, name="quantum.txt"):
            self._content = content
            self.name = name
        def getvalue(self):
            return self._content
        def read(self):
            return self._content
        def seek(self, pos):
            pass
    return DummyTXT(txt_content)

# ----------------- Unit & Integration Tests ----------------- #

def test_config_and_personas():
    """Verify system config, default models, and personas are properly registered."""
    assert "OpenRouter" in PROVIDERS
    assert "OpenAI" in PROVIDERS
    assert "Custom" in PROVIDERS
    assert "JARVIS Supreme" in PERSONAS
    assert "Deep Research Analyst" in PERSONAS
    assert "Data & Vision Scientist" in PERSONAS
    assert len(PERSONAS["JARVIS Supreme"]["prompt"]) > 50

def test_universal_document_parser(sample_pdf, sample_csv, sample_text):
    """Test text extraction across PDF, CSV, and TXT files."""
    # PDF
    pdf_text, pdf_meta = extract_text_from_file(sample_pdf)
    assert "Jarvis Super Intelligence" in pdf_text
    assert pdf_meta["filename"] == "test_doc.pdf"

    # CSV
    csv_text, csv_meta = extract_text_from_file(sample_csv)
    assert "Alice" in csv_text
    assert "salary" in csv_text
    assert csv_meta["rows"] == 3
    assert "age" in csv_meta["columns"]

    # TXT
    txt_text, txt_meta = extract_text_from_file(sample_text)
    assert "Quantum computing" in txt_text
    assert txt_meta["filename"] == "quantum.txt"

def test_file_hash_change_detection(sample_pdf, sample_csv):
    """Verify hash calculation correctly identifies file state changes."""
    hash1 = get_files_hash([sample_pdf])
    hash2 = get_files_hash([sample_pdf, sample_csv])
    hash3 = get_files_hash([sample_pdf])
    assert hash1 == hash3
    assert hash1 != hash2

def test_python_interpreter_tool():
    """Verify sandboxed Python REPL executes calculations and handles errors safely."""
    # Simple math
    result = python_interpreter.invoke("x = 15 * 4; print(f'Result: {x}')")
    assert "Result: 60" in result

    # Data manipulation with numpy / math
    calc = python_interpreter.invoke("import math; print(round(math.sqrt(144) * 10, 2))")
    assert "120" in calc

    # Plot generation capture
    plot_res = python_interpreter.invoke("""
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
print('Plot created')
""")
    assert "Plot created" in plot_res
    figs = get_and_clear_figure_buffer()
    assert len(figs) >= 1

    # Error handling
    err_res = python_interpreter.invoke("print(1 / 0)")
    assert "ZeroDivisionError" in err_res

def test_web_tools():
    """Verify Wikipedia lookup and Web URL fetcher tools handle queries and fallbacks."""
    # Wikipedia
    wiki_res = wikipedia_lookup.invoke("Artificial Intelligence")
    assert "Wikipedia:" in wiki_res or "intelligence" in wiki_res.lower()

    # Invalid URL handling
    fail_url = read_webpage_content.invoke("http://invalid-non-existent-domain-12345.xyz")
    assert "Failed to retrieve" in fail_url

def test_session_manager_persistence():
    """Verify session creation, saving, loading, and markdown transcript export."""
    test_session_id = "test_session_9999"
    messages = [
        HumanMessage(content="Hello Jarvis, what is the square root of 256?"),
        AIMessage(content="The square root of 256 is 16.")
    ]
    
    SessionManager.save_session(test_session_id, messages, persona="JARVIS Supreme")
    loaded_msgs, loaded_persona = SessionManager.load_session(test_session_id)
    
    assert len(loaded_msgs) == 2
    assert loaded_msgs[0].content == messages[0].content
    assert loaded_msgs[1].content == messages[1].content
    assert loaded_persona == "JARVIS Supreme"

    # Export markdown
    md = SessionManager.export_as_markdown(test_session_id, messages, "JARVIS Supreme")
    assert "J.A.R.V.I.S. Intelligence Briefing" in md
    assert "square root of 256" in md

    # Clean up test session file
    test_file = SESSIONS_DIR / f"{test_session_id}.json"
    if test_file.exists():
        test_file.unlink()

def test_vision_bridge_registration():
    """Verify image registration and active image tracking."""
    clear_active_images()
    assert len(_ACTIVE_IMAGES) == 0

    # Create dummy 100x100 RGB image
    import io
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(0, 240, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    class DummyImageFile:
        def __init__(self, content, name="test_image.png"):
            self._content = content
            self.name = name
        def getvalue(self):
            return self._content

    res = register_uploaded_image(DummyImageFile(buf.getvalue()))
    assert res["status"] == "success"
    assert "test_image.png" in _ACTIVE_IMAGES
    assert _ACTIVE_IMAGES["test_image.png"]["size"] == (100, 100)

    clear_active_images()
    assert len(_ACTIVE_IMAGES) == 0

def test_personal_profile_and_memory():
    """Verify ProfileManager handles user profiles and persistent long-term memories."""
    # Profile load/save
    profile = ProfileManager.load_profile()
    assert "user_name" in profile
    
    # Memory addition
    ProfileManager.add_memory("User prefers dark mode and concise summaries", "ui_pref")
    memories = ProfileManager.load_memories()
    assert len(memories) >= 1
    assert any("User prefers dark mode" in m["fact"] for m in memories)

    # Assistant system context injection
    ctx = ProfileManager.get_assistant_system_context()
    assert "PERSONAL ASSISTANT PROTOCOL" in ctx
    assert "LONG-TERM MEMORY CONTEXT" in ctx

def test_workspace_file_operations():
    """Verify workspace tools: write, read, list, excel, and word document generation."""
    # 1. Write file
    write_res = write_workspace_file.invoke({
        "filename": "test_report.md",
        "content": "# Test Executive Briefing\n\nKey finding: Revenue increased by 42%."
    })
    assert "Successfully created workspace file" in write_res
    
    # 2. Read file
    read_res = read_workspace_file.invoke({"filename": "test_report.md"})
    assert "Revenue increased by 42%" in read_res

    # 3. List workspace files
    list_res = list_workspace_files.invoke({"subdirectory": ""})
    assert "test_report.md" in list_res

    # 4. Generate Excel spreadsheet
    json_data = json.dumps([
        {"Company": "TechCorp", "Q1_Rev": 50000, "Growth": 0.15},
        {"Company": "InnovateAI", "Q1_Rev": 85000, "Growth": 0.35}
    ])
    excel_res = generate_excel_spreadsheet.invoke({
        "filename": "financial_model.xlsx",
        "json_table_data": json_data,
        "sheet_name": "Revenue"
    })
    assert "Successfully generated Excel spreadsheet" in excel_res

    # 5. Generate Word document
    word_res = generate_word_document.invoke({
        "filename": "briefing.docx",
        "title": "Quarterly Strategy",
        "markdown_content": "## Overview\nThis is a strategy brief.\n- Point 1\n- Point 2"
    })
    assert "Successfully generated" in word_res or "Saved document" in word_res

def test_autonomous_runner_mock_execution():
    """Verify AutonomousRunner executes subtasks sequentially and generates mission deliverables."""
    class MockOrchestrator:
        def __init__(self):
            self.tools = []
            self.agent_executor = None
        def _build_executor(self):
            return None
        def run(self, user_input, chat_history):
            return {
                "output": f"Delivered result for subtask: {user_input[:50]}...",
                "steps": [{"type": "tool_end", "output": "ok", "timestamp": "12:00:00"}],
                "figures": [],
                "annotated_images": []
            }

    mock_orch = MockOrchestrator()
    runner = AutonomousRunner(orchestrator=mock_orch)

    mock_plan = {
        "goal_summary": "Test Multi-Step Autonomous Goal",
        "estimated_steps": 2,
        "tasks": [
            {
                "id": "task_1",
                "title": "Information Research",
                "instruction": "Search for market facts",
                "expected_deliverable": "Market facts summary",
                "status": "pending"
            },
            {
                "id": "task_2",
                "title": "Report Generation",
                "instruction": "Write final summary file",
                "expected_deliverable": "Workspace report",
                "status": "pending"
            }
        ]
    }

    result = runner.execute_plan(mock_plan, "Perform Market Analysis")
    assert result["plan"]["status"] == "completed"
    assert result["plan"]["tasks"][0]["status"] == "completed"
    assert result["plan"]["tasks"][1]["status"] == "completed"
    assert "Autonomous Mission Complete" in result["final_summary"]
