# J.A.R.V.I.S. — Autonomous Multimodal Intelligence & Personal Assistant System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.1+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blueviolet.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**J.A.R.V.I.S.** is an enterprise-grade autonomous AI personal assistant and multimodal super-intelligence engine. It decomposes complex human goals into dependency-aware execution DAGs, executes multi-step workflows autonomously on your behalf with semantic output verification, generates real-world deliverables (Excel spreadsheets, Word documents, Markdown reports, Python scripts), evaluates ATS resume compatibility, executes personalized cold email outreach campaigns with built-in safety gates, analyzes visual feeds with YOLOv8 & OCR, and maintains persistent long-term memory across sessions.

---

## Core Capabilities

### 1. Autonomous Goal Planning & Mission Execution
- **Dependency-Aware Task Decomposition**: Automatically breaks high-level objectives into actionable subtask DAGs resolved via topological sorting (Kahn's algorithm).
- **Artifact Store Pattern**: Subtasks receive only the outputs from their declared dependencies rather than concatenating full history, eliminating token bloat.
- **Semantic Output Verification**: After tool execution, the engine evaluates whether deliverables meet instruction specifications before proceeding, triggering self-correction retries when outputs fall short.
- **Live Mission Telemetry**: Streams real-time progress percentages, step statuses, and intermediate deliverables in a dedicated **Autonomous Mission Control** dashboard.

### 2. Smart HR Outreach & Cold Email Engine ([src/modules/outreach/](src/modules/outreach/))
- **Dynamic Personalization Engine**: Ingests recipient spreadsheets (CSV/Excel) and substitutes `{firstName}`, `{company}`, `{role}`, and custom variables across bulk campaigns.
- **Multi-Stage Follow-Up Sequence Copilot**: Generates structured 4-stage outreach sequences (Day 1 Pitch, Day 4 Value Add, Day 8 Soft Nudge, Day 14 Graceful Breakup).
- **Interactive Live Previewer**: Renders per-recipient email previews before dispatching.
- **Mandatory Simulation Gate & SMTP Dispatcher**: Agent tool dispatches are strictly forced into simulation mode with Excel audit logs (`workspace/`); live SMTP dispatch requires explicit human-in-the-loop confirmation in the UI.

### 3. Career Intelligence & ATS Resume Engine ([src/modules/career/](src/modules/career/))
- **5-Pillar ATS Compatibility Scoring**: Evaluates candidate resumes against target job descriptions across Keywords, Skills, Experience, Education, and Formatting, clamped strictly to $[0, 100]$.
- **Missing Keyword Detection**: Uncovers missing critical, important, and optional keywords with negation awareness.
- **Technical Skill Extraction**: Identifies 200+ technical skills categorized across 13 domains (Programming, Cloud/DevOps, AI/ML, Databases, Frameworks).
- **Heuristic Compensation Estimation**: Estimates competitive salary bands based on experience, education level, and role taxonomy using transparent feature-weighted estimation.
- **Tailored Resume Generator**: Generates formatted, ATS-optimized Microsoft Word (`.docx`) and Markdown (`.md`) resumes directly into the workspace.

### 4. Workspace File Operations & Deliverable Generation
- **Dedicated Sandbox**: Operates inside a secure `workspace/` directory with strict path confinement.
- **Microsoft Excel Generator**: Synthesizes tabular data into structured `.xlsx` workbooks with custom sheets and formatted columns.
- **Microsoft Word & Markdown Generator**: Generates formal reports, whitepapers, and briefings in `.docx` and `.md`.
- **Python Script Generator & Runner**: Writes and executes standalone Python automation scripts.

### 5. Universal Document & Data RAG Engine
- **Multi-Format Ingestion**: Ingests **PDF, Word (.docx), Excel (.xlsx), CSV, Markdown (.md), JSON, and Code (.py, .txt)**.
- **Tabular Data Understanding**: Automatically extracts tabular schemas, row counts, and previews for CSV and Excel files.
- **Vector Retrieval**: Computes MD5 content hashes for smart caching and performs semantic search via FAISS vector indexing.

### 6. Computer Vision & Optical Intelligence ([src/modules/vision/](src/modules/vision/))
- **YOLOv8 Object Detection**: Real-time object localization, counting, classification, and visual bounding box annotations rendered in chat.
- **OCR Text Extraction**: Extracts text from receipts, charts, diagrams, and photos using Tesseract OCR and OpenCV.
- **Color & Quality Metrics**: K-Means dominant color extraction, brightness, contrast, and blur detection.

### 7. Controlled Python Execution & Data Analytics
- **Controlled REPL Environment**: Executes calculations, statistical modeling, and data manipulation with blocked dangerous imports (`os`, `subprocess`, `socket`), restricted builtins, a 30s timeout, and a 50KB output limit.
- **Visual Chart Capture**: Automatically captures and renders **Matplotlib and Plotly figures** directly in the chat stream.

### 8. Deep Web & Encyclopedic Research
- **DuckDuckGo Search**: Real-time web queries and news verification.
- **Wikipedia Tool**: Encyclopedic lookups and scientific summaries.
- **SSRF-Guarded Web Scraper**: Fetches full article content with URL scheme validation, private/internal IP blocking, response size caps, and redirect limits.

### 9. Personal Profile & Long-Term Memory Lifecycle
- **Customized User Profile**: Identifies your name, role, preferred output style, and custom directives.
- **Persistent Memory Lifecycle**: Full CRUD support (`add`, `update`, `delete`, `clear`) with `source` tracking and `confidence` scoring.
- **Multi-Session Chat**: Save, switch, and export chat transcripts to Markdown (`.md`).

---

## UI Architecture

The interface provides a clean frosted glass design system with 6 dedicated hubs:
1. **Intelligence Chat**: Multimodal conversational interface with Thought Telemetry, Vision cards, and Python REPL.
2. **Autonomous Mission Control**: Goal assignment input, quick presets, real-time checklist progress, and executive deliverable synthesis.
3. **Career & ATS Studio**: Resume audit, ATS compatibility score gauges, missing keyword chips, compensation estimates, and one-click tailored resume generation.
4. **HR Outreach & Campaigns**: Dynamic spreadsheet recipient parsing, live per-recipient previewer, 4-stage follow-up cadence generator, and campaign execution telemetry with live approval gates.
5. **Workspace Files**: Live file explorer to preview, inspect, and download generated `.xlsx`, `.docx`, `.md`, `.csv`, and `.py` files.
6. **Personal Profile & Memory**: Customize user moniker, role, directives, manage long-term memories with confidence indicators.

---

## Quick Start

### 1. Prerequisites
- Python 3.10 - 3.12 (Python 3.12 recommended)
- Tesseract OCR (optional, for OCR image text extraction)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/vutikurishanmukha9/Jarvis.git
cd Jarvis

# Create and activate virtual environment
python -m venv venv312
.\venv312\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch J.A.R.V.I.S.
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Automated Test Suite

Run the full end-to-end test suite:
```bash
pytest tests/ -v
```

### Test Coverage (27/27 Tests Passing):
- **`tests/test_core.py`**: Configuration, personas, and session manager persistence (`test_config_and_personas`, `test_session_manager_persistence`).
- **`tests/test_assistant.py`**: Personal profile, memory CRUD lifecycle, workspace operations, autonomous runner artifact store, and topological sort with cycle detection (`test_personal_profile_and_memory`, `test_workspace_file_operations`, `test_autonomous_runner_mock_execution`, `test_memory_lifecycle_delete_and_update`, `test_autonomous_runner_artifact_store`, `test_topological_sort_linear_dependencies`, `test_topological_sort_parallel_tasks`, `test_topological_sort_cycle_detection`).
- **`tests/test_modules.py`**: Vision registration, ATS scoring with range clamping, candidate skill & compensation estimation tools, outreach campaign manager, simulation dispatch, and safety gates (`test_vision_bridge_registration`, `test_career_ats_scoring`, `test_career_skill_and_salary_tools`, `test_ats_score_clamped_to_range`, `test_outreach_campaign_manager`, `test_outreach_bridge_and_dispatcher`, `test_outreach_dispatch_always_simulated`).
- **`tests/test_tools.py`**: Document parsing, file hash detection, controlled Python executor with import blocklist and size limits, web tools with scheme validation and private IP SSRF blocking (`test_universal_document_parser`, `test_file_hash_change_detection`, `test_python_interpreter_tool`, `test_web_tools`, `test_python_executor_blocks_dangerous_imports`, `test_python_executor_blocks_nested_imports`, `test_python_executor_output_size_limit`, `test_web_tools_blocks_private_ips`, `test_web_tools_rejects_non_http_schemes`, `test_web_tools_allows_valid_urls`).

---

## License
Distributed under the MIT License. Built with Streamlit, LangChain, PyTorch, YOLOv8, and FAISS.
