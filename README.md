# J.A.R.V.I.S. — Autonomous Multimodal Intelligence & Personal Assistant System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.1+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blueviolet.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**J.A.R.V.I.S.** is an enterprise-grade autonomous AI personal assistant and multimodal super-intelligence engine. It decomposes complex human goals into executable plans, executes multi-step workflows autonomously on your behalf, generates real-world deliverables (Excel spreadsheets, Word documents, Markdown reports, Python scripts), evaluates ATS resume compatibility, executes personalized cold email outreach campaigns, analyzes visual feeds with YOLOv8 & OCR, and maintains persistent long-term memory across sessions.

---

## Core Capabilities

### 1. Autonomous Goal Planning & Mission Execution
- **Autonomous Task Decomposition**: Automatically breaks high-level objectives into sequential, actionable subtask DAGs.
- **Self-Correction & Reflection Loop**: Catches errors in script executions or tool queries, analyzes error tracebacks, and self-corrects without requiring manual intervention.
- **Live Mission Telemetry**: Streams real-time progress percentages, step statuses, and intermediate deliverables in a dedicated **Autonomous Mission Control** dashboard.

### 2. Smart HR Outreach & Cold Email Engine ([src/modules/outreach/](src/modules/outreach/))
- **Dynamic Personalization Engine**: Ingests recipient spreadsheets (CSV/Excel) and substitutes `{firstName}`, `{company}`, `{role}`, and custom variables across bulk campaigns.
- **Multi-Stage Follow-Up Sequence Copilot**: Generates structured 4-stage outreach sequences (Day 1 Pitch, Day 4 Value Add, Day 8 Soft Nudge, Day 14 Graceful Breakup).
- **Interactive Live Previewer**: Renders per-recipient email previews before dispatching.
- **Safe Simulation & SMTP Dispatcher**: Executes sandboxed dry-run campaign batches with Excel audit logs (`workspace/`) and supports live TLS/SSL SMTP dispatch.

### 3. Career Intelligence & ATS Resume Engine ([src/modules/career/](src/modules/career/))
- **5-Pillar ATS Compatibility Scoring**: Evaluates candidate resumes against target job descriptions across Keywords, Skills, Experience, Education, and Formatting.
- **Missing Keyword Detection**: Uncovers missing critical, important, and optional keywords with negation awareness.
- **Technical Skill Extraction**: Identifies 200+ technical skills categorized across 13 domains (Programming, Cloud/DevOps, AI/ML, Databases, Frameworks).
- **Market Valuation & Salary Prediction**: Estimates competitive salary bands based on experience, education level, and role taxonomy.
- **Tailored Resume Generator**: Generates formatted, ATS-optimized Microsoft Word (`.docx`) and Markdown (`.md`) resumes directly into the workspace.

### 4. Workspace File Operations & Deliverable Generation
- **Dedicated Sandbox**: Operates inside a secure `workspace/` directory.
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

### 7. Live Python Sandbox & Data Analytics
- **Sandboxed REPL**: Executes calculations, statistical simulations, and machine learning models in a sandboxed Python runtime.
- **Visual Chart Capture**: Automatically captures and renders **Matplotlib and Plotly figures** directly in the chat stream.

### 8. Deep Web & Encyclopedic Research
- **DuckDuckGo Search**: Real-time web queries and news verification.
- **Wikipedia Tool**: Encyclopedic lookups and scientific summaries.
- **Web URL Scraper**: Fetches full article content for deep cross-referencing.

### 9. Personal Profile & Long-Term Memory
- **Customized User Profile**: Identifies your name, role, preferred output style, and custom directives.
- **Persistent Memory Store**: Retains facts, project knowledge, and user preferences across application restarts.
- **Multi-Session Chat**: Save, switch, and export chat transcripts to Markdown (`.md`).

---

## UI Architecture

The interface provides a clean frosted glass design system with 6 dedicated hubs:
1. **Intelligence Chat**: Multimodal conversational interface with Thought Telemetry, Vision cards, and Python REPL.
2. **Autonomous Mission Control**: Goal assignment input, quick presets, real-time checklist progress, and executive deliverable synthesis.
3. **Career & ATS Studio**: Resume audit, ATS compatibility score gauges, missing keyword chips, salary band predictions, and one-click tailored resume generation.
4. **HR Outreach & Campaigns**: Dynamic spreadsheet recipient parsing, live per-recipient previewer, 4-stage follow-up cadence generator, and campaign execution telemetry.
5. **Workspace Files**: Live file explorer to preview, inspect, and download generated `.xlsx`, `.docx`, `.md`, `.csv`, and `.py` files.
6. **Personal Profile & Memory**: Customize user moniker, role, directives, and view recorded long-term memories.

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

### Test Coverage (14/14 Tests Passing):
- **`tests/test_core.py`**: Configuration, personas, and session manager persistence (`test_config_and_personas`, `test_session_manager_persistence`).
- **`tests/test_assistant.py`**: Personal profile, long-term memory, workspace operations, and autonomous runner (`test_personal_profile_and_memory`, `test_workspace_file_operations`, `test_autonomous_runner_mock_execution`).
- **`tests/test_modules.py`**: Vision registration, ATS scoring, skill & salary estimation, outreach campaign manager & dispatcher (`test_vision_bridge_registration`, `test_career_ats_scoring`, `test_career_skill_and_salary_tools`, `test_outreach_campaign_manager`, `test_outreach_bridge_and_dispatcher`).
- **`tests/test_tools.py`**: Document parsing, file hash detection, sandboxed Python executor, and web search tools (`test_universal_document_parser`, `test_file_hash_change_detection`, `test_python_interpreter_tool`, `test_web_tools`).

---

## License
Distributed under the MIT License. Built with Streamlit, LangChain, PyTorch, YOLOv8, and FAISS.
