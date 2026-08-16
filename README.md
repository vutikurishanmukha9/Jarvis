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

## Modular Automated Test Suite

Run the full production test suite:
```bash
pytest tests/ -v
```

### Modular Architecture (25 Dedicated Test Files):
- **`tests/core/`**: Configuration, multi-session persistence, thought step tracer callbacks, and orchestrator tool aggregation ([test_config.py](tests/core/test_config.py), [test_session_manager.py](tests/core/test_session_manager.py), [test_thought_tracer.py](tests/core/test_thought_tracer.py), [test_orchestrator.py](tests/core/test_orchestrator.py)).
- **`tests/assistant/`**: Goal decomposition schemas, Kahn's topological sort across complex DAGs, autonomous runner artifact store, semantic output verification, memory CRUD lifecycle, and path-confined workspace operations ([test_goal_planner.py](tests/assistant/test_goal_planner.py), [test_topological_sort.py](tests/assistant/test_topological_sort.py), [test_autonomous_runner.py](tests/assistant/test_autonomous_runner.py), [test_output_verification.py](tests/assistant/test_output_verification.py), [test_profile_manager.py](tests/assistant/test_profile_manager.py), [test_workspace_tools.py](tests/assistant/test_workspace_tools.py)).
- **`tests/tools/`**: Controlled Python executor with dangerous import blocking, SSRF-guarded web scraper with private IP filters, multi-format document parsers (TXT, CSV, DOCX, XLSX, JSON, PY), and MD5 cache hashing ([test_python_executor.py](tests/tools/test_python_executor.py), [test_python_security.py](tests/tools/test_python_security.py), [test_web_tools.py](tests/tools/test_web_tools.py), [test_web_security_ssrf.py](tests/tools/test_web_security_ssrf.py), [test_document_parsers.py](tests/tools/test_document_parsers.py), [test_document_hash_rag.py](tests/tools/test_document_hash_rag.py)).
- **`tests/modules/`**: 5-pillar ATS scoring formulas, section-aware field weights, 13-domain skill taxonomy, heuristic compensation estimation, recruiter spreadsheet normalization, 4-stage cadence generation, simulated email delivery with Excel audit exports, and vision quality & K-Means clustering ([test_ats_scorer.py](tests/modules/test_ats_scorer.py), [test_ats_helpers.py](tests/modules/test_ats_helpers.py), [test_skill_extractor.py](tests/modules/test_skill_extractor.py), [test_career_bridge.py](tests/modules/test_career_bridge.py), [test_outreach_campaign.py](tests/modules/test_outreach_campaign.py), [test_outreach_dispatcher.py](tests/modules/test_outreach_dispatcher.py), [test_outreach_bridge.py](tests/modules/test_outreach_bridge.py), [test_vision_bridge.py](tests/modules/test_vision_bridge.py), [test_vision_algorithms.py](tests/modules/test_vision_algorithms.py)).

---

## License
Distributed under the MIT License. Built with Streamlit, LangChain, PyTorch, YOLOv8, and FAISS.
