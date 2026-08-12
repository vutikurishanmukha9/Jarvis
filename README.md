# J.A.R.V.I.S. — Autonomous Multimodal Intelligence & Personal Assistant System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.1+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blueviolet.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**J.A.R.V.I.S.** is a next-generation autonomous AI personal assistant and multimodal super-intelligence engine. It can decompose complex human goals into executable plans, execute multi-step workflows autonomously on your behalf, generate real-world deliverables (Excel spreadsheets, Word documents, Markdown reports, Python scripts), analyze visual feeds with YOLOv8 & OCR, and maintain persistent long-term memory across sessions.

---

## Core Capabilities

### 1. Autonomous Goal Planning & Mission Execution
- **Autonomous Task Decomposition**: Automatically breaks high-level objectives into sequential, actionable subtask DAGs.
- **Self-Correction & Reflection Loop**: Catches errors in script executions or tool queries, analyzes error tracebacks, and self-corrects without requiring manual intervention.
- **Live Mission Telemetry**: Streams real-time progress percentages, step statuses, and intermediate deliverables in a dedicated **Autonomous Mission Control** dashboard.

### 2. Workspace File Operations & Deliverable Generation
- **Dedicated Sandbox**: Operates inside a secure `workspace/` directory.
- **Microsoft Excel Generator**: Synthesizes tabular data into structured `.xlsx` workbooks with custom sheets and formatted columns.
- **Microsoft Word & Markdown Generator**: Generates formal reports, whitepapers, and briefings in `.docx` and `.md`.
- **Python Script Generator & Runner**: Writes and executes standalone Python automation scripts.

### 3. Universal Document & Data RAG Engine
- **Multi-Format Ingestion**: Ingests **PDF, Word (.docx), Excel (.xlsx), CSV, Markdown (.md), JSON, and Code (.py, .txt)**.
- **Tabular Data Understanding**: Automatically extracts tabular schemas, row counts, and previews for CSV and Excel files.
- **Vector Retrieval**: Computes MD5 content hashes for smart caching and performs semantic search via FAISS vector indexing.

### 4. Computer Vision & Optical Intelligence ([vision_engine/](vision_engine/))
- **YOLOv8 Object Detection**: Real-time object localization, counting, classification, and visual bounding box annotations rendered in chat.
- **OCR Text Extraction**: Extracts text from receipts, charts, diagrams, and photos using Tesseract OCR and OpenCV.
- **Color & Quality Metrics**: K-Means dominant color extraction, brightness, contrast, and blur detection.

### 5. Live Python Sandbox & Data Analytics
- **Sandboxed REPL**: Executes calculations, statistical simulations, and machine learning models in a sandboxed Python runtime.
- **Visual Chart Capture**: Automatically captures and renders **Matplotlib and Plotly figures** directly in the chat stream.

### 6. Deep Web & Encyclopedic Research
- **DuckDuckGo Search**: Real-time web queries and news verification.
- **Wikipedia Tool**: Encyclopedic lookups and scientific summaries.
- **Web URL Scraper**: Fetches full article content for deep cross-referencing.

### 7. Personal Profile & Long-Term Memory
- **Customized User Profile**: Identifies your name, role, preferred output style, and custom directives.
- **Persistent Memory Store**: Retains facts, project knowledge, and user preferences across application restarts.
- **Multi-Session Chat**: Save, switch, and export chat transcripts to Markdown (`.md`).

---

## UI Architecture

The interface provides a clean frosted glass design system with 4 dedicated hubs:
1. **Intelligence Chat**: Direct multimodal conversational interface with Thought Telemetry, Vision cards, and Python Sandbox.
2. **Autonomous Mission Control**: Goal assignment input, quick presets, real-time checklist progress, and executive deliverable synthesis.
3. **Workspace Files**: Live file explorer to preview, inspect, and download generated `.xlsx`, `.docx`, `.md`, `.csv`, and `.py` files.
4. **Personal Profile & Memory**: Customize user moniker, role, directives, and view recorded long-term memories.


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
pytest test_app.py -v
```

### Test Coverage (10/10 Tests Passing):
- System configuration and provider registration (`test_config_and_personas`)
- Universal document parsing across PDF, CSV, TXT (`test_universal_document_parser`)
- Hash-based vector cache change detection (`test_file_hash_change_detection`)
- Sandboxed Python code execution and figure capture (`test_python_interpreter_tool`)
- DuckDuckGo and Wikipedia web research tools (`test_web_tools`)
- Multi-session chat persistence and transcript export (`test_session_manager_persistence`)
- YOLOv8 image registration and tracking (`test_vision_bridge_registration`)
- Personal Profile and persistent memory persistence (`test_personal_profile_and_memory`)
- Workspace file operations, Excel and Word generation (`test_workspace_file_operations`)
- Autonomous multi-step execution loop (`test_autonomous_runner_mock_execution`)

---

## License
Distributed under the MIT License. Built with Streamlit, LangChain, PyTorch, YOLOv8, and FAISS.
