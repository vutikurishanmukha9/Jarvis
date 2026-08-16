"""
J.A.R.V.I.S. — Autonomous Multimodal Intelligence & Personal Assistant System
Main Streamlit Application Entrypoint.
"""

import logging
import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure Jarvis project root is at index 0 of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path or sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Jarvis modules
from src.config import (
    PROVIDERS, PERSONAS, SUPPORTED_ALL_EXTENSIONS,
    SUPPORTED_DOC_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K,
    WORKSPACE_DIR
)
from src.ui.styles import APPLE_JARVIS_CSS, render_apple_header
from src.tools.document_tools import (
    get_files_hash, process_documents_and_build_vector_store,
    create_document_retriever_tool
)
from src.modules.vision import (
    register_uploaded_image, clear_active_images
)
from src.core.session_manager import SessionManager
from src.core.orchestrator import JarvisOrchestrator
from src.assistant.profile_manager import ProfileManager
from src.assistant.workspace_tools import get_workspace_tools, generate_word_document
from src.assistant.goal_planner import GoalPlanner
from src.assistant.autonomous_runner import AutonomousRunner
from src.modules.career import (
    calculate_deep_ats_metrics,
    get_salary_and_role_estimate,
    get_resume_skills_categorized
)
from src.modules.outreach import CampaignManager, EmailDispatcher

# 1. Page Configuration
st.set_page_config(
    page_title="J.A.R.V.I.S. — Super-Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject UI Design System
st.markdown(APPLE_JARVIS_CSS, unsafe_allow_html=True)

# 3. Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = SessionManager.generate_session_id()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_artifacts" not in st.session_state:
    st.session_state.message_artifacts = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None
if "document_summaries" not in st.session_state:
    st.session_state.document_summaries = []
if "image_summaries" not in st.session_state:
    st.session_state.image_summaries = []
if "mission_plan" not in st.session_state:
    st.session_state.mission_plan = None
if "mission_history" not in st.session_state:
    st.session_state.mission_history = []

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("### **J.A.R.V.I.S. Control**")
    st.caption("_Personal Intelligence Command Center_")
    
    # Provider & API Key
    api_provider = st.selectbox(
        "LLM Provider",
        list(PROVIDERS.keys()),
        index=0,
        help="Select your AI reasoning provider"
    )
    provider_config = PROVIDERS[api_provider]
    
    if api_provider == "Custom":
        base_url = st.text_input("Custom Base URL", value=provider_config["base_url"])
    else:
        base_url = provider_config["base_url"]
        
    api_key = st.text_input(
        f"{api_provider} API Key",
        type="password",
        help=provider_config["api_key_help"],
        placeholder="Paste your API key here..."
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    
    # Persona & Reasoning Configuration
    st.markdown("### **Intelligence Persona**")
    persona_choice = st.selectbox("Active Persona", list(PERSONAS.keys()), index=0)
    persona_info = PERSONAS[persona_choice]
    st.caption(f"_{persona_info['tagline']}_")
    
    deep_research = st.toggle(
        "Deep Research & Autonomous Reasoning",
        value=False,
        help="Enables multi-step planning, iterative search, and structured intelligence synthesis."
    )

    # Model Configuration
    use_custom_model = st.checkbox("Custom Model Name", value=False)
    if use_custom_model:
        model_name = st.text_input("Model Name", value=provider_config["default_model"])
    else:
        model_name = st.selectbox("Reasoning Model", provider_config["default_models"], index=0)
        
    temperature = st.slider("Creativity / Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")

    # Universal Media & Document Ingestion
    st.markdown("### **Universal Data Ingestion**")
    uploaded_files = st.file_uploader(
        "Upload Documents, Datasets & Images",
        type=SUPPORTED_ALL_EXTENSIONS,
        accept_multiple_files=True,
        help="Supports PDF, Word (.docx), Excel, CSV, Text/Code (.txt, .md, .json), and Images (PNG, JPG, WEBP)"
    )

    # Advanced Settings Expander
    with st.expander("Vector & RAG Parameters"):
        chunk_size = st.slider("Chunk Size", 400, 2500, DEFAULT_CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, 50)
        num_chunks = st.slider("Relevant Chunks (Top-K)", 2, 10, DEFAULT_TOP_K)

    st.markdown("---")

    # Session Manager
    st.markdown("### **Session History**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("New Chat", use_container_width=True):
            st.session_state.session_id = SessionManager.generate_session_id()
            st.session_state.chat_history = []
            st.session_state.message_artifacts = {}
            st.rerun()
    with col_s2:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.message_artifacts = {}
            st.rerun()

    # Session switcher
    saved_sessions = SessionManager.list_sessions()
    if saved_sessions:
        selected_session = st.selectbox(
            "Resume Past Session",
            ["Current"] + saved_sessions,
            index=0
        )
        if selected_session != "Current" and selected_session != st.session_state.session_id:
            loaded_msgs, loaded_persona = SessionManager.load_session(selected_session)
            st.session_state.session_id = selected_session
            st.session_state.chat_history = loaded_msgs
            st.session_state.message_artifacts = {}
            st.rerun()

# 5. Process Uploaded Files & Media
doc_files = []
img_files = []
if uploaded_files:
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            img_files.append(f)
        else:
            doc_files.append(f)

# Register active images
clear_active_images()
st.session_state.image_summaries = []
for img in img_files:
    info = register_uploaded_image(img)
    if info.get("status") == "success":
        st.session_state.image_summaries.append(info)

# Process documents if present
if doc_files:
    current_hash = get_files_hash(doc_files)
    if st.session_state.processed_file_hash != current_hash:
        with st.spinner("Processing and vectorizing documents..."):
            v_store, summaries, status_msg = process_documents_and_build_vector_store(
                doc_files,
                api_provider=api_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.session_state.vector_store = v_store
            st.session_state.processed_file_hash = current_hash
            st.session_state.document_summaries = summaries
            st.sidebar.success(status_msg)
else:
    if st.session_state.vector_store is not None:
        st.session_state.vector_store = None
        st.session_state.processed_file_hash = None
        st.session_state.document_summaries = []

# 6. Render Header
profile = ProfileManager.load_profile()
user_name = profile.get("user_name", "Boss")
mode_label = "Deep Research Mode" if deep_research else "Direct Mode"
st.markdown(render_apple_header(persona_choice, mode_label), unsafe_allow_html=True)

# 7. Ingestion Status Bar (if any files uploaded)
if st.session_state.document_summaries or st.session_state.image_summaries:
    with st.container():
        st.markdown("<div class='apple-card'>", unsafe_allow_html=True)
        st.markdown(f"**Active Knowledge & Visual Feed (Assigned to {user_name}):**")
        for doc in st.session_state.document_summaries:
            fname = doc.get("filename", "Doc")
            extra = f"{doc.get('pages')} pages" if "pages" in doc else (f"{doc.get('rows')} rows" if "rows" in doc else f"{doc.get('size')} B")
            st.markdown(f"<span class='apple-pill'>{fname} ({extra})</span>", unsafe_allow_html=True)
        for img in st.session_state.image_summaries:
            st.markdown(f"<span class='apple-pill'>{img.get('filename')} ({img.get('dimensions')})</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# 8. API Key Guard
if not api_key:
    st.warning("JARVIS Standby: Please enter your API Key in the sidebar to activate the personal intelligence engine.")
    st.info("Tip: Select OpenRouter to use GPT-4o, Claude 3.5 Sonnet, or Gemini with a single API key.")
    st.stop()

# 9. Build Document Retriever Tool if vector store exists
document_tool = None
if st.session_state.vector_store is not None:
    document_tool = create_document_retriever_tool(st.session_state.vector_store, top_k=num_chunks)

# 10. Initialize Jarvis Agent Orchestrator
try:
    jarvis_engine = JarvisOrchestrator(
        api_provider=api_provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        persona=persona_choice,
        deep_research_mode=deep_research,
        document_tool=document_tool
    )
except Exception as e:
    st.error(f"Failed to initialize Jarvis Orchestrator: {str(e)}")
    logger.error(f"Orchestrator init error: {str(e)}", exc_info=True)
    st.stop()

# 11. Navigation Tabs
tab_chat, tab_mission, tab_career, tab_outreach, tab_workspace, tab_profile = st.tabs([
    "Intelligence Chat",
    "Autonomous Mission Control",
    "Career & ATS Studio",
    "HR Outreach & Campaigns",
    "Workspace Files",
    "Personal Profile & Memory"
])

# ==============================================================================
# TAB 1: INTELLIGENCE CHAT
# ==============================================================================
with tab_chat:
    # Quick Action Chips
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    quick_prompt = None
    with col_q1:
        if st.button("Summarize Knowledge", use_container_width=True):
            quick_prompt = "Provide a comprehensive executive summary of all uploaded documents and data."
    with col_q2:
        if st.button("Analyze Data & Trends", use_container_width=True):
            quick_prompt = "Perform deep statistical analysis on the uploaded dataset and generate key trend plots using Python."
    with col_q3:
        if st.button("Vision & OCR Scan", use_container_width=True):
            quick_prompt = "Analyze all uploaded images: detect objects with YOLOv8, read all text via OCR, and report image metrics."
    with col_q4:
        if st.button("Deep Web Research", use_container_width=True):
            quick_prompt = "Perform deep web and Wikipedia research on the latest breakthroughs in AI and robotics."

    # Display Chat Messages & Artifacts
    for idx, message in enumerate(st.session_state.chat_history):
        is_user = isinstance(message, HumanMessage)
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(message.content)
            
            # Display associated artifacts if present
            if not is_user and idx in st.session_state.message_artifacts:
                artifacts = st.session_state.message_artifacts[idx]
                
                # Thought & Tool Inspector
                steps = artifacts.get("steps", [])
                if steps:
                    with st.expander("Thought Process & Tool Telemetry", expanded=False):
                        for step in steps:
                            st_type = step.get("type")
                            t_time = step.get("timestamp", "")
                            if st_type == "tool_start":
                                st.markdown(f"**[{t_time}] Executing Tool:** `{step.get('tool')}`")
                                st.code(step.get("input", ""), language="python" if "python" in step.get("tool", "") else "text")
                            elif st_type == "tool_end":
                                st.markdown(f"**[{t_time}] Tool Result:**")
                                st.caption(step.get("output", ""))
                            elif st_type in ["tool_error", "error"]:
                                st.markdown(f"**[{t_time}] Error:** {step.get('error')}")

                # Render Figures
                figures = artifacts.get("figures", [])
                for fig in figures:
                    st.pyplot(fig)

                # Render Annotated Images
                annotated_imgs = artifacts.get("annotated_images", [])
                for name, ann_img in annotated_imgs:
                    st.image(ann_img, caption=f"YOLOv8 Object Detection: {name}", use_column_width=True)

    # Chat Input & Execution
    user_query = st.chat_input(f"Instruct JARVIS (math, documents, vision, web research, workspace files)...")
    if quick_prompt:
        user_query = quick_prompt

    if user_query:
        # Append and render user message
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute with Jarvis Orchestrator
        with st.chat_message("assistant"):
            with st.spinner("J.A.R.V.I.S. analyzing query and orchestrating tools..."):
                result = jarvis_engine.run(
                    user_input=user_query,
                    chat_history=st.session_state.chat_history[:-1]
                )
                
                answer_text = result.get("output", "")
                st.markdown(answer_text)

                # Render thought process
                steps = result.get("steps", [])
                if steps:
                    with st.expander("Thought Process & Tool Telemetry", expanded=False):
                        for step in steps:
                            st_type = step.get("type")
                            t_time = step.get("timestamp", "")
                            if st_type == "tool_start":
                                st.markdown(f"**[{t_time}] Executing Tool:** `{step.get('tool')}`")
                                st.code(step.get("input", ""), language="python" if "python" in step.get("tool", "") else "text")
                            elif st_type == "tool_end":
                                st.markdown(f"**[{t_time}] Tool Result:**")
                                st.caption(step.get("output", ""))
                            elif st_type in ["tool_error", "error"]:
                                st.markdown(f"**[{t_time}] Error:** {step.get('error')}")

                # Render Figures
                figures = result.get("figures", [])
                for fig in figures:
                    st.pyplot(fig)

                # Render Annotated Images
                annotated_imgs = result.get("annotated_images", [])
                for name, ann_img in annotated_imgs:
                    st.image(ann_img, caption=f"YOLOv8 Object Detection: {name}", use_column_width=True)

                # Store AI Message and Artifacts
                ai_msg = AIMessage(content=answer_text)
                st.session_state.chat_history.append(ai_msg)
                msg_idx = len(st.session_state.chat_history) - 1
                st.session_state.message_artifacts[msg_idx] = {
                    "steps": steps,
                    "figures": figures,
                    "annotated_images": annotated_imgs
                }

                # Save Session
                SessionManager.save_session(
                    st.session_state.session_id,
                    st.session_state.chat_history,
                    persona=persona_choice
                )

# ==============================================================================
# TAB 2: AUTONOMOUS MISSION CONTROL
# ==============================================================================
with tab_mission:
    st.markdown("### **Autonomous Mission Control**")
    st.caption(f"Assign high-level goals for JARVIS to plan, execute, and deliver autonomously on your behalf.")

    # Mission Preset Selectors
    col_mp1, col_mp2, col_mp3, col_mp4, col_mp5 = st.columns(5)
    preset_goal = ""
    with col_mp1:
        if st.button("Market & Competitor Dossier", use_container_width=True):
            preset_goal = "Conduct deep research on the top 3 AI humanoid robotics companies, generate a comparison Excel spreadsheet with key metrics, and write a comprehensive market report in Markdown."
    with col_mp2:
        if st.button("Financial Data & Modeling", use_container_width=True):
            preset_goal = "Analyze the uploaded financial data, write a Python script to compute YoY growth rates, plot revenue trend charts, and export a structured Excel financial model."
    with col_mp3:
        if st.button("Executive Briefing & Docx", use_container_width=True):
            preset_goal = "Synthesize all uploaded documents and recent web developments into a 3-section Executive Briefing Word document and save it in the workspace."
    with col_mp4:
        if st.button("Resume ATS Optimization", use_container_width=True):
            preset_goal = "Audit the candidate resume against modern tech industry benchmarks, extract key skills, identify missing critical keywords, generate a tailored Word document resume (.docx) in the workspace, and provide an executive improvement brief."
    with col_mp5:
        if st.button("Recruiter Outreach Campaign", use_container_width=True):
            preset_goal = "Draft a personalized recruiter outreach email with a 4-stage follow-up cadence for the target company and role, render previews with dynamic tags, and save the campaign briefing in the workspace."

    # Mission Input Box
    goal_input = st.text_area(
        "Goal / Mission Directive for JARVIS:",
        value=preset_goal if preset_goal else "",
        placeholder="e.g. Research solid-state battery breakthroughs, build a comparison table in Excel, write a formal summary in Markdown, and generate a Python projection chart...",
        height=100
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_mission = st.button("Launch Autonomous Mission", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Reset Mission State", use_container_width=False):
            st.session_state.mission_plan = None
            st.rerun()

    # Mission Execution Workflow
    if start_mission and goal_input.strip():
        # 1. Decompose Goal into Subtask DAG
        with st.spinner("Decomposing goal into structured subtasks..."):
            planner = GoalPlanner(
                api_provider=api_provider,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                temperature=0.1
            )
            doc_context = None
            if st.session_state.document_summaries:
                doc_context = f"Uploaded Documents: {[d.get('filename') for d in st.session_state.document_summaries]}"

            plan = planner.plan_goal(goal_input.strip(), context=doc_context)
            st.session_state.mission_plan = plan

        # 2. Autonomous Multi-Step Execution Runner
        plan_container = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_ui_callback(curr_plan: dict, task_id: str, message: str):
            status_text.markdown(f"**Status**: `{message}`")
            tasks = curr_plan.get("tasks", [])
            completed_count = len([t for t in tasks if t.get("status") == "completed"])
            total = max(len(tasks), 1)
            progress_bar.progress(completed_count / total)

        runner = AutonomousRunner(
            orchestrator=jarvis_engine,
            step_callback=update_ui_callback
        )

        with st.spinner("Auto-JARVIS executing mission subtasks autonomously..."):
            mission_result = runner.execute_plan(st.session_state.mission_plan, goal_input.strip())
            st.session_state.mission_plan = mission_result.get("plan")
            st.session_state.mission_history.append({
                "goal": goal_input.strip(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": mission_result.get("final_summary", "")
            })
            progress_bar.progress(1.0)
            status_text.success("Mission completed successfully!")

    # Display Active Mission Checklist & Progress
    if st.session_state.mission_plan:
        plan = st.session_state.mission_plan
        st.markdown("---")
        st.markdown(f"#### **Mission Plan**: {plan.get('goal_summary', 'Autonomous Plan')}")

        for idx, task in enumerate(plan.get("tasks", [])):
            t_status = task.get("status", "pending")
            badge_tone = "green" if t_status == "completed" else ("amber" if t_status == "in_progress" else "blue")
            
            with st.container():
                st.markdown(
                    f"<div class='apple-card'>"
                    f"<strong>Step {idx+1}: {task.get('title')}</strong> — <span class='apple-badge {badge_tone}'>{t_status.upper()}</span><br>"
                    f"<span style='color:#A1A1A6; font-size:0.85rem;'>{task.get('instruction')}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if task.get("result"):
                    with st.expander(f"Inspect Step {idx+1} Output", expanded=False):
                        st.markdown(task.get("result"))

    # Display Latest Mission Summary & Generated Deliverables
    if st.session_state.mission_history:
        latest = st.session_state.mission_history[-1]
        st.markdown("---")
        st.markdown("### **Mission Deliverables & Executive Briefing**")
        st.markdown(latest.get("summary", ""))

# ==============================================================================
# TAB 3: CAREER & ATS STUDIO
# ==============================================================================
with tab_career:
    st.markdown("### **Career & ATS Resume Studio**")
    st.caption("AI-powered resume optimization, ATS compatibility scoring, skill extraction, and market salary projection.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### **Candidate Resume**")
        input_resume_text = st.text_area(
            "Paste Resume Content (Text/Markdown):",
            height=240,
            placeholder="Senior Full Stack Engineer with 6+ years experience in Python, PyTorch, Docker, FastAPI, PostgreSQL, AWS. Developed high-throughput microservices. Master of Science in Computer Science."
        )
    with col_c2:
        st.markdown("#### **Target Job Description (JD)**")
        input_jd_text = st.text_area(
            "Paste Target Job Description:",
            height=240,
            placeholder="Looking for a Senior Python Developer with 5+ years experience. Required: Python, PyTorch, Docker, Kubernetes, PostgreSQL. Nice to have: AWS, CI/CD, Microservices. MS or BS in Computer Science."
        )

    col_act1, col_act2 = st.columns([2, 2])
    with col_act1:
        run_ats_btn = st.button("Run Full ATS Compatibility & Career Audit", type="primary", use_container_width=True)
    with col_act2:
        generate_opt_btn = st.button("Generate Tailored ATS Resume in Workspace (.docx)", use_container_width=True)

    if run_ats_btn:
        if not input_resume_text.strip():
            st.warning("Please paste resume content to analyze.")
        else:
            with st.spinner("Analyzing resume against ATS engine & ML models..."):
                role_info = get_salary_and_role_estimate(input_resume_text)
                ats_results = calculate_deep_ats_metrics(input_resume_text, input_jd_text.strip() if input_jd_text.strip() else input_resume_text)
                detected_skills = get_resume_skills_categorized(input_resume_text)
                
                st.session_state.last_ats_results = ats_results
                st.session_state.last_role_info = role_info
                st.session_state.last_detected_skills = detected_skills

    if "last_ats_results" in st.session_state:
        ats = st.session_state.last_ats_results
        role = st.session_state.last_role_info
        skills = st.session_state.last_detected_skills

        st.markdown("---")
        
        # Overall Score Metric
        score = ats.get("ats_score", 0)
        interp = ats.get("interpretation", {})
        sub = ats.get("sub_scores", {})
        badge_color = interp.get("color", "blue")
        if badge_color == "red":
            badge_tone = "red"
        elif badge_color in ["green", "teal"]:
            badge_tone = "green"
        else:
            badge_tone = "amber"

        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
        with col_m1:
            st.metric("Overall ATS Score", f"{score}/100")
        with col_m2:
            st.metric("Skill Match", f"{sub.get('skill_match', 0)}%")
        with col_m3:
            st.metric("Title Match", f"{sub.get('title_match', 0)}%")
        with col_m4:
            st.metric("Experience", f"{sub.get('experience', 0)}%")
        with col_m5:
            st.metric("Achievements", f"{sub.get('achievement', 0)}%")
        with col_m6:
            st.metric("Formatting", f"{100 - sub.get('formatting_penalty', 0)}%")

        st.markdown(
            f"<div class='apple-card'>"
            f"<strong>ATS Assessment: <span class='apple-badge {badge_tone}'>{interp.get('badge', 'Status').upper()}</span></strong><br>"
            f"<span style='color:#A1A1A6; font-size:0.9rem;'>{interp.get('message', '')}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Market Compensation & Role Prediction
        sal = role.get("salary_estimate", {})
        if sal and "range" in sal:
            curr = sal.get("currency", "₹")
            st.markdown(
                f"<div class='apple-card'>"
                f"<strong>Market Valuation & Role Prediction:</strong><br>"
                f"• Target Job Classification: <strong>{role.get('job_title')}</strong> ({role.get('category')} Domain)<br>"
                f"• Estimated Market Base: <strong>{curr}{sal.get('base', 0):,}</strong> (Range: {curr}{sal['range'].get('min', 0):,} – {curr}{sal['range'].get('max', 0):,}) &bull; Confidence: <strong>{sal.get('confidence', 'Moderate')}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Skills & Missing Keywords
        col_sk1, col_sk2 = st.columns(2)
        with col_sk1:
            st.markdown("#### **Detected Candidate Skills by Domain**")
            if skills:
                for cat, items in skills.items():
                    c_name = cat.replace("_", " ").title()
                    pills = " ".join([f"<span class='apple-pill'>{s}</span>" for s in items])
                    st.markdown(f"**{c_name}**: {pills}", unsafe_allow_html=True)
            else:
                st.caption("No standard technical skills detected.")

        with col_sk2:
            st.markdown("#### **Missing Job Keywords & Skill Gaps**")
            missing = ats.get("missing_keywords", {})
            crit = missing.get("critical", [])
            imp = missing.get("important", [])
            
            if crit:
                st.markdown("**Critical Keywords (Must Include):**")
                crit_pills = " ".join([f"<span class='apple-badge red'>{k}</span>" for k in crit])
                st.markdown(crit_pills, unsafe_allow_html=True)
            if imp:
                st.markdown("**Important Keywords (Recommended):**")
                imp_pills = " ".join([f"<span class='apple-badge amber'>{k}</span>" for k in imp])
                st.markdown(imp_pills, unsafe_allow_html=True)
            if not crit and not imp:
                st.success("No missing critical or important keywords detected!")

        # Actionable Suggestions
        suggs = ats.get("suggestions", [])
        if suggs:
            st.markdown("#### **Priority Optimization Actions**")
            for idx, s in enumerate(suggs):
                st.markdown(f"- **Action {idx+1}**: {s}")

    if generate_opt_btn and input_resume_text.strip():
        with st.spinner("Generating tailored ATS-optimized resume document in workspace..."):
            opt_title = "Tailored Professional Resume"
            doc_content = f"# Professional Resume\n\n## Summary\nExperienced professional with technical expertise aligned to target job description.\n\n## Core Competencies\n"
            detected_skills = get_resume_skills_categorized(input_resume_text)
            for cat, items in detected_skills.items():
                doc_content += f"- **{cat.replace('_', ' ').title()}**: {', '.join(items)}\n"
            doc_content += f"\n## Professional Experience & Achievements\n{input_resume_text}\n"
            
            res_path = generate_word_document.invoke({
                "filename": "optimized_resume.docx",
                "title": opt_title,
                "markdown_content": doc_content
            })
            st.success(f"Generated: `{res_path}`. You can inspect or download it from the **Workspace Files** tab!")

# ==============================================================================
# TAB 4: HR OUTREACH & CAMPAIGNS
# ==============================================================================
with tab_outreach:
    st.markdown("### **Smart HR Outreach & Cold Email Campaigns**")
    st.caption("AI-powered personalized outreach, spreadsheet recipient parsing, dynamic tag substitution, and multi-stage follow-up sequences.")

    outreach_mode = st.radio("Outreach Workflow Mode", ["Campaign Dispatcher & Live Preview", "Multi-Stage Follow-Up Sequence Generator"], horizontal=True)

    if outreach_mode == "Campaign Dispatcher & Live Preview":
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown("#### **Recipient Ingestion (CSV / Excel)**")
            st.caption("Format: `email, firstName, company, role` (additional columns become dynamic `{tags}`)")
            
            col_ld1, col_ld2 = st.columns(2)
            with col_ld1:
                if st.button("Load Tech Recruiters List", use_container_width=True):
                    st.session_state.current_recipients_csv = CampaignManager.get_sample_recipients_csv()
            with col_ld2:
                if st.button("Load Investor List", use_container_width=True):
                    inv_path = Path("src/modules/outreach/data/sample_recipients_vc_investors.csv")
                    if inv_path.exists():
                        with open(inv_path, "r", encoding="utf-8") as inf:
                            st.session_state.current_recipients_csv = inf.read()

            default_csv = st.session_state.get("current_recipients_csv", CampaignManager.get_sample_recipients_csv())
            recipients_input = st.text_area("Recipients CSV:", value=default_csv, height=155)
            parsed_records = CampaignManager.parse_recipients_data(recipients_input)
            st.caption(f"Valid recipients detected: **{len(parsed_records)}**")

        with col_o2:
            st.markdown("#### **Campaign Template Composer**")
            template_lib = CampaignManager.get_template_library()
            template_names = [v.get("name") for v in template_lib.values()] if template_lib else [
                "Recruiter & Hiring Manager Direct Pitch", "HR Recruiter to Candidate Sourcing", "Business Development & Partnership"
            ]
            template_choice = st.selectbox("Load Template Preset:", template_names + ["Custom Template"])
            
            selected_tmpl = None
            for v in template_lib.values():
                if v.get("name") == template_choice:
                    selected_tmpl = v
                    break

            if selected_tmpl:
                default_subj = selected_tmpl.get("subject", "")
                default_body = selected_tmpl.get("body", "")
            elif template_choice == "Custom Template":
                default_subj = "Connecting regarding {role} at {company}"
                default_body = "Hi {firstName},\n\nReaching out regarding {role} at {company}.\n\nBest,\n{senderName}"
            else:
                default_subj = "Quick question regarding {role} at {company} - {candidateName}"
                default_body = "Hi {firstName},\n\nI noticed {company} is actively scaling for {role}."

            subj_input = st.text_input("Email Subject Template:", value=default_subj)
            body_input = st.text_area("Email Body Template:", value=default_body, height=180)

        # Dynamic Tag Chips
        detected_tags = CampaignManager.extract_template_tags(subj_input + " " + body_input)
        if detected_tags:
            st.markdown(
                "**Detected Dynamic Personalization Tags:** " +
                " ".join([f"<span class='apple-pill'>{'{' + t + '}'}</span>" for t in detected_tags]),
                unsafe_allow_html=True
            )

        # Live Per-Recipient Previewer
        if parsed_records:
            st.markdown("---")
            st.markdown("#### **Live Recipient Previewer**")
            preview_idx = st.slider("Select Recipient to Preview Rendered Output:", 1, len(parsed_records), 1) - 1
            curr_rec = parsed_records[preview_idx]
            
            rend_subj = CampaignManager.render_template(subj_input, curr_rec, {"candidateName": user_name, "portfolioUrl": "https://linkedin.com"})
            rend_body = CampaignManager.render_template(body_input, curr_rec, {"candidateName": user_name, "portfolioUrl": "https://linkedin.com"})

            col_pv1, col_pv2 = st.columns([1, 2])
            with col_pv1:
                st.markdown(
                    f"<div class='apple-card'>"
                    f"<strong>Recipient Metadata:</strong><br>"
                    f"• Email: <code>{curr_rec.get('email')}</code><br>"
                    f"• Name: <strong>{curr_rec.get('firstName', '')}</strong><br>"
                    f"• Company: <strong>{curr_rec.get('company', '')}</strong><br>"
                    f"• Role: <strong>{curr_rec.get('role', '')}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_pv2:
                st.markdown(
                    f"<div class='apple-card'>"
                    f"<strong>Subject:</strong> {rend_subj}<br><br>"
                    f"<div style='white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 0.9rem;'>{rend_body}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Campaign Dispatch Controls
        st.markdown("---")
        col_dc1, col_dc2 = st.columns([2, 1])
        with col_dc1:
            is_simulated = st.checkbox("Safe Simulation Mode (Dry-run with preview validation & Excel audit logs, no live emails sent)", value=True)
        with col_dc2:
            dispatch_btn = st.button("Execute Outreach Campaign", type="primary", use_container_width=True)

        if dispatch_btn:
            if not parsed_records:
                st.warning("Please provide valid recipient records.")
            else:
                with st.spinner("Executing outreach campaign batch..."):
                    res = EmailDispatcher.dispatch(
                        subject_template=subj_input,
                        body_template=body_input,
                        recipients=parsed_records,
                        global_tags={"candidateName": user_name, "portfolioUrl": "https://linkedin.com"},
                        simulated=is_simulated
                    )
                    st.session_state.last_outreach_result = res

        if "last_outreach_result" in st.session_state:
            ores = st.session_state.last_outreach_result
            st.markdown(
                f"<div class='apple-card'>"
                f"<strong>Campaign Status: <span class='apple-badge green'>COMPLETED ({ores.get('sent')}/{ores.get('total')} Sent)</span></strong><br>"
                f"<span style='color:#A1A1A6; font-size:0.9rem;'>{ores.get('message')}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            if ores.get("delivery_logs"):
                st.dataframe(pd.DataFrame(ores.get("delivery_logs")))

    else:
        # Mode 2: Multi-Stage Follow-Up Sequence Generator
        st.markdown("#### **Multi-Stage Follow-Up Sequence Generator**")
        st.caption("Generate a 4-stage cadence (Initial Pitch, Day 4 Value Add, Day 8 Soft Nudge, Day 14 Graceful Breakup).")

        col_sq1, col_sq2 = st.columns(2)
        with col_sq1:
            t_role = st.text_input("Target Role / Position:", value="Senior Staff Software Engineer")
            t_comp = st.text_input("Target Company:", value="Stripe")
            c_name = st.text_input("Candidate Name:", value=user_name)
        with col_sq2:
            k_skills = st.text_input("Key Technical Skills / Stack:", value="Python, PyTorch, Distributed Systems, Kubernetes")
            k_achieve = st.text_input("Key Quantified Achievement:", value="increased core system throughput by 45% and cut latency by 30ms")
            p_url = st.text_input("Portfolio / GitHub URL:", value="https://github.com/vutikurishanmukha9")

        gen_seq_btn = st.button("Generate 4-Stage Cadence", type="primary", use_container_width=False)

        if gen_seq_btn or "last_sequence" in st.session_state:
            if gen_seq_btn:
                seq_data = CampaignManager.build_multi_stage_sequence(
                    target_role=t_role,
                    target_company=t_comp,
                    candidate_name=c_name,
                    key_skills=k_skills,
                    key_achievement=k_achieve,
                    portfolio_url=p_url
                )
                st.session_state.last_sequence = seq_data

            if "last_sequence" in st.session_state:
                seq = st.session_state.last_sequence
                for step in seq:
                    with st.container():
                        st.markdown(
                            f"<div class='apple-card'>"
                            f"<strong>{step['stage']}</strong><br>"
                            f"<strong>Subject:</strong> <code>{step['subject']}</code><br><br>"
                            f"<div style='white-space: pre-wrap; font-size:0.9rem;'>{step['body']}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                if st.button("Save Sequence to Workspace (.md)"):
                    md_text = f"# 4-Stage Outreach Sequence: {t_role} at {t_comp}\n\n"
                    for step in seq:
                        md_text += f"## {step['stage']}\n**Subject**: `{step['subject']}`\n\n```text\n{step['body']}\n```\n\n---\n\n"
                    seq_file = WORKSPACE_DIR / "outreach_cadence.md"
                    with open(seq_file, "w", encoding="utf-8") as sf:
                        sf.write(md_text)
                    st.success(f"Saved to `{seq_file}`! Inspect in **Workspace Files** tab.")

# ==============================================================================
# TAB 5: WORKSPACE FILES EXPLORER
# ==============================================================================
with tab_workspace:
    st.markdown("### **Workspace Files & Deliverables**")
    st.caption(f"All files generated or analyzed by JARVIS are securely stored in `{WORKSPACE_DIR.resolve()}`.")

    col_w1, col_w2 = st.columns([1, 4])
    with col_w1:
        if st.button("Refresh Workspace", use_container_width=True):
            st.rerun()

    ws_files = list(WORKSPACE_DIR.rglob("*"))
    actual_files = [f for f in ws_files if f.is_file()]

    if not actual_files:
        st.info("The workspace is currently empty. Ask JARVIS to generate reports, spreadsheets, or code to see them here.")
    else:
        for f in sorted(actual_files):
            rel = f.relative_to(WORKSPACE_DIR)
            size_kb = round(f.stat().st_size / 1024, 2)
            
            col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
            with col_f1:
                st.markdown(f"**`{rel}`** ({size_kb} KB)")
            with col_f2:
                # Read and download button
                try:
                    with open(f, "rb") as file_bytes:
                        st.download_button(
                            "Download",
                            data=file_bytes,
                            file_name=f.name,
                            key=f"dl_{f.name}_{f.stat().st_mtime}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.caption(f"Cannot read: {str(e)}")
            with col_f3:
                if st.button("Preview", key=f"prev_{f.name}"):
                    if f.suffix.lower() in [".txt", ".md", ".py", ".json", ".csv"]:
                        with open(f, "r", encoding="utf-8", errors="ignore") as pf:
                            st.code(pf.read(), language="python" if f.suffix == ".py" else ("json" if f.suffix == ".json" else "markdown"))
                    elif f.suffix.lower() in [".xlsx", ".xls"]:
                        try:
                            df_prev = pd.read_excel(f)
                            st.dataframe(df_prev)
                        except Exception as ex:
                            st.error(f"Cannot preview Excel: {str(ex)}")

# ==============================================================================
# TAB 4: PERSONAL PROFILE & MEMORY
# ==============================================================================
with tab_profile:
    st.markdown("### **Personal Assistant Profile & Memory**")
    st.caption("Customize how JARVIS identifies you, understands your preferences, and retains long-term knowledge.")

    current_prof = ProfileManager.load_profile()
    
    with st.form("profile_form"):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            u_name = st.text_input("Your Name / Moniker", value=current_prof.get("user_name", "Boss"))
            a_name = st.text_input("Assistant Name", value=current_prof.get("assistant_name", "Jarvis"))
        with col_p2:
            r_desc = st.text_input("Your Role / Focus Area", value=current_prof.get("role_description", "Lead Engineer & Creator"))
            p_style = st.text_input("Preferred Output Style", value=current_prof.get("preferred_style", "Concise, structured, executive-ready"))

        c_inst = st.text_area(
            "Persistent Custom Directives for JARVIS:",
            value=current_prof.get("custom_instructions", ""),
            height=80,
            help="Directives that JARVIS will always follow for all tasks."
        )

        save_prof_btn = st.form_submit_button("Save Profile Settings", type="primary")
        if save_prof_btn:
            new_prof = {
                "user_name": u_name,
                "assistant_name": a_name,
                "role_description": r_desc,
                "preferred_style": p_style,
                "custom_instructions": c_inst,
                "auto_execute_safe_code": True,
                "default_workspace": "workspace"
            }
            ProfileManager.save_profile(new_prof)
            st.success("Personal assistant profile updated successfully!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### **Long-Term Memory Facts**")
    memories = ProfileManager.load_memories()
    if memories:
        for m in memories:
            st.markdown(f"- **{m.get('fact')}** (_{m.get('timestamp')}_) <span class='apple-badge blue'>{m.get('category', 'general')}</span>", unsafe_allow_html=True)
    else:
        st.caption("No long-term memories recorded yet. Instruct JARVIS in chat or add below.")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        new_fact = st.text_input("Add a persistent memory fact about you or your projects:", placeholder="e.g. My primary coding language is Python 3.12 and I prefer dark mode visuals.")
    with col_m2:
        if st.button("Add Fact", use_container_width=True) and new_fact.strip():
            ProfileManager.add_memory(new_fact.strip(), "user_preference")
            st.success("Memory added!")
            st.rerun()

# Footer
st.markdown("---")
st.caption("J.A.R.V.I.S. SUPREME — Autonomous Personal Assistant & Super-Intelligence System")