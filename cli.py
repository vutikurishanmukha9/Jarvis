#!/usr/bin/env python3
"""
J.A.R.V.I.S. Command Line Interface (CLI)
=========================================
Enterprise-grade Terminal Interface for J.A.R.V.I.S. Super-Intelligence.
Supports Interactive Chat, Autonomous DAG Goal Missions, ATS Audits,
Document Intelligence, Vision Inspection, and Web Matrix Operations.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Configure Windows UTF-8 Terminal Stream
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Initialize Rich Console with robust terminal settings
console = Console(force_terminal=True, legacy_windows=False)

# ASCII Banner
BANNER = r"""[bold cyan]
      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
      ██║███████║██████╔╝██║   ██║██║███████╗
 ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
 ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝[/bold cyan]
[dim]Joint Autonomous Real-time Vision & Intelligence System — CLI v2.0[/dim]
"""


def print_banner(persona: str = "Executive Assistant", model: str = "Default"):
    """Print the stylized system banner with active telemetry status."""
    console.print(BANNER)
    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="bold yellow")
    status_table.add_column(style="white")
    status_table.add_column(style="bold yellow")
    status_table.add_column(style="white")

    status_table.add_row("Persona:", persona, "Model:", model)
    status_table.add_row(
        "Guardrails AI:",
        "[bold green]ACTIVE (Ingress & Egress)[/bold green]",
        "Process Sandbox:",
        "[bold cyan]Win32 Kernel Job Object[/bold cyan]",
    )
    status_table.add_row(
        "Web Matrix:",
        "[bold magenta]3-Tier (Scrapling + Crawl4AI + Firecrawl)[/bold magenta]",
        "Voice Studio:",
        "[bold green]Faster-Whisper / SAPI Ready[/bold green]",
    )

    console.print(Panel(status_table, title="[bold cyan]System State Telemetry[/bold cyan]", border_style="cyan"))
    console.print(
        "[dim]Type your prompt, or use slash commands like [bold]/help[/bold], [bold]/goal <desc>[/bold], [bold]/workspace[/bold], [bold]/clear[/bold], [bold]/exit[/bold].[/dim]\n"
    )


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--persona", default="Executive Assistant", help="Agent persona to adopt.")
@click.option("--model", default=None, help="LLM model name (e.g. openai/gpt-4o, deepseek/deepseek-chat).")
@click.option("--no-guardrails", is_flag=True, help="Disable Guardrails AI safety inspection.")
def cli(ctx: click.Context, persona: str, model: Optional[str], no_guardrails: bool):
    """J.A.R.V.I.S. — Autonomous Multi-Modal Agent Intelligence Platform."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat, persona=persona, model=model, no_guardrails=no_guardrails)


# ----------------------------------------------------------------------
# Command: Interactive Chat
# ----------------------------------------------------------------------
@cli.command("chat")
@click.option("--persona", default="Executive Assistant", help="Agent persona to adopt.")
@click.option("--model", default=None, help="LLM model name override.")
@click.option("--no-guardrails", is_flag=True, help="Disable Guardrails AI.")
def chat(persona: str, model: Optional[str], no_guardrails: bool):
    """Launch the interactive conversational intelligence session."""
    from langchain_core.messages import AIMessage, HumanMessage

    from src.assistant.profile_manager import ProfileManager
    from src.assistant.workspace_tools import WORKSPACE_DIR
    from src.config import PERSONAS
    from src.core.orchestrator import JarvisOrchestrator

    system_instructions = PERSONAS.get(persona, PERSONAS.get("Executive Assistant", ""))
    model_name = model or os.getenv("JARVIS_MODEL", "openai/gpt-4o")

    print_banner(persona=persona, model=model_name)

    with console.status("[bold cyan]Initializing J.A.R.V.I.S. neural harness...[/bold cyan]", spinner="dots"):
        try:
            orchestrator = JarvisOrchestrator(
                model_name=model_name,
                system_instructions=system_instructions,
                enable_guardrails=not no_guardrails,
            )
        except Exception as e:
            console.print(f"[bold red]Initialization Error:[/bold red] {e}")
            sys.exit(1)

    profile_mgr = ProfileManager()
    chat_history: List[Any] = []

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]jarvis[/bold cyan] [bold green]>[/bold green]").strip()
            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd in ("/exit", "/quit", "/q"):
                    console.print("[yellow]Shutting down J.A.R.V.I.S. session. Goodbye, sir.[/yellow]")
                    break

                elif cmd in ("/clear", "/reset"):
                    chat_history.clear()
                    console.print("[green][OK] Conversation history cleared.[/green]\n")
                    continue

                elif cmd == "/history":
                    if not chat_history:
                        console.print("[dim]No messages in current session history.[/dim]\n")
                    else:
                        for i, msg in enumerate(chat_history):
                            sender = "User" if isinstance(msg, HumanMessage) else "J.A.R.V.I.S."
                            style = "bold cyan" if sender == "User" else "bold green"
                            console.print(f"[{style}][Turn {i + 1} - {sender}]:[/{style}] {msg.content[:150]}...")
                        console.print()
                    continue

                elif cmd == "/workspace":
                    _render_workspace_table()
                    continue

                elif cmd == "/read":
                    if not arg:
                        console.print("[red]Usage: /read <filename>[/red]")
                    else:
                        target = WORKSPACE_DIR / arg
                        if target.exists() and target.is_file():
                            console.print(Panel(target.read_text(encoding="utf-8", errors="replace"), title=arg))
                        else:
                            console.print(f"[red]File '{arg}' not found in workspace.[/red]")
                    continue

                elif cmd == "/profile":
                    prof = profile_mgr.get_profile()
                    mems = profile_mgr.get_memories()
                    console.print(f"[bold cyan]User Moniker:[/bold cyan] {prof.get('moniker', 'Sir')}")
                    console.print(f"[bold cyan]User Role:[/bold cyan] {prof.get('role', 'Executive')}")
                    console.print(f"[bold cyan]Active Memories Count:[/bold cyan] {len(mems)}")
                    for m in mems[:5]:
                        console.print(f"  • {m.get('content', '')} [dim](conf: {m.get('confidence', 1.0)})[/dim]")
                    console.print()
                    continue

                elif cmd == "/goal":
                    if not arg:
                        console.print("[red]Usage: /goal <mission description>[/red]")
                    else:
                        _execute_autonomous_goal_cli(arg, orchestrator)
                    continue

                elif cmd == "/search":
                    if not arg:
                        console.print("[red]Usage: /search <keyword or phrase>[/red]")
                    else:
                        from src.core.sqlite_session import get_sqlite_memory

                        store = get_sqlite_memory()
                        matches = store.search_memory(arg, limit=5)
                        if not matches:
                            console.print(f"[dim]No matching conversation memories found for '{arg}'.[/dim]\n")
                        else:
                            search_table = Table(
                                title=f"Memory Search Results for '{arg}'", show_header=True, header_style="bold yellow"
                            )
                            search_table.add_column("Session", style="cyan", width=22)
                            search_table.add_column("Role", style="magenta", width=8)
                            search_table.add_column("Matching Snippet", style="white")
                            for m in matches:
                                search_table.add_row(m["session_id"][:20], m["role"], m["snippet"])
                            console.print(search_table)
                            console.print()
                    continue

                elif cmd == "/skills":
                    from src.assistant.skill_learner import get_skill_learner

                    learner = get_skill_learner()
                    skills = learner.get_registered_skills()
                    if not skills:
                        console.print(
                            "[dim]No procedural skills learned yet. Run '/goal <mission>' to synthesize skills automatically.[/dim]\n"
                        )
                    else:
                        skill_table = Table(
                            title="Learned Procedural Skills (agentskills.io)",
                            show_header=True,
                            header_style="bold yellow",
                        )
                        skill_table.add_column("Skill Slug", style="bold green", width=22)
                        skill_table.add_column("Goal Summary", style="white")
                        skill_table.add_column("Tools", style="cyan")
                        skill_table.add_column("Steps", style="magenta", width=6)
                        for name, meta in skills.items():
                            skill_table.add_row(
                                name,
                                meta.get("summary", "")[:45],
                                ", ".join(meta.get("tools", [])),
                                str(meta.get("steps_count", 0)),
                            )
                        console.print(skill_table)
                        console.print()
                    continue

                elif cmd == "/help":
                    help_table = Table(title="Slash Commands", show_header=True, header_style="bold magenta")
                    help_table.add_column("Command", style="cyan")
                    help_table.add_column("Description")
                    help_table.add_row("/goal <desc>", "Decompose and execute an autonomous multi-step mission")
                    help_table.add_row(
                        "/search <query>", "Execute FTS5 full-text search across historical conversation memory"
                    )
                    help_table.add_row("/skills", "List procedural skills synthesized from past autonomous missions")
                    help_table.add_row("/workspace", "List all deliverable files in the workspace")
                    help_table.add_row("/read <file>", "Preview the content of a workspace file")
                    help_table.add_row("/profile", "Inspect user profile directives and long-term memory")
                    help_table.add_row("/history", "View message history of current session")
                    help_table.add_row("/clear", "Reset active chat memory")
                    help_table.add_row("/exit, /quit", "Terminate the CLI session")
                    console.print(help_table)
                    console.print()
                    continue

                else:
                    console.print(
                        f"[yellow]Unknown command '{cmd}'. Type [bold]/help[/bold] for available commands.[/yellow]\n"
                    )
                    continue

            # Execute Query via JarvisOrchestrator
            with console.status("[bold cyan]J.A.R.V.I.S. is deliberating...[/bold cyan]", spinner="dots"):
                start_time = time.time()
                result = orchestrator.run(user_input=user_input, chat_history=chat_history)
                elapsed = time.time() - start_time

            # Guardrails telemetry notice
            guardrails_meta = result.get("guardrails", {})
            if guardrails_meta.get("input_interception"):
                console.print(
                    f"[bold red]🛡️ Guardrails Intercept:[/bold red] {guardrails_meta.get('input_policy', 'Security violation')}"
                )

            # Thought step telemetry
            steps = result.get("steps", [])
            if steps:
                step_lines = []
                for s in steps:
                    tool_name = s.get("tool", s.get("type", "step"))
                    duration = s.get("duration", 0.0)
                    step_lines.append(f"• [bold cyan]{tool_name}[/bold cyan] ({duration:.2f}s)")
                telemetry_text = " | ".join(step_lines)
                console.print(f"[dim]Thought Trace ({elapsed:.2f}s): {telemetry_text}[/dim]")

            # Render output
            output_text = result.get("output", "").strip()
            if output_text:
                console.print(
                    Panel(
                        Markdown(output_text),
                        title=f"[bold green]J.A.R.V.I.S.[/bold green] [dim]({elapsed:.1f}s)[/dim]",
                        border_style="green",
                    )
                )
            else:
                console.print("[dim](No text output generated)[/dim]")

            # Render generated charts or files
            charts = result.get("charts", [])
            if charts:
                console.print(f"[magenta]📊 Generated {len(charts)} chart(s) in memory.[/magenta]")

            # Append to history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=output_text))
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Terminating J.A.R.V.I.S.[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Execution Exception:[/bold red] {e}\n")


# ----------------------------------------------------------------------
# Command: Autonomous Goal Mode
# ----------------------------------------------------------------------
@cli.command("goal")
@click.argument("mission", required=True)
@click.option("--context", default=None, help="Additional context or file text to guide decomposition.")
@click.option("--yes", "-y", is_flag=True, help="Auto-confirm DAG execution without prompting.")
def goal_cmd(mission: str, context: Optional[str], yes: bool):
    """Decompose and execute a high-level mission via topological DAG."""
    from src.core.orchestrator import JarvisOrchestrator

    console.print(BANNER)
    console.print(
        Panel(
            f"[bold white]{mission}[/bold white]",
            title="[bold cyan]Autonomous Mission Directive[/bold cyan]",
            border_style="cyan",
        )
    )

    with console.status("[bold cyan]Initializing J.A.R.V.I.S. core engine...[/bold cyan]", spinner="dots"):
        orchestrator = JarvisOrchestrator()

    _execute_autonomous_goal_cli(mission, orchestrator, context=context, auto_confirm=yes)


def _execute_autonomous_goal_cli(
    goal_str: str,
    orchestrator: Any,
    context: Optional[str] = None,
    auto_confirm: bool = False,
):
    """Deconstruct and execute an autonomous goal with Rich progress telemetry."""
    from src.assistant.autonomous_runner import AutonomousRunner
    from src.assistant.goal_planner import GoalPlanner

    planner = GoalPlanner()
    with console.status("[bold cyan]Deconstructing mission into dependency DAG...[/bold cyan]", spinner="dots"):
        plan = planner.plan_goal(goal_str, context=context)

    tasks = plan.get("tasks", [])
    if not tasks:
        console.print("[bold red]Planner could not decompose the mission into executable subtasks.[/bold red]")
        return

    # Render DAG Plan
    dag_table = Table(
        title="[bold yellow]Mission Dependency DAG[/bold yellow]", show_header=True, header_style="bold cyan"
    )
    dag_table.add_column("ID", style="bold yellow", width=10)
    dag_table.add_column("Subtask Title", style="white")
    dag_table.add_column("Tool Hint", style="cyan", width=14)
    dag_table.add_column("Expected Deliverable", style="green")
    dag_table.add_column("Depends On", style="magenta", width=14)

    for t in tasks:
        deps = ", ".join(t.get("depends_on", [])) or "None (Root)"
        dag_table.add_row(
            t.get("id", ""),
            t.get("title", ""),
            t.get("tool_hint", "auto"),
            t.get("expected_deliverable", "")[:35],
            deps,
        )

    console.print(dag_table)
    console.print(f"[dim]Goal Summary: {plan.get('goal_summary', '')}[/dim]\n")

    if not auto_confirm:
        if not Confirm.ask("[bold yellow]Do you authorize execution of this DAG plan?[/bold yellow]", default=True):
            console.print("[red]Execution aborted by user.[/red]\n")
            return

    # Execute
    console.print("[bold green]Executing Autonomous Subtasks...[/bold green]")
    runner = AutonomousRunner(orchestrator=orchestrator)

    with console.status("[bold green]Executing DAG...[/bold green]", spinner="dots") as status_spinner:

        def on_step(curr_plan: Dict[str, Any], task_id: str, msg: str):
            status_spinner.update(f"[bold green]{msg}[/bold green]")
            console.print(f"  [cyan][>][/cyan] {msg}")

        runner.step_callback = on_step
        final_result = runner.execute_plan(plan, initial_goal=goal_str)

    console.print("\n[bold green][OK] Mission Execution Completed![/bold green]")
    synthesis = final_result.get("synthesis", "Mission finished successfully.")
    console.print(
        Panel(Markdown(synthesis), title="[bold green]Executive Mission Synthesis[/bold green]", border_style="green")
    )

    _render_workspace_table()


# ----------------------------------------------------------------------
# Command: 5-Pillar ATS Resume Audit
# ----------------------------------------------------------------------
@cli.command("ats")
@click.argument("resume_file", type=click.Path(exists=True))
@click.argument("job_desc_file", type=click.Path(exists=True))
def ats_audit(resume_file: str, job_desc_file: str):
    """Run a comprehensive 5-Pillar ATS compatibility audit on a resume."""
    from src.modules.career.career_bridge import calculate_deep_ats_metrics, get_salary_and_role_estimate
    from src.tools.doc_rag import convert_document_with_docling

    console.print(BANNER)
    console.print(f"[bold cyan]Auditing Resume:[/bold cyan] {resume_file}")
    console.print(f"[bold cyan]Against Job Description:[/bold cyan] {job_desc_file}\n")

    def _read_file_text(path_str: str) -> str:
        p = Path(path_str)
        if p.suffix.lower() in (".pdf", ".docx", ".pptx"):
            res = convert_document_with_docling(p.name, p.read_bytes())
            if res:
                return res[0]
        return p.read_text(encoding="utf-8", errors="replace")

    with console.status(
        "[bold cyan]Extracting document contents and computing ATS metrics...[/bold cyan]", spinner="dots"
    ):
        resume_text = _read_file_text(resume_file)
        jd_text = _read_file_text(job_desc_file)

        metrics = calculate_deep_ats_metrics(resume_text, jd_text)
        salary_info = get_salary_and_role_estimate(resume_text)

    score = metrics.get("ats_score", 0)
    interpretation = metrics.get("interpretation", {})
    color = "green" if score >= 80 else "yellow" if score >= 60 else "red"

    # Score Panel
    console.print(
        Panel(
            f"[bold {color}]ATS Compatibility Score: {score}/100[/bold {color}]\n"
            f"[bold]Rating:[/bold] {interpretation.get('badge', 'N/A')} — {interpretation.get('message', '')}\n"
            f"[bold]Target Role Category:[/bold] {salary_info.get('job_title', 'General')} ({salary_info.get('category', 'Technology')})\n"
            f"[bold]Market Compensation Band:[/bold] ₹{salary_info.get('salary_range', {}).get('min_sal', 0):,} – ₹{salary_info.get('salary_range', {}).get('max_sal', 0):,}",
            title="[bold cyan]ATS Executive Summary[/bold cyan]",
            border_style=color,
        )
    )

    # 5 Pillars Breakdown
    sub_scores = metrics.get("sub_scores", {})
    pillar_table = Table(title="5-Pillar Compatibility Breakdown", show_header=True, header_style="bold magenta")
    pillar_table.add_column("Evaluation Pillar", style="white")
    pillar_table.add_column("Score", style="bold yellow", width=12)

    for pillar, p_score in sub_scores.items():
        pillar_table.add_row(pillar.capitalize(), f"{p_score}/100")
    console.print(pillar_table)

    # Missing Keywords
    missing = metrics.get("missing_keywords", {})
    crit = missing.get("critical", [])
    imp = missing.get("important", [])
    if crit or imp:
        console.print(f"\n[bold red]Critical Missing Keywords:[/bold red] {', '.join(crit[:12]) if crit else 'None'}")
        console.print(
            f"[bold yellow]Important Preferred Keywords:[/bold yellow] {', '.join(imp[:12]) if imp else 'None'}"
        )
    console.print()


# ----------------------------------------------------------------------
# Command: Document Intelligence
# ----------------------------------------------------------------------
@cli.command("doc")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--query", "-q", default=None, help="Semantic question to ask against the document.")
def doc_cmd(file_path: str, query: Optional[str]):
    """Ingest and analyze documents (PDF, Word, Markdown, Excel) via Docling/MinerU."""
    from src.tools.doc_rag import convert_document_with_docling

    p = Path(file_path)
    console.print(f"[bold cyan]Document Intelligence Engine:[/bold cyan] Ingesting {p.name}...")

    with console.status("[bold cyan]Parsing document structure and tables...[/bold cyan]", spinner="dots"):
        content_bytes = p.read_bytes()
        res = convert_document_with_docling(p.name, content_bytes)
        if res:
            markdown_text, meta = res
        else:
            markdown_text = p.read_text(encoding="utf-8", errors="replace")
            meta = {"filename": p.name, "engine": "native"}

    console.print(
        f"[bold green][OK] Ingestion Complete[/bold green] [dim](Engine: {meta.get('engine', 'native')})[/dim]"
    )

    if query:
        console.print(f"\n[bold yellow]Querying Document:[/bold yellow] {query}")
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        with console.status("[bold cyan]Building vector index and querying...[/bold cyan]", spinner="dots"):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(markdown_text)
            try:
                embeddings = OpenAIEmbeddings()
                db = FAISS.from_texts(chunks, embeddings)
                docs = db.similarity_search(query, k=3)
                answer = "\n\n---\n\n".join([d.page_content for d in docs])
                console.print(Panel(answer, title="[bold green]Relevant Excerpts[/bold green]"))
            except Exception as e:
                console.print(f"[bold red]Vector retrieval fallback error:[/bold red] {e}")
                console.print(Panel(markdown_text[:2000] + "\n...", title="Document Preview"))
    else:
        preview = markdown_text[:1500] + ("\n\n...[dim](truncated)[/dim]" if len(markdown_text) > 1500 else "")
        console.print(Panel(Markdown(preview), title=f"[bold cyan]{p.name} Structure[/bold cyan]"))


# ----------------------------------------------------------------------
# Command: Computer Vision & OCR
# ----------------------------------------------------------------------
@cli.command("vision")
@click.argument("image_path", type=click.Path(exists=True))
def vision_cmd(image_path: str):
    """Run YOLOv8 object detection and OCR text extraction on an image."""

    from src.modules.vision.vision_bridge import analyze_image_deep, register_uploaded_image

    p = Path(image_path)
    console.print(f"[bold cyan]Vision Intelligence Engine:[/bold cyan] Inspecting {p.name}...")

    class _FileWrapper:
        def __init__(self, path: Path):
            self.name = path.name
            self._bytes = path.read_bytes()

        def getvalue(self) -> bytes:
            return self._bytes

    with console.status("[bold cyan]Executing YOLOv8 and OCR models...[/bold cyan]", spinner="dots"):
        register_uploaded_image(_FileWrapper(p))
        analysis = analyze_image_deep(p.name)

    # Objects table
    objs = analysis.get("objects", [])
    if objs:
        obj_table = Table(title="Detected Objects (YOLOv8)", show_header=True, header_style="bold yellow")
        obj_table.add_column("Class", style="white")
        obj_table.add_column("Confidence", style="bold green", width=12)
        obj_table.add_column("Bounding Box [x1, y1, x2, y2]", style="cyan")

        for o in objs:
            obj_table.add_row(o["object"], f"{o['confidence']:.2f}", str(o["box"]))
        console.print(obj_table)
    else:
        console.print("[dim]No high-confidence objects detected by YOLOv8.[/dim]")

    # OCR text
    ocr_text = analysis.get("text_ocr", "").strip()
    if ocr_text:
        console.print(Panel(ocr_text, title="[bold cyan]Extracted OCR Scene Text[/bold cyan]", border_style="cyan"))
    else:
        console.print("[dim]No scene text detected in image.[/dim]")


# ----------------------------------------------------------------------
# Command: 3-Tier Web Scraper
# ----------------------------------------------------------------------
@cli.command("web")
@click.argument("url")
@click.option(
    "--tier",
    default="scrapling",
    type=click.Choice(["scrapling", "crawl4ai", "firecrawl"]),
    help="Scraping tier to utilize.",
)
def web_cmd(url: str, tier: str):
    """Extract clean content from a live URL using the 3-Tier Web Intelligence Matrix."""
    from src.tools.web import crawl_web_dynamic, map_website_urls, scrape_web_adaptive

    console.print(f"[bold cyan]Web Intelligence Engine:[/bold cyan] Fetching {url} via Tier: {tier.upper()}...")

    with console.status("[bold cyan]Connecting and parsing DOM...[/bold cyan]", spinner="dots"):
        if tier == "scrapling":
            result = scrape_web_adaptive.invoke({"url": url})
        elif tier == "crawl4ai":
            result = crawl_web_dynamic.invoke({"url": url})
        else:
            result = map_website_urls.invoke({"root_url": url})

    preview = str(result)[:2000] + ("\n...[dim](truncated)[/dim]" if len(str(result)) > 2000 else "")
    console.print(
        Panel(Markdown(preview), title=f"[bold green]Scraped Result ({tier})[/bold green]", border_style="green")
    )


# ----------------------------------------------------------------------
# Command: Voice Directive & Ambient Wake-Word Engine
# ----------------------------------------------------------------------
@cli.command("voice")
@click.argument("audio_file", required=False, type=click.Path(exists=True))
@click.option(
    "--listen", is_flag=True, help="Activate hands-free ambient wake-word listener ('Jarvis' / 'Hey Jarvis')."
)
def voice_cmd(audio_file: Optional[str], listen: bool):
    """Transcribe and execute a voice directive or run continuous ambient wake-word listening."""
    if listen:
        from src.core.orchestrator import JarvisOrchestrator
        from src.modules.voice.wake_word import get_wake_listener

        listener = get_wake_listener()
        orchestrator = JarvisOrchestrator()
        console.print(
            "[bold cyan]Wake-Word Engine Active:[/bold cyan] Listening for [bold green]'Jarvis'[/bold green] or [bold green]'Hey Jarvis'[/bold green]..."
        )
        console.print("[dim]Speak 'Jarvis' followed by your instruction. Press Ctrl+C to terminate.[/dim]\n")

        def _on_wake(phrase: str):
            console.print(f"[bold yellow]Wake keyword spotted:[/bold yellow] '{phrase}'! Listening for directive...")

        def _on_directive(directive: str):
            console.print(f'\n[bold green]Captured Directive:[/bold green] "{directive}"')
            with console.status("[bold cyan]Executing directive with J.A.R.V.I.S....[/bold cyan]", spinner="dots"):
                res = orchestrator.run(user_input=directive, chat_history=[])
            console.print(
                Panel(Markdown(res.get("output", "")), title="[bold green]J.A.R.V.I.S. Response[/bold green]")
            )

        listener.on_wake_detected = _on_wake
        listener.start_listening(on_directive=_on_directive)

        try:
            while listener.is_listening():
                time.sleep(0.5)
        except KeyboardInterrupt:
            listener.stop_listening()
            console.print("\n[yellow]Wake-word listener deactivated.[/yellow]")
        return

    from src.modules.voice.voice_bridge import transcribe_voice_directive

    if audio_file:
        console.print(f"[bold cyan]Voice Intelligence Engine:[/bold cyan] Transcribing {audio_file}...")
        with console.status("[bold cyan]Running Faster-Whisper transcription...[/bold cyan]", spinner="dots"):
            trans_res = transcribe_voice_directive(audio_file)
            transcript = trans_res.get("text", "") if isinstance(trans_res, dict) else str(trans_res)
    else:
        console.print("[bold yellow]Microphone recording:[/bold yellow] Press Enter to finish dictation...")
        # Fallback microphone prompt
        transcript = Prompt.ask("[bold green]Voice Directive[/bold green]")

    console.print(f'[bold green]Transcribed Directive:[/bold green] "{transcript}"\n')
    if Confirm.ask("[bold cyan]Dispatch this directive to J.A.R.V.I.S. now?[/bold cyan]", default=True):
        from src.core.orchestrator import JarvisOrchestrator

        orchestrator = JarvisOrchestrator()
        with console.status("[bold cyan]Executing directive...[/bold cyan]", spinner="dots"):
            res = orchestrator.run(user_input=transcript, chat_history=[])
        console.print(Panel(Markdown(res.get("output", "")), title="[bold green]J.A.R.V.I.S. Response[/bold green]"))


# ----------------------------------------------------------------------
# Command: Omnichannel Messaging Gateway
# ----------------------------------------------------------------------
@cli.command("gateway")
@click.option("--telegram", is_flag=True, default=True, help="Launch Telegram Omnichannel Gateway.")
@click.option("--token", default=None, help="Telegram Bot Token override (or TELEGRAM_BOT_TOKEN env var).")
def gateway_cmd(telegram: bool, token: Optional[str]):
    """Launch the Telegram Omnichannel Gateway for mobile and remote assistant control."""
    from src.modules.gateway.telegram_bridge import get_telegram_gateway

    gw = get_telegram_gateway(bot_token=token)
    if not gw.is_configured:
        console.print(
            "[bold red]Telegram Gateway Error:[/bold red] No TELEGRAM_BOT_TOKEN provided or configured in .env."
        )
        console.print(
            "[dim]Set TELEGRAM_BOT_TOKEN in your environment or run: jarvis gateway --token <your_token>[/dim]"
        )
        return

    console.print("[bold cyan]Omnichannel Gateway:[/bold cyan] Verifying bot credentials...")
    info = gw.get_me()
    if not info.get("ok"):
        console.print(
            f"[bold red]Telegram Gateway Connection Failed:[/bold red] {info.get('description', 'Unknown error')}"
        )
        return

    bot_user = info.get("result", {}).get("username", "UnknownBot")
    console.print(f"[bold green][OK] Connected as @{bot_user}[/bold green]")
    console.print(
        "[dim]Long-polling active. Send voice memos, /goal, or chat messages to your Telegram bot. Press Ctrl+C to stop.[/dim]\n"
    )

    try:
        gw.poll_forever()
    except KeyboardInterrupt:
        gw.stop()
        console.print("\n[yellow]Telegram Gateway halted.[/yellow]")


# ----------------------------------------------------------------------
# Command: Workspace Explorer
# ----------------------------------------------------------------------
@cli.command("workspace")
@click.option("--clean", is_flag=True, help="Purge temporary audit spreadsheets.")
def workspace_cmd(clean: bool):
    """Inspect and manage deliverable files generated in workspace/."""
    from src.assistant.workspace_tools import WORKSPACE_DIR

    if clean:
        count = 0
        for f in WORKSPACE_DIR.glob("outreach_audit_camp_*.xlsx"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        console.print(f"[green][OK] Cleaned {count} transient audit file(s).[/green]")

    _render_workspace_table()


def _render_workspace_table():
    """Render a clean Rich table of workspace files."""
    from src.assistant.workspace_tools import WORKSPACE_DIR

    if not WORKSPACE_DIR.exists():
        console.print("[dim]Workspace directory does not exist.[/dim]")
        return

    files = [f for f in WORKSPACE_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"]
    if not files:
        console.print("[dim]No deliverable files currently in workspace/.[/dim]\n")
        return

    table = Table(
        title="[bold cyan]Workspace Deliverables (workspace/)[/bold cyan]", show_header=True, header_style="bold yellow"
    )
    table.add_column("Filename", style="bold white")
    table.add_column("Size", style="cyan", width=12)
    table.add_column("Type", style="green", width=10)
    table.add_column("Last Modified", style="magenta")

    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        sz = f.stat().st_size
        size_str = f"{sz / 1024:.1f} KB" if sz > 1024 else f"{sz} B"
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
        table.add_row(f.name, size_str, f.suffix.lstrip(".").upper() or "FILE", mod_time)

    console.print(table)
    console.print()


if __name__ == "__main__":
    cli()
