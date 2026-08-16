"""
Autonomous Multi-Step Execution Runner for Auto-JARVIS.
Executes subtask DAGs sequentially, performs error self-correction, and streams progress telemetry.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from langchain_core.messages import HumanMessage

from ..config import MAX_RETRY_PER_TASK, WORKSPACE_DIR
from ..core.orchestrator import JarvisOrchestrator
from .workspace_tools import get_workspace_tools
from .profile_manager import ProfileManager

logger = logging.getLogger(__name__)

class AutonomousRunner:
    """Executes a decomposed goal plan with continuous reflection, error recovery, and tool execution."""

    def __init__(
        self,
        orchestrator: Optional[JarvisOrchestrator] = None,
        step_callback: Optional[Callable[[Dict[str, Any], str, str], None]] = None
    ):
        self.orchestrator = orchestrator
        self.step_callback = step_callback
        
        if self.orchestrator:
            # Ensure orchestrator has workspace tools
            ws_tools = get_workspace_tools()
            for t in ws_tools:
                if t.name not in [x.name for x in self.orchestrator.tools]:
                    self.orchestrator.tools.append(t)
            # Rebuild executor with all tools
            self.orchestrator.agent_executor = self.orchestrator._build_executor()

    def execute_plan(self, plan: Dict[str, Any], initial_goal: str) -> Dict[str, Any]:
        """
        Execute all subtasks in a goal plan sequentially, carrying forward context and self-correcting on error.
        """
        tasks: List[Dict[str, Any]] = plan.get("tasks", [])
        completed_results: List[str] = []
        all_steps: List[Dict[str, Any]] = []
        all_figures: List[Any] = []
        all_annotated_imgs: List[Any] = []

        total_tasks = len(tasks)
        plan["status"] = "running"

        for idx, task in enumerate(tasks):
            task_id = task.get("id", f"task_{idx+1}")
            task_title = task.get("title", task.get("description", f"Subtask {idx+1}"))
            task_instruction = task.get("instruction", task.get("description", ""))
            task["status"] = "in_progress"
            task["attempts"] = 1

            if self.step_callback:
                self.step_callback(plan, task_id, f"Executing Subtask {idx+1}/{total_tasks}: {task_title}")

            # If no orchestrator is attached, perform simulated execution
            if not self.orchestrator:
                task["status"] = "completed"
                task_result_text = f"Simulated autonomous completion of: {task_title}"
                task["result"] = task_result_text
                completed_results.append(f"Subtask '{task_title}' Deliverable:\n{task_result_text}")
                continue

            # Build enriched contextual prompt for this subtask
            context_history = ""
            if completed_results:
                context_history = "\n\nPREVIOUS SUBTASK OUTPUTS:\n" + "\n---\n".join(completed_results)

            subtask_prompt = (
                f"[AUTONOMOUS MISSION: '{initial_goal}']\n"
                f"CURRENT SUBTASK ({idx+1}/{total_tasks}): {task_title}\n"
                f"SPECIFIC INSTRUCTION: {task_instruction}\n"
                f"EXPECTED DELIVERABLE: {task.get('expected_deliverable', 'Actionable result')}\n"
                f"{context_history}\n\n"
                f"Execute the appropriate tools autonomously. If creating files, save them into the workspace. "
                f"Be precise and comprehensive."
            )

            # Execution with self-correction retry loop
            success = False
            task_result_text = ""
            for attempt in range(1, MAX_RETRY_PER_TASK + 1):
                try:
                    logger.info(f"Running subtask {task_id} (Attempt {attempt}/{MAX_RETRY_PER_TASK})")
                    run_res = self.orchestrator.run(
                        user_input=subtask_prompt if attempt == 1 else (
                            f"[SELF-CORRECTION RETRY {attempt}]: Previous attempt encountered an issue. "
                            f"Analyze and fix: {subtask_prompt}"
                        ),
                        chat_history=[]
                    )
                    
                    task_result_text = run_res.get("output", "")
                    all_steps.extend(run_res.get("steps", []))
                    all_figures.extend(run_res.get("figures", []))
                    all_annotated_imgs.extend(run_res.get("annotated_images", []))

                    # Check for explicit failure markers
                    if "An error occurred during agent processing" in task_result_text:
                        raise RuntimeError(task_result_text)

                    success = True
                    task["status"] = "completed"
                    task["result"] = task_result_text
                    completed_results.append(f"Subtask '{task_title}' Deliverable:\n{task_result_text}")
                    break

                except Exception as e:
                    logger.warning(f"Subtask {task_id} attempt {attempt} failed: {str(e)}")
                    task["attempts"] = attempt
                    if attempt < MAX_RETRY_PER_TASK:
                        if self.step_callback:
                            self.step_callback(plan, task_id, f"Error in '{task_title}'. Self-correcting (Attempt {attempt+1})...")
                        time.sleep(1.0)
                    else:
                        task["status"] = "failed"
                        task["result"] = f"Failed after {MAX_RETRY_PER_TASK} attempts: {str(e)}"
                        completed_results.append(f"Subtask '{task_title}' failed with error: {str(e)}")

            if self.step_callback:
                self.step_callback(plan, task_id, f"Finished Subtask {idx+1}/{total_tasks}: {task['status'].upper()}")

        # Final synthesis
        plan["status"] = "completed"
        final_summary = self._generate_mission_summary(initial_goal, plan, completed_results)

        return {
            "plan": plan,
            "final_summary": final_summary,
            "steps": all_steps,
            "figures": all_figures,
            "annotated_images": all_annotated_imgs
        }

    def _generate_mission_summary(
        self,
        goal: str,
        plan: Dict[str, Any],
        completed_results: List[str]
    ) -> str:
        """Compile a structured final mission briefing."""
        profile = ProfileManager.load_profile()
        user_name = profile.get("user_name", "Boss")
        
        lines = [
            f"# Autonomous Mission Complete",
            f"**Assigned Goal**: {goal}",
            f"**Executed For**: {user_name}",
            f"**Completed Steps**: {len([t for t in plan.get('tasks', []) if t.get('status') == 'completed'])} / {len(plan.get('tasks', []))}",
            "---",
            "### Executive Summary of Deliverables",
            ""
        ]

        for t in plan.get("tasks", []):
            status_tag = "[COMPLETED]" if t.get("status") == "completed" else "[FAILED]"
            lines.append(f"#### {status_tag} {t.get('title')}")
            res_snippet = t.get("result", "").strip()
            lines.append(res_snippet)
            lines.append("")

        # Check for created files in workspace
        try:
            ws_files = list(WORKSPACE_DIR.glob("*"))
            if ws_files:
                lines.append("---")
                lines.append("### Workspace Files Generated on Your Behalf")
                for f in sorted(ws_files):
                    if f.is_file():
                        size_kb = round(f.stat().st_size / 1024, 2)
                        lines.append(f"- **`{f.name}`** ({size_kb} KB)")
        except Exception:
            pass

        return "\n".join(lines)
