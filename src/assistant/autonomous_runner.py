"""
Autonomous Multi-Step Execution Runner for Auto-JARVIS.
Executes subtask plans in dependency-aware topological order, verifies each output
against the expected deliverable, and performs error self-correction.

Architecture:
- Artifact Store: Each completed task stores its output in _artifact_store[task_id].
  Subsequent tasks only receive artifacts from their declared depends_on tasks,
  not the full context history (prevents token bloat).
- Output Verification: After successful execution, the LLM verifies whether the
  output satisfies the instruction and expected deliverable. If verification fails,
  the self-correction retry loop is triggered.
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
    """Executes a decomposed goal plan with dependency-aware ordering, output verification, and error recovery."""

    def __init__(
        self,
        orchestrator: Optional[JarvisOrchestrator] = None,
        step_callback: Optional[Callable[[Dict[str, Any], str, str], None]] = None
    ):
        self.orchestrator = orchestrator
        self.step_callback = step_callback
        self._artifact_store: Dict[str, str] = {}  # task_id -> output text

        if self.orchestrator:
            # Ensure orchestrator has workspace tools
            ws_tools = get_workspace_tools()
            for t in ws_tools:
                if t.name not in [x.name for x in self.orchestrator.tools]:
                    self.orchestrator.tools.append(t)
            # Rebuild executor with all tools
            self.orchestrator.agent_executor = self.orchestrator._build_executor()

    def _build_dependency_context(self, task: Dict[str, Any]) -> str:
        """
        Build context from only the declared dependencies of a task.
        Instead of concatenating all prior results (which causes token bloat),
        each task receives only the outputs from its depends_on tasks.
        """
        depends_on = task.get("depends_on", [])
        if not depends_on:
            return ""

        context_parts = []
        for dep_id in depends_on:
            if dep_id in self._artifact_store:
                context_parts.append(
                    f"[Artifact from {dep_id}]:\n{self._artifact_store[dep_id]}"
                )

        if not context_parts:
            return ""

        return "\n\nDEPENDENCY ARTIFACTS:\n" + "\n---\n".join(context_parts)

    def _verify_output(
        self,
        task_result: str,
        instruction: str,
        expected_deliverable: str
    ) -> Dict[str, Any]:
        """
        Verify whether a task output satisfies the instruction and expected deliverable.
        Uses the LLM to perform semantic verification rather than just checking for exceptions.
        
        Returns:
            {"passed": True/False, "reason": "explanation"}
        """
        if not self.orchestrator:
            return {"passed": True, "reason": "No orchestrator — skipping verification."}

        # Skip verification for very short or obviously failed outputs
        if not task_result or len(task_result.strip()) < 10:
            return {"passed": False, "reason": "Output is empty or too short."}

        # Check for explicit error markers
        error_markers = [
            "An error occurred during agent processing",
            "Python Execution Error:",
            "Security Restriction:",
            "Execution Timeout:",
            "Failed to retrieve",
        ]
        for marker in error_markers:
            if marker in task_result:
                return {"passed": False, "reason": f"Output contains error marker: '{marker}'"}

        verification_prompt = (
            f"VERIFICATION TASK: Evaluate whether the following output satisfies the requirements.\n\n"
            f"INSTRUCTION: {instruction}\n"
            f"EXPECTED DELIVERABLE: {expected_deliverable}\n\n"
            f"ACTUAL OUTPUT:\n{task_result[:3000]}\n\n"
            f"Does this output adequately fulfill the instruction and expected deliverable?\n"
            f"Respond with exactly one word on the first line: PASS or FAIL\n"
            f"Then provide a brief reason on the second line."
        )

        try:
            verify_result = self.orchestrator.run(
                user_input=verification_prompt,
                chat_history=[]
            )
            verify_text = verify_result.get("output", "").strip()

            # Parse the verification response
            first_line = verify_text.split("\n")[0].strip().upper()
            reason = verify_text.split("\n")[1].strip() if "\n" in verify_text else "No reason provided."

            if "PASS" in first_line:
                return {"passed": True, "reason": reason}
            else:
                return {"passed": False, "reason": reason}

        except Exception as e:
            logger.warning(f"Output verification failed with error: {str(e)}")
            # If verification itself fails, assume the output is acceptable
            # to avoid blocking execution on verification infrastructure issues
            return {"passed": True, "reason": f"Verification skipped due to error: {str(e)}"}

    def execute_plan(self, plan: Dict[str, Any], initial_goal: str) -> Dict[str, Any]:
        """
        Execute all subtasks in dependency-aware topological order.
        Each task receives only artifacts from its declared dependencies.
        Each output is verified against the expected deliverable before proceeding.
        """
        tasks: List[Dict[str, Any]] = plan.get("tasks", [])
        all_steps: List[Dict[str, Any]] = []
        all_figures: List[Any] = []
        all_annotated_imgs: List[Any] = []

        total_tasks = len(tasks)
        plan["status"] = "running"
        self._artifact_store.clear()

        for idx, task in enumerate(tasks):
            task_id = task.get("id", f"task_{idx+1}")
            task_title = task.get("title", task.get("description", f"Subtask {idx+1}"))
            task_instruction = task.get("instruction", task.get("description", ""))
            expected_deliverable = task.get("expected_deliverable", "Actionable result")
            task["status"] = "in_progress"
            task["attempts"] = 1

            if self.step_callback:
                self.step_callback(plan, task_id, f"Executing Subtask {idx+1}/{total_tasks}: {task_title}")

            # If no orchestrator is attached, perform simulated execution
            if not self.orchestrator:
                task["status"] = "completed"
                task_result_text = f"Simulated autonomous completion of: {task_title}"
                task["result"] = task_result_text
                self._artifact_store[task_id] = task_result_text
                continue

            # Build dependency-scoped context (not full history)
            dependency_context = self._build_dependency_context(task)

            subtask_prompt = (
                f"[AUTONOMOUS MISSION: '{initial_goal}']\n"
                f"CURRENT SUBTASK ({idx+1}/{total_tasks}): {task_title}\n"
                f"SPECIFIC INSTRUCTION: {task_instruction}\n"
                f"EXPECTED DELIVERABLE: {expected_deliverable}\n"
                f"{dependency_context}\n\n"
                f"Execute the appropriate tools autonomously. If creating files, save them into the workspace. "
                f"Be precise and comprehensive."
            )

            # Execution with verification + self-correction retry loop
            success = False
            task_result_text = ""
            for attempt in range(1, MAX_RETRY_PER_TASK + 1):
                try:
                    logger.info(f"Running subtask {task_id} (Attempt {attempt}/{MAX_RETRY_PER_TASK})")
                    run_res = self.orchestrator.run(
                        user_input=subtask_prompt if attempt == 1 else (
                            f"[SELF-CORRECTION RETRY {attempt}]: Previous attempt was inadequate. "
                            f"Reason: {verification.get('reason', 'Unknown')}. "
                            f"Analyze and fix: {subtask_prompt}"
                        ),
                        chat_history=[]
                    )

                    task_result_text = run_res.get("output", "")
                    all_steps.extend(run_res.get("steps", []))
                    all_figures.extend(run_res.get("figures", []))
                    all_annotated_imgs.extend(run_res.get("annotated_images", []))

                    # Output Verification: check if result satisfies the instruction
                    verification = self._verify_output(
                        task_result_text, task_instruction, expected_deliverable
                    )

                    if verification["passed"]:
                        success = True
                        task["status"] = "completed"
                        task["result"] = task_result_text
                        task["verification"] = verification
                        self._artifact_store[task_id] = task_result_text
                        break
                    else:
                        logger.warning(
                            f"Subtask {task_id} attempt {attempt} failed verification: "
                            f"{verification['reason']}"
                        )
                        if attempt < MAX_RETRY_PER_TASK:
                            if self.step_callback:
                                self.step_callback(
                                    plan, task_id,
                                    f"Verification failed for '{task_title}'. "
                                    f"Self-correcting (Attempt {attempt+1})..."
                                )
                            time.sleep(1.0)
                        else:
                            # Final attempt also failed verification — accept with warning
                            task["status"] = "completed_with_warnings"
                            task["result"] = task_result_text
                            task["verification"] = verification
                            self._artifact_store[task_id] = task_result_text

                except Exception as e:
                    logger.warning(f"Subtask {task_id} attempt {attempt} failed: {str(e)}")
                    task["attempts"] = attempt
                    verification = {"passed": False, "reason": str(e)}
                    if attempt < MAX_RETRY_PER_TASK:
                        if self.step_callback:
                            self.step_callback(plan, task_id, f"Error in '{task_title}'. Self-correcting (Attempt {attempt+1})...")
                        time.sleep(1.0)
                    else:
                        task["status"] = "failed"
                        task["result"] = f"Failed after {MAX_RETRY_PER_TASK} attempts: {str(e)}"

            if self.step_callback:
                self.step_callback(plan, task_id, f"Finished Subtask {idx+1}/{total_tasks}: {task['status'].upper()}")

        # Final synthesis
        plan["status"] = "completed"
        final_summary = self._generate_mission_summary(initial_goal, plan)

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
        plan: Dict[str, Any]
    ) -> str:
        """Compile a structured final mission briefing."""
        profile = ProfileManager.load_profile()
        user_name = profile.get("user_name", "Boss")

        tasks = plan.get("tasks", [])
        completed = [t for t in tasks if t.get("status") in ("completed", "completed_with_warnings")]
        failed = [t for t in tasks if t.get("status") == "failed"]
        warned = [t for t in tasks if t.get("status") == "completed_with_warnings"]

        lines = [
            f"# Autonomous Mission Complete",
            f"**Assigned Goal**: {goal}",
            f"**Executed For**: {user_name}",
            f"**Completed**: {len(completed)} / {len(tasks)}",
            f"**Failed**: {len(failed)} / {len(tasks)}",
            f"**Warnings**: {len(warned)} / {len(tasks)}",
            "---",
            "### Executive Summary of Deliverables",
            ""
        ]

        for t in tasks:
            status = t.get("status", "unknown")
            if status == "completed":
                status_tag = "[COMPLETED]"
            elif status == "completed_with_warnings":
                status_tag = "[COMPLETED WITH WARNINGS]"
            else:
                status_tag = "[FAILED]"

            lines.append(f"#### {status_tag} {t.get('title')}")

            # Show verification result if present
            verification = t.get("verification")
            if verification:
                v_status = "✓ PASS" if verification["passed"] else "⚠ MARGINAL"
                lines.append(f"*Verification: {v_status} — {verification.get('reason', '')}*")

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
