"""
Autonomous Goal Decomposition Engine for Auto-JARVIS.
Deconstructs complex user instructions into structured subtask plans with dependency declarations.
Supports topological ordering for dependency-aware execution.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import PROVIDERS, MAX_AUTONOMOUS_SUBTASKS
from .profile_manager import ProfileManager

logger = logging.getLogger(__name__)

PLANNING_SYSTEM_PROMPT = """
You are the Autonomous Goal Planning Architect for Auto-JARVIS, a Super-Intelligent Personal Assistant.
Your task is to take any high-level human goal and decompose it into a clean, logical plan of 2 to 6 actionable subtasks with explicit dependency declarations.

Available Tool Capabilities of the Agent:
1. Web Search & Wikipedia research (DuckDuckGo, Wikipedia, Web Scraping)
2. Document Search (Vector RAG on uploaded files)
3. Vision Intelligence (YOLOv8 object detection, OCR text reading, image analysis)
4. Python Execution (math calculations, statistical analysis, Matplotlib & Plotly plot generation)
5. Workspace Operations (create/read files, generate Excel spreadsheets, generate Word documents, write Markdown reports)

Rules for Subtask Decomposition:
- Break the goal into logical stages (e.g., Stage 1: Information Gathering, Stage 2: Data Processing, Stage 3: Artifact Creation).
- Each subtask must have a clear, distinct purpose.
- Declare dependencies explicitly using "depends_on" — a list of task IDs whose outputs this task requires.
- Tasks with no dependencies can potentially run in parallel.
- You MUST respond ONLY with a valid JSON object matching the schema below:

{
  "goal_summary": "Short 1-sentence summary of overall mission",
  "estimated_steps": 3,
  "tasks": [
    {
      "id": "task_1",
      "title": "Short title (e.g., Web Research on Quantum Computing)",
      "instruction": "Concrete detailed instruction for what the agent should investigate or execute.",
      "tool_hint": "web_search / python / workspace / vision / rag",
      "expected_deliverable": "Description of output (e.g., Raw facts or summary text)",
      "depends_on": []
    },
    {
      "id": "task_2",
      "title": "Short title (e.g., Generate Summary Report & Save to Workspace)",
      "instruction": "Concrete detailed instruction for synthesizing data into a workspace file.",
      "tool_hint": "workspace",
      "expected_deliverable": "File created in workspace (e.g. quantum_brief.md)",
      "depends_on": ["task_1"]
    }
  ]
}
"""


def topological_sort(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform a topological sort on tasks using Kahn's algorithm.
    Tasks are ordered so that dependencies are always executed before dependents.
    
    If cycles are detected, falls back to the original task order.
    
    Args:
        tasks: List of task dicts, each with "id" and "depends_on" fields.
        
    Returns:
        Tasks reordered in valid topological (dependency-respecting) order.
    """
    if not tasks:
        return tasks

    # Build adjacency and in-degree maps
    task_map = {t["id"]: t for t in tasks}
    in_degree = {t["id"]: 0 for t in tasks}
    dependents = {t["id"]: [] for t in tasks}  # id -> list of tasks that depend on it

    for task in tasks:
        deps = task.get("depends_on", [])
        for dep_id in deps:
            if dep_id in task_map:
                in_degree[task["id"]] += 1
                dependents[dep_id].append(task["id"])

    # Kahn's algorithm
    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    sorted_ids = []

    while queue:
        # Sort queue for deterministic ordering (alphabetical by id)
        queue.sort()
        current = queue.pop(0)
        sorted_ids.append(current)

        for dependent_id in dependents[current]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)

    # Cycle detection: if not all tasks are sorted, there's a cycle
    if len(sorted_ids) != len(tasks):
        logger.warning(
            f"Cycle detected in task dependencies. "
            f"Sorted {len(sorted_ids)}/{len(tasks)} tasks. Falling back to original order."
        )
        return tasks

    return [task_map[tid] for tid in sorted_ids]


class GoalPlanner:
    """Plans and decomposes high-level user instructions into dependency-aware subtask plans."""

    def __init__(
        self,
        api_provider: str = "OpenRouter",
        api_key: str = "",
        model_name: str = "openai/gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.api_provider = api_provider
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url or PROVIDERS.get(api_provider, {}).get("base_url")
        self.temperature = temperature
        self.llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI:
        kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_retries": 2,
            "timeout": 45
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    def plan_goal(self, goal: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Deconstruct a high-level user goal into an executable subtask plan
        with dependency declarations and topological ordering.
        """
        user_prompt = f"User Goal to Decompose:\n\"\"\"{goal}\"\"\""
        if context:
            user_prompt += f"\n\nAdditional Context / Uploaded Data:\n\"\"\"{context}\"\"\""

        messages = [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content.strip()

            # Extract JSON block
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group(1))
            else:
                # Direct JSON parse
                plan_json = json.loads(raw_text)

            # Ensure tasks list is bounded, schema-validated, and initialized
            raw_tasks = plan_json.get("tasks", [])[:MAX_AUTONOMOUS_SUBTASKS]
            validated_tasks = []
            for idx, t in enumerate(raw_tasks):
                # Normalize keys for robust schema compatibility
                task_id = str(t.get("id") or f"task_{idx+1}")
                task_title = str(t.get("title") or f"Execute Subtask {idx+1}")
                task_instruction = str(t.get("instruction") or task_title)
                tool_hint = str(t.get("tool_hint") or t.get("tool") or "general_assistant")
                deliverable = str(t.get("expected_deliverable") or t.get("deliverable") or "Deliverable")
                raw_deps = t.get("depends_on", [])
                deps = [str(d) for d in raw_deps] if isinstance(raw_deps, list) else []

                subtask_entry = {
                    "id": task_id,
                    "title": task_title,
                    "instruction": task_instruction,
                    "tool_hint": tool_hint,
                    "expected_deliverable": deliverable,
                    "depends_on": deps,
                    "status": "pending",
                    "result": "",
                    "attempts": 0
                }
                validated_tasks.append(subtask_entry)

            # Apply topological sort for dependency-aware execution order
            tasks = topological_sort(validated_tasks)
            plan_json["tasks"] = tasks
            plan_json["status"] = "planned"
            return plan_json

        except Exception as e:
            logger.error(f"Goal planning error: {str(e)}", exc_info=True)
            # Fallback single/dual-step plan
            return {
                "goal_summary": goal[:80] + "...",
                "estimated_steps": 2,
                "status": "planned",
                "tasks": [
                    {
                        "id": "task_1",
                        "title": "Analyze and Gather Information",
                        "instruction": f"Perform comprehensive research and analysis to fulfill the goal: {goal}",
                        "tool_hint": "web_search / python / rag",
                        "expected_deliverable": "Findings and intermediate data",
                        "depends_on": [],
                        "status": "pending",
                        "result": "",
                        "attempts": 0
                    },
                    {
                        "id": "task_2",
                        "title": "Synthesize & Deliver Artifacts",
                        "instruction": f"Compile final deliverables and write workspace files for: {goal}",
                        "tool_hint": "workspace",
                        "expected_deliverable": "Workspace report and final answer",
                        "depends_on": ["task_1"],
                        "status": "pending",
                        "result": "",
                        "attempts": 0
                    }
                ]
            }
