"""
Autonomous Goal Decomposition Engine for Auto-JARVIS.
Deconstructs complex user instructions into structured subtask DAGs.
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
Your task is to take any high-level human goal and decompose it into a clean, logical, sequential plan of 2 to 6 actionable subtasks.

Available Tool Capabilities of the Agent:
1. Web Search & Wikipedia research (DuckDuckGo, Wikipedia, Web Scraping)
2. Document Search (Vector RAG on uploaded files)
3. Vision Intelligence (YOLOv8 object detection, OCR text reading, image analysis)
4. Python Execution Sandbox (math calculations, statistical analysis, Matplotlib & Plotly plot generation)
5. Workspace Operations (create/read files, generate Excel spreadsheets, generate Word documents, write Markdown reports)

Rules for Subtask Decomposition:
- Break the goal into logical stages (e.g., Stage 1: Information Gathering / Extraction, Stage 2: Data Processing / Computation, Stage 3: Artifact & Report Creation).
- Each subtask must have a clear, distinct purpose.
- Order them so dependent outputs feed into subsequent steps.
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
      "expected_deliverable": "Description of output (e.g., Raw facts or summary text)"
    },
    {
      "id": "task_2",
      "title": "Short title (e.g., Generate Summary Report & Save to Workspace)",
      "instruction": "Concrete detailed instruction for synthesizing data into a workspace file.",
      "tool_hint": "workspace",
      "expected_deliverable": "File created in workspace (e.g. quantum_brief.md)"
    }
  ]
}
"""

class GoalPlanner:
    """Plans and decomposes high-level user instructions into actionable subtask DAGs."""

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
        Deconstruct a high-level user goal into an executable subtask plan.
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

            # Ensure tasks list is bounded and initialized
            tasks = plan_json.get("tasks", [])[:MAX_AUTONOMOUS_SUBTASKS]
            for idx, t in enumerate(tasks):
                t["status"] = "pending"
                t["result"] = ""
                t["attempts"] = 0
                if "id" not in t:
                    t["id"] = f"task_{idx+1}"

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
                        "status": "pending",
                        "result": "",
                        "attempts": 0
                    }
                ]
            }
