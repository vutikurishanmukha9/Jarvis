"""
Closed-Loop Procedural Skill Learner for Auto-JARVIS.
Inspired by Hermes Agent's closed learning loop (skills_tool.py & skill_manager_tool.py).
Automatically extracts procedural workflows from completed DAG missions and persists
them as reusable skills adhering to the agentskills.io open standard.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = Path("workspace/skills")
SKILLS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = SKILLS_DIR / "registry.json"


class SkillLearner:
    """Extracts, lints, and registers procedural skills from successful goal executions."""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.skills_dir / "registry.json"
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        """Ensure registry.json exists."""
        if not self.registry_file.exists():
            try:
                self.registry_file.write_text("{}", encoding="utf-8")
            except Exception as e:
                logger.debug("Failed to initialize skills registry: %s", e)

    def learn_from_completed_plan(
        self,
        goal: str,
        plan: Dict[str, Any],
        final_synthesis: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Distill a successful goal execution into an agentskills.io skill bundle.
        """
        tasks = plan.get("tasks", [])
        if not tasks or len(tasks) < 2:
            # Only synthesize skills from non-trivial multi-step missions
            return None

        # Generate a clean slug for the skill
        slug = self._generate_slug(goal)
        skill_dir = self.skills_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Collect unique tools and steps
        tools_used = sorted(list({t.get("tool_hint", "general") for t in tasks if t.get("tool_hint")}))
        steps_doc = []
        for i, t in enumerate(tasks):
            t_title = t.get("title", f"Step {i + 1}")
            t_inst = t.get("instruction", "")
            t_deliv = t.get("expected_deliverable", "")
            steps_doc.append(f"{i + 1}. **{t_title}**\n   - Directive: {t_inst}\n   - Expected Output: {t_deliv}")

        steps_markdown = "\n".join(steps_doc)
        summary = plan.get("goal_summary", goal)

        # Construct agentskills.io YAML frontmatter + Markdown
        skill_content = f"""---
name: {slug}
description: {summary[:120]}
tools: {tools_used}
version: 1.0.0
created_at: {time.strftime("%Y-%m-%d %H:%M:%S")}
---

# {goal}

## Purpose
Automated procedure synthesized from a successful J.A.R.V.I.S. autonomous execution.

## Execution Procedure
{steps_markdown}

## Expected Deliverables
{final_synthesis[:400] if final_synthesis else "Structured artifacts in workspace/."}
"""

        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_content, encoding="utf-8")

        # Update registry
        registry = self.get_registered_skills()
        try:
            rel_path = str(skill_file.relative_to(Path(".")))
        except ValueError:
            rel_path = str(skill_file)

        skill_meta = {
            "name": slug,
            "goal": goal,
            "summary": summary,
            "tools": tools_used,
            "steps_count": len(tasks),
            "path": rel_path,
            "created_at": time.time(),
        }
        registry[slug] = skill_meta

        try:
            self.registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to update skill registry: %s", e)

        logger.info("Successfully learned new skill '%s' into %s", slug, skill_file)
        return skill_meta

    def get_registered_skills(self) -> Dict[str, Any]:
        """Read all registered skills from the registry."""
        if not self.registry_file.exists():
            return {}
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def find_matching_skill(self, goal: str) -> Optional[Dict[str, Any]]:
        """Check if an existing skill matches the incoming goal prompt."""
        skills = self.get_registered_skills()
        clean_goal = goal.lower()
        for name, meta in skills.items():
            if name.replace("-", " ") in clean_goal or meta.get("goal", "").lower() in clean_goal:
                return meta
        return None

    def _generate_slug(self, text: str) -> str:
        """Create a filesystem-friendly slug."""
        clean = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
        slug = re.sub(r"[\s_-]+", "-", clean)[:40].strip("-")
        return slug or f"skill-{int(time.time())}"


_GLOBAL_SKILL_LEARNER: Optional[SkillLearner] = None


def get_skill_learner() -> SkillLearner:
    """Return the global SkillLearner singleton."""
    global _GLOBAL_SKILL_LEARNER
    if _GLOBAL_SKILL_LEARNER is None:
        _GLOBAL_SKILL_LEARNER = SkillLearner()
    return _GLOBAL_SKILL_LEARNER
