"""
Telegram Remote Gateway Bridge for J.A.R.V.I.S.
Enables smartphone remote control, voice memo processing, goal dispatching,
and deliverable file uploads via Telegram Bot API using httpx.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.assistant.autonomous_runner import AutonomousRunner
from src.assistant.goal_planner import GoalPlanner
from src.assistant.workspace_tools import WORKSPACE_DIR
from src.core.orchestrator import JarvisOrchestrator
from src.modules.voice.voice_bridge import transcribe_voice_directive

logger = logging.getLogger(__name__)


class TelegramGateway:
    """Manages Telegram Bot polling, inbound dispatching, and file delivery for J.A.R.V.I.S."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        orchestrator: Optional[JarvisOrchestrator] = None,
        allowed_chat_ids: Optional[List[int]] = None,
    ):
        if bot_token is not None:
            self.bot_token = bot_token.strip()
        else:
            self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self._orchestrator = orchestrator
        self.allowed_chat_ids = allowed_chat_ids or []
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update_id = 0
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    @property
    def orchestrator(self) -> JarvisOrchestrator:
        """Lazily initialize the JarvisOrchestrator when needed."""
        if self._orchestrator is None:
            self._orchestrator = JarvisOrchestrator()
        return self._orchestrator

    @orchestrator.setter
    def orchestrator(self, val: JarvisOrchestrator) -> None:
        self._orchestrator = val

    @property
    def is_configured(self) -> bool:
        """Check if a valid Telegram bot token is configured."""
        return bool(self.bot_token and len(self.bot_token) > 10)

    def is_active(self) -> bool:
        """Return whether the polling loop is currently active."""
        return self._is_running and self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the background polling loop."""
        if not self.is_configured:
            logger.warning("TelegramGateway: TELEGRAM_BOT_TOKEN is not set.")
            return False

        if self.is_active():
            logger.info("TelegramGateway is already running.")
            return True

        self._is_running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="TelegramGatewayThread")
        self._thread.start()
        logger.info("TelegramGateway started polling.")
        return True

    def get_me(self) -> Dict[str, Any]:
        """Fetch bot metadata from getMe endpoint."""
        if not self.is_configured:
            return {"ok": False, "description": "Token not configured."}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/getMe")
                return resp.json()
        except Exception as e:
            return {"ok": False, "description": str(e)}

    def poll_forever(self) -> None:
        """Run the polling loop synchronously on the calling thread."""
        self._is_running = True
        self._poll_loop()

    def stop(self) -> None:
        """Stop the background polling loop."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("TelegramGateway stopped.")

    def _poll_loop(self) -> None:
        """Long-polling update loop for Telegram Bot API."""
        with httpx.Client(timeout=35.0) as client:
            while self._is_running:
                try:
                    params = {"offset": self._last_update_id + 1, "timeout": 25}
                    resp = client.get(f"{self.base_url}/getUpdates", params=params)
                    if resp.status_code != 200:
                        logger.debug("Telegram polling status %d: %s", resp.status_code, resp.text)
                        time.sleep(3.0)
                        continue

                    data = resp.json()
                    if not data.get("ok"):
                        time.sleep(3.0)
                        continue

                    updates = data.get("result", [])
                    for update in updates:
                        update_id = update.get("update_id", 0)
                        if update_id > self._last_update_id:
                            self._last_update_id = update_id

                        self._handle_update(client, update)

                except Exception as e:
                    if self._is_running:
                        logger.debug("Telegram polling exception: %s", e)
                        time.sleep(2.0)

    def _handle_update(self, client: httpx.Client, update: Dict[str, Any]) -> None:
        """Route incoming message, voice note, or command."""
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat = message.get("chat", {})
        chat_id = chat.get("id")
        if not chat_id:
            return

        # Security check: allowed chats (if configured)
        if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
            logger.warning("Unauthorized Telegram access attempt from chat_id %s", chat_id)
            self._send_text(
                client, chat_id, "⛔ Access Denied. Your Telegram chat ID is not authorized to control J.A.R.V.I.S."
            )
            return

        # Check for Voice Note
        voice = message.get("voice") or message.get("audio")
        if voice:
            self._handle_voice_message(client, chat_id, voice)
            return

        # Check for Text Message
        text = (message.get("text") or "").strip()
        if not text:
            return

        # Slash Commands
        if text.startswith("/start") or text.startswith("/help"):
            welcome = (
                "🤖 *J.A.R.V.I.S. Remote Gateway Connected*\n\n"
                "I am your autonomous personal assistant. You can send me directives via text or voice memo!\n\n"
                "*Available Commands:*\n"
                "• `/goal <objective>` — Plan and run an autonomous mission\n"
                "• `/workspace` — List deliverable files created\n"
                "• Send any voice memo to transcribe and execute\n"
                "• Send any question to chat directly"
            )
            self._send_text(client, chat_id, welcome)
            return

        if text.startswith("/workspace"):
            files = [f for f in WORKSPACE_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"]
            if not files:
                self._send_text(client, chat_id, "📁 *Workspace is currently empty.*")
            else:
                lines = [f"• `{f.name}` ({f.stat().st_size / 1024:.1f} KB)" for f in files[:15]]
                self._send_text(client, chat_id, "📁 *Workspace Deliverables:*\n" + "\n".join(lines))
            return

        if text.startswith("/goal"):
            goal_prompt = text.replace("/goal", "", 1).strip()
            if not goal_prompt:
                self._send_text(client, chat_id, "⚠️ Please provide a mission description: `/goal <objective>`")
                return
            self._execute_remote_goal(client, chat_id, goal_prompt)
            return

        # Standard Conversational Query
        self._send_text(client, chat_id, "💭 *J.A.R.V.I.S. is deliberating...*")
        try:
            result = self.orchestrator.run(user_input=text, chat_history=[])
            output = result.get("output", "Done.")
            self._send_text(client, chat_id, output)
        except Exception as e:
            self._send_text(client, chat_id, f"❌ Execution error: {e}")

    def _handle_voice_message(self, client: httpx.Client, chat_id: int, voice: Dict[str, Any]) -> None:
        """Download and transcribe audio voice memo via Faster-Whisper, then execute."""
        file_id = voice.get("file_id")
        if not file_id:
            return

        self._send_text(client, chat_id, "🎙️ *Transcribing voice directive with Faster-Whisper...*")

        try:
            # 1. Fetch file path from Telegram
            file_meta = client.get(f"{self.base_url}/getFile", params={"file_id": file_id}).json()
            if not file_meta.get("ok"):
                self._send_text(client, chat_id, "❌ Could not download voice memo.")
                return

            file_path = file_meta["result"]["file_path"]
            download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            audio_bytes = client.get(download_url).content

            # 2. Transcribe
            trans_res = transcribe_voice_directive(audio_bytes)
            transcript_text = (
                trans_res.get("text", "").strip() if isinstance(trans_res, dict) else str(trans_res).strip()
            )
            if not transcript_text:
                self._send_text(client, chat_id, "⚠️ Could not understand voice memo.")
                return

            self._send_text(client, chat_id, f'🗣️ *Transcribed Directive:*\n_"{transcript_text}"_\n\n⚙️ *Executing...*')

            # 3. Execute Directive
            result = self.orchestrator.run(user_input=transcript_text, chat_history=[])
            self._send_text(client, chat_id, result.get("output", "Execution completed."))

        except Exception as e:
            logger.error("Failed to process Telegram voice memo: %s", e)
            self._send_text(client, chat_id, f"❌ Voice memo processing error: {e}")

    def _execute_remote_goal(self, client: httpx.Client, chat_id: int, goal_str: str) -> None:
        """Deconstruct and execute an autonomous mission, streaming progress and sending files."""
        self._send_text(client, chat_id, f"🎯 *Deconstructing Autonomous Mission:*\n_{goal_str}_")

        try:
            planner = GoalPlanner()
            plan = planner.plan_goal(goal_str)
            tasks = plan.get("tasks", [])

            task_summary = "\n".join([f"• `{t.get('id')}`: {t.get('title')}" for t in tasks])
            self._send_text(
                client,
                chat_id,
                f"📋 *Generated Plan DAG ({len(tasks)} subtasks):*\n{task_summary}\n\n▶ *Beginning execution...*",
            )

            # Snapshot existing workspace files to detect new deliverables
            existing_files = set(WORKSPACE_DIR.iterdir()) if WORKSPACE_DIR.exists() else set()

            runner = AutonomousRunner(orchestrator=self.orchestrator)
            final_result = runner.execute_plan(plan, initial_goal=goal_str)

            synthesis = final_result.get("synthesis", "Mission finished.")
            self._send_text(client, chat_id, f"✅ *Mission Completed!*\n\n{synthesis}")

            # Send newly created deliverable files to Telegram!
            if WORKSPACE_DIR.exists():
                new_files = [
                    f
                    for f in WORKSPACE_DIR.iterdir()
                    if f not in existing_files and f.is_file() and f.name != ".gitkeep"
                ]
                for f in new_files:
                    self._send_document(client, chat_id, f, caption=f"Deliverable: {f.name}")

        except Exception as e:
            logger.error("Failed to execute remote Telegram goal: %s", e)
            self._send_text(client, chat_id, f"❌ Goal execution error: {e}")

    def _send_text(self, client: httpx.Client, chat_id: int, text: str) -> None:
        """Send a text message to Telegram chat (with automatic fallback to plain text)."""
        url = f"{self.base_url}/sendMessage"
        # Try Markdown first
        payload = {"chat_id": chat_id, "text": text[:4000], "parse_mode": "Markdown"}
        try:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                # Fallback to plain text without Markdown parsing errors
                client.post(url, json={"chat_id": chat_id, "text": text[:4000]})
        except Exception as e:
            logger.debug("Failed to send Telegram message: %s", e)

    def _send_document(self, client: httpx.Client, chat_id: int, file_path: Path, caption: str = "") -> None:
        """Send a document or deliverable file directly to the user's Telegram."""
        url = f"{self.base_url}/sendDocument"
        try:
            with open(file_path, "rb") as fp:
                files = {"document": (file_path.name, fp)}
                data = {"chat_id": chat_id, "caption": caption[:1000]}
                client.post(url, data=data, files=files, timeout=60.0)
        except Exception as e:
            logger.warning("Failed to send document %s to Telegram: %s", file_path.name, e)


_GLOBAL_TELEGRAM_GATEWAY: Optional[TelegramGateway] = None
_GATEWAY_LOCK = threading.Lock()


def get_telegram_gateway(bot_token: Optional[str] = None) -> TelegramGateway:
    """Return the global TelegramGateway singleton."""
    global _GLOBAL_TELEGRAM_GATEWAY
    if _GLOBAL_TELEGRAM_GATEWAY is None:
        with _GATEWAY_LOCK:
            if _GLOBAL_TELEGRAM_GATEWAY is None:
                _GLOBAL_TELEGRAM_GATEWAY = TelegramGateway(bot_token=bot_token)
    elif bot_token and bot_token != _GLOBAL_TELEGRAM_GATEWAY.bot_token:
        _GLOBAL_TELEGRAM_GATEWAY.bot_token = bot_token
        _GLOBAL_TELEGRAM_GATEWAY.base_url = f"https://api.telegram.org/bot{bot_token}"
    return _GLOBAL_TELEGRAM_GATEWAY
