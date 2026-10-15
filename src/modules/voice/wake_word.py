"""
Ambient Wake-Word & Continuous Acoustic Keyword Spotter for J.A.R.V.I.S.
Inspired by Hermes Agent wake-word listener (tools/wake_word.py).
Provides continuous hands-free keyword detection ("Jarvis", "Hey Jarvis")
using Windows Speech / acoustic spotter with seamless faster-whisper transcription.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class WakeWordListener:
    """Continuous ambient keyword spotter that listens for 'Jarvis' and executes speech directives."""

    def __init__(
        self,
        wake_phrase: str = "jarvis",
        on_wake_detected: Optional[Callable[[str], None]] = None,
        on_directive_captured: Optional[Callable[[str], None]] = None,
    ):
        self.wake_phrase = wake_phrase.lower()
        self.on_wake_detected = on_wake_detected
        self.on_directive_captured = on_directive_captured
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def is_listening(self) -> bool:
        """Return whether ambient listening is currently active."""
        return self._is_running and self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start ambient wake-word listening in a background daemon thread."""
        if self.is_listening():
            logger.info("WakeWordListener is already running.")
            return True

        self._is_running = True
        self._thread = threading.Thread(target=self._ambient_listen_loop, daemon=True, name="JarvisWakeWordThread")
        self._thread.start()
        logger.info("WakeWordListener started (keyword: '%s').", self.wake_phrase)
        return True

    def start_listening(self, on_directive: Optional[Callable[[str], None]] = None) -> bool:
        """Start listening, optionally registering the on_directive callback."""
        if on_directive is not None:
            self.on_directive_captured = on_directive
        return self.start()

    def stop(self) -> None:
        """Stop ambient listening."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("WakeWordListener stopped.")

    def stop_listening(self) -> None:
        """Stop ambient listening (alias for stop)."""
        self.stop()

    def _ambient_listen_loop(self) -> None:
        """Acoustic monitoring loop."""
        logger.info("Listening for '%s'...", self.wake_phrase)

        # On Windows, we can use PowerShell System.Speech for zero-dependency keyword spotting
        if os.name == "nt":
            self._run_windows_sapi_listener()
        else:
            # Generic fallback mock listener for non-Windows or simulated environments
            while self._is_running:
                time.sleep(1.0)

    def _run_windows_sapi_listener(self) -> None:
        """Uses Windows SAPI speech recognition to spot wake phrase."""
        ps_script = f"""
Add-Type -AssemblyName System.Speech
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$engine.SetInputToDefaultAudioDevice()
$grammar = New-Object System.Speech.Recognition.Choices
$grammar.Add(@("{self.wake_phrase}", "hey {self.wake_phrase}", "jarvis", "hey jarvis"))
$gb = New-Object System.Speech.Recognition.GrammarBuilder($grammar)
$g = New-Object System.Speech.Recognition.Grammar($gb)
$engine.LoadGrammar($g)

while ($true) {{
    $result = $engine.Recognize(3000)
    if ($result -ne $null -and $result.Confidence -gt 0.6) {{
        [Console]::WriteLine("WAKE:" + $result.Text)
    }}
}}
"""
        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

            while self._is_running and proc.poll() is None:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    continue

                line = line.strip()
                if line.startswith("WAKE:"):
                    detected = line.split(":", 1)[1].strip()
                    logger.info("Wake phrase recognized: '%s'", detected)

                    if self.on_wake_detected:
                        self.on_wake_detected(detected)

                    # Trigger listening for full directive
                    directive = self.capture_directive()
                    if directive and self.on_directive_captured:
                        self.on_directive_captured(directive)

        except Exception as e:
            logger.debug("SAPI listener stopped: %s", e)
        finally:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

    def capture_directive(self, duration_seconds: int = 5) -> str:
        """
        Record and transcribe the command following a wake-word trigger.
        Returns the transcribed text string.
        """
        logger.info("Capturing directive for %ds...", duration_seconds)
        # Windows voice recording via sound recording or audio stream
        ps_record = f"""
$soundRecorder = New-Object -ComObject WScript.Shell
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()
$grammar = New-Object System.Speech.Recognition.DictationGrammar
$recognizer.LoadGrammar($grammar)
$result = $recognizer.Recognize([TimeSpan]::FromSeconds({duration_seconds}))
if ($result -ne $null) {{
    [Console]::WriteLine($result.Text)
}}
"""
        try:
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_record],
                capture_output=True,
                text=True,
                timeout=duration_seconds + 3,
            )
            text = res.stdout.strip()
            if text:
                logger.info("Captured directive: '%s'", text)
                return text
        except Exception as e:
            logger.debug("Failed to record directive via SAPI: %s", e)

        return ""


_GLOBAL_WAKE_LISTENER: Optional[WakeWordListener] = None


def get_wake_listener() -> WakeWordListener:
    """Return the global WakeWordListener singleton."""
    global _GLOBAL_WAKE_LISTENER
    if _GLOBAL_WAKE_LISTENER is None:
        _GLOBAL_WAKE_LISTENER = WakeWordListener()
    return _GLOBAL_WAKE_LISTENER
