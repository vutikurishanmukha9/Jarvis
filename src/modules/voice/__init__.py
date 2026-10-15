"""
Voice Intelligence module for J.A.R.V.I.S.
Integrates local faster-whisper speech recognition and voice-to-goal command pipelines.
"""

from .voice_bridge import (
    get_local_whisper_model,
    is_local_whisper_available,
    transcribe_voice_directive,
)
from .wake_word import WakeWordListener, get_wake_listener

__all__ = [
    "transcribe_voice_directive",
    "is_local_whisper_available",
    "get_local_whisper_model",
    "WakeWordListener",
    "get_wake_listener",
]
