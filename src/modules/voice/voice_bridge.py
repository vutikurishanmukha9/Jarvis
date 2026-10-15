"""
Voice Intelligence Bridge for J.A.R.V.I.S.
Provides offline local Speech-to-Text via faster-whisper (inspired by VoiceStudio ASR sidecar)
with seamless cloud Whisper fallback (OpenAI/OpenRouter) for voice directives.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Cached local model singleton
_MODEL_LOCK = threading.Lock()
_CACHED_WHISPER_MODEL: Any = None
_CACHED_MODEL_SIZE: Optional[str] = None


def is_local_whisper_available() -> bool:
    """Check if faster-whisper is installed and importable."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def get_local_whisper_model(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """
    Get or initialize a cached singleton instance of WhisperModel from faster-whisper.
    Default to 'base' with 'int8' for sub-second CPU transcription.
    """
    global _CACHED_WHISPER_MODEL, _CACHED_MODEL_SIZE

    with _MODEL_LOCK:
        if _CACHED_WHISPER_MODEL is not None and _CACHED_MODEL_SIZE == model_size:
            return _CACHED_WHISPER_MODEL

        try:
            from faster_whisper import WhisperModel

            logger.info("Initializing faster-whisper model (size: %s, device: %s, compute: %s)...", model_size, device, compute_type)
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            _CACHED_WHISPER_MODEL = model
            _CACHED_MODEL_SIZE = model_size
            return model
        except Exception as e:
            logger.warning("Failed to initialize local faster-whisper model '%s': %s", model_size, e)
            return None


def transcribe_voice_directive(
    audio_data: Union[bytes, io.BytesIO, str, Path],
    model_size: str = "base",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Transcribes voice audio to text.
    
    Accepts:
        - raw bytes
        - io.BytesIO or file-like buffer (from Streamlit st.audio_input)
        - str/Path to a recorded audio file
        
    Returns:
        {
            "success": True/False,
            "text": "Transcribed voice command",
            "language": "en",
            "duration": float,
            "engine": "faster-whisper-local" | "openai-whisper-api" | "error",
            "error": Optional[str]
        }
    """
    if not audio_data:
        return {
            "success": False,
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "engine": "error",
            "error": "No audio data provided.",
        }

    # Normalize audio into a temp file for engines that require a path
    temp_path: Optional[str] = None
    try:
        if isinstance(audio_data, (str, Path)):
            audio_path = str(audio_data)
        else:
            # Extract bytes
            if hasattr(audio_data, "getvalue"):
                raw_bytes = audio_data.getvalue()
            elif hasattr(audio_data, "read"):
                audio_data.seek(0)
                raw_bytes = audio_data.read()
            elif isinstance(audio_data, bytes):
                raw_bytes = audio_data
            else:
                return {
                    "success": False,
                    "text": "",
                    "language": "unknown",
                    "duration": 0.0,
                    "engine": "error",
                    "error": f"Unsupported audio_data type: {type(audio_data)}",
                }

            if not raw_bytes or len(raw_bytes) < 64:
                return {
                    "success": False,
                    "text": "",
                    "language": "unknown",
                    "duration": 0.0,
                    "engine": "error",
                    "error": "Audio data buffer is empty or corrupted.",
                }

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_bytes)
                temp_path = tmp.name
            audio_path = temp_path

        # 1. Primary Engine: faster-whisper local execution
        if is_local_whisper_available():
            try:
                model = get_local_whisper_model(model_size=model_size)
                if model is not None:
                    segments, info = model.transcribe(
                        audio_path,
                        beam_size=5,
                        language=language,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )
                    transcribed_text = " ".join(seg.text.strip() for seg in segments).strip()
                    if transcribed_text:
                        return {
                            "success": True,
                            "text": transcribed_text,
                            "language": info.language if info else "en",
                            "duration": getattr(info, "duration", 0.0),
                            "engine": "faster-whisper-local",
                            "error": None,
                        }
            except Exception as local_err:
                logger.warning("Local faster-whisper transcription error: %s. Falling back to API...", local_err)

        # 2. Fallback Engine: Cloud Whisper API (OpenAI / OpenRouter)
        effective_key = api_key or os.environ.get("OPENAI_API_KEY")
        if effective_key:
            try:
                import openai

                client = openai.OpenAI(api_key=effective_key, base_url=base_url)
                with open(audio_path, "rb") as f:
                    transcript_resp = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=language,
                    )
                text = getattr(transcript_resp, "text", str(transcript_resp)).strip()
                return {
                    "success": True,
                    "text": text,
                    "language": language or "en",
                    "duration": 0.0,
                    "engine": "openai-whisper-api",
                    "error": None,
                }
            except Exception as api_err:
                logger.error("Cloud Whisper API transcription error: %s", api_err)
                return {
                    "success": False,
                    "text": "",
                    "language": "unknown",
                    "duration": 0.0,
                    "engine": "error",
                    "error": f"Transcription failed locally and via API: {api_err}",
                }

        return {
            "success": False,
            "text": "",
            "language": "unknown",
            "duration": 0.0,
            "engine": "error",
            "error": "No ASR engine available (faster-whisper uninitialized and no API key provided).",
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
