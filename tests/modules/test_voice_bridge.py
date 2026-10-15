"""
Unit tests for J.A.R.V.I.S. Voice Intelligence Bridge.
"""

import io
from unittest.mock import MagicMock, patch
import pytest

from src.modules.voice.voice_bridge import (
    is_local_whisper_available,
    transcribe_voice_directive,
)


def test_voice_availability():
    """Verify that faster-whisper is recognized as available."""
    assert is_local_whisper_available() is True


def test_transcribe_empty_audio():
    """Verify that empty or None audio data is handled safely."""
    res1 = transcribe_voice_directive(b"")
    assert res1["success"] is False
    assert "No audio data" in res1["error"] or "empty" in res1["error"]

    res2 = transcribe_voice_directive(None)
    assert res2["success"] is False


def test_transcribe_invalid_type():
    """Verify that non-audio objects are rejected gracefully."""
    res = transcribe_voice_directive(12345)  # type: ignore
    assert res["success"] is False
    assert "Unsupported audio_data type" in res["error"]


def test_transcribe_local_whisper_mock():
    """Verify faster-whisper transcription workflow with mocked model segments."""
    fake_audio_bytes = b"RIFF" + b"\x00" * 100  # valid-length dummy header

    mock_segment_1 = MagicMock()
    mock_segment_1.text = " Research the top 3 AI humanoid "
    mock_segment_2 = MagicMock()
    mock_segment_2.text = " robotics companies "

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 4.2

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment_1, mock_segment_2], mock_info)

    with patch("src.modules.voice.voice_bridge.get_local_whisper_model", return_value=mock_model):
        res = transcribe_voice_directive(fake_audio_bytes, model_size="base")

    assert res["success"] is True
    assert res["text"] == "Research the top 3 AI humanoid robotics companies"
    assert res["language"] == "en"
    assert res["duration"] == 4.2
    assert res["engine"] == "faster-whisper-local"


def test_transcribe_bytes_io_stream():
    """Verify that io.BytesIO stream buffers are correctly processed."""
    stream = io.BytesIO(b"RIFF" + b"\x00" * 128)

    mock_segment = MagicMock()
    mock_segment.text = "Build an Excel model"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 2.0

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch("src.modules.voice.voice_bridge.get_local_whisper_model", return_value=mock_model):
        res = transcribe_voice_directive(stream)

    assert res["success"] is True
    assert res["text"] == "Build an Excel model"
    assert res["engine"] == "faster-whisper-local"


def test_transcribe_fallback_to_openai_api():
    """Verify fallback to cloud Whisper API when local engine is uninitialized."""
    fake_audio = b"RIFF" + b"\x00" * 100

    mock_openai_client = MagicMock()
    mock_transcript = MagicMock()
    mock_transcript.text = "Draft a recruiter outreach campaign"
    mock_openai_client.audio.transcriptions.create.return_value = mock_transcript

    with patch("src.modules.voice.voice_bridge.is_local_whisper_available", return_value=False), \
         patch("openai.OpenAI", return_value=mock_openai_client):
        res = transcribe_voice_directive(fake_audio, api_key="sk-test-key")

    assert res["success"] is True
    assert res["text"] == "Draft a recruiter outreach campaign"
    assert res["engine"] == "openai-whisper-api"
