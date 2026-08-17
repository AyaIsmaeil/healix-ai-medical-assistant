"""Tests for speech_client.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import speech_client
from speech_client import SpeechUnavailable, set_whisper_model, synthesize, transcribe


@pytest.fixture(autouse=True)
def _no_real_retry_delay(monkeypatch):
    # Every synthesize() test in this file should run instantly — real
    # retry backoff (_SYNTHESIS_RETRY_DELAY_SECONDS) is only for
    # production, same "indirected sleep" pattern llm_client.py uses.
    monkeypatch.setattr(speech_client, "_async_sleep", AsyncMock())


class _FakeSegment:
    def __init__(self, text: str):
        self.text = text


@pytest.fixture(autouse=True)
def _reset_whisper_model():
    set_whisper_model(None)
    yield
    set_whisper_model(None)


def test_transcribe_empty_audio_raises():
    with pytest.raises(SpeechUnavailable, match="empty"):
        transcribe(b"")


def test_transcribe_returns_joined_segments(tmp_path):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (
        [_FakeSegment(" عندي "), _FakeSegment("صداع")],
        MagicMock(),
    )
    set_whisper_model(fake_model)

    text = transcribe(b"fake-audio-bytes")
    assert text == "عندي صداع"
    fake_model.transcribe.assert_called_once()
    _args, kwargs = fake_model.transcribe.call_args
    assert kwargs["language"] == "ar"


def test_transcribe_no_text_raises():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([], MagicMock())
    set_whisper_model(fake_model)

    with pytest.raises(SpeechUnavailable, match="no text"):
        transcribe(b"audio")


def test_transcribe_whisper_failure_raises():
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("ffmpeg missing")
    set_whisper_model(fake_model)

    with pytest.raises(SpeechUnavailable, match="transcription failed"):
        transcribe(b"audio")


@pytest.mark.asyncio
async def test_synthesize_empty_text_raises():
    with pytest.raises(SpeechUnavailable, match="empty"):
        await synthesize("   ")


@pytest.mark.asyncio
async def test_synthesize_returns_mp3_chunks():
    async def fake_stream():
        yield {"type": "audio", "data": b"\xff\xfb"}
        yield {"type": "audio", "data": b"\x90\x00"}

    fake_communicate = MagicMock()
    fake_communicate.stream = fake_stream

    with patch("edge_tts.Communicate", return_value=fake_communicate):
        audio = await synthesize("مرحبا")
    assert audio == b"\xff\xfb\x90\x00"


@pytest.mark.asyncio
async def test_synthesize_no_audio_raises_after_both_attempts_empty():
    async def empty_stream():
        if False:
            yield {}

    fake_communicate = MagicMock()
    fake_communicate.stream = empty_stream

    with patch("edge_tts.Communicate", return_value=fake_communicate) as fake_ctor:
        with pytest.raises(SpeechUnavailable, match="no audio"):
            await synthesize("مرحبا")
    assert fake_ctor.call_count == 2  # both retry attempts exhausted


# --- retry: absorbs one transient edge-tts failure (module docstring) -----------


@pytest.mark.asyncio
async def test_synthesize_retries_once_after_a_failure_then_succeeds():
    async def failing_stream():
        raise RuntimeError("NoAudioReceived: transient upstream failure")
        yield {}  # pragma: no cover - unreachable, makes this an async generator

    async def working_stream():
        yield {"type": "audio", "data": b"\xff\xfb"}

    first_communicate = MagicMock()
    first_communicate.stream = failing_stream
    second_communicate = MagicMock()
    second_communicate.stream = working_stream

    with patch("edge_tts.Communicate", side_effect=[first_communicate, second_communicate]):
        audio = await synthesize("مرحبا")

    assert audio == b"\xff\xfb"


@pytest.mark.asyncio
async def test_synthesize_raises_after_both_attempts_fail():
    async def failing_stream():
        raise RuntimeError("still failing")
        yield {}  # pragma: no cover

    fake_communicate = MagicMock()
    fake_communicate.stream = failing_stream

    with patch("edge_tts.Communicate", return_value=fake_communicate) as fake_ctor:
        with pytest.raises(SpeechUnavailable, match="still failing"):
            await synthesize("مرحبا")
    assert fake_ctor.call_count == 2


@pytest.mark.asyncio
async def test_synthesize_uses_a_fresh_communicate_instance_per_attempt():
    # A real edge_tts.Communicate object raises RuntimeError if .stream()
    # is called twice — retrying MUST construct a new instance, never
    # reuse the exhausted one.
    async def failing_stream():
        raise RuntimeError("first attempt fails")
        yield {}  # pragma: no cover

    async def working_stream():
        yield {"type": "audio", "data": b"ok"}

    first_communicate = MagicMock()
    first_communicate.stream = failing_stream
    second_communicate = MagicMock()
    second_communicate.stream = working_stream

    with patch(
        "edge_tts.Communicate", side_effect=[first_communicate, second_communicate]
    ) as fake_ctor:
        await synthesize("مرحبا")

    assert fake_ctor.call_count == 2


@pytest.mark.asyncio
async def test_synthesize_sleeps_between_attempts_not_after_the_last_one(monkeypatch):
    sleep_mock = AsyncMock()
    monkeypatch.setattr(speech_client, "_async_sleep", sleep_mock)

    async def failing_stream():
        raise RuntimeError("fails every time")
        yield {}  # pragma: no cover

    fake_communicate = MagicMock()
    fake_communicate.stream = failing_stream

    with patch("edge_tts.Communicate", return_value=fake_communicate):
        with pytest.raises(SpeechUnavailable):
            await synthesize("مرحبا")

    # _SYNTHESIS_MAX_ATTEMPTS=2 -> exactly one sleep, between attempt 1 and 2.
    sleep_mock.assert_awaited_once_with(speech_client._SYNTHESIS_RETRY_DELAY_SECONDS)
