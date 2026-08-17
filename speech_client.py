"""Single entry point for speech I/O in this service.

STT (speech-to-text): OpenAI Whisper via faster-whisper — converts patient
audio into Arabic text for POST /chat.

TTS (text-to-speech): Microsoft Edge TTS via edge-tts — reads assistant
replies aloud. Whisper does NOT do TTS; these are separate backends behind
one module so callers (api/main.py) see a uniform interface.

Deliberately NOT implemented: no fallback from Whisper to a cloud STT API,
no streaming transcription, no voice cloning. Same "no silent fallback"
discipline as llm_client.py.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Lazy-loaded Whisper model — first transcribe() pays the load cost so
# api/main.py startup (graph + checkpointer) is not blocked on a multi-GB
# model download unless speech is actually used.
_whisper_model: Any | None = None


class SpeechError(Exception):
    """Base for every failure this module raises."""


class SpeechConfigError(SpeechError):
    """Required configuration or dependency is missing."""


class SpeechUnavailable(SpeechError):
    """Transcription or synthesis could not complete."""


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw or default


def _get_whisper_model() -> Any:
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on env
        raise SpeechConfigError(
            "faster-whisper is not installed. Install it with "
            "`pip install faster-whisper`, or disable speech endpoints."
        ) from exc

    model_size = _env_str("HEALIX_WHISPER_MODEL", "base")
    device = _env_str("HEALIX_WHISPER_DEVICE", "cpu")
    compute_type = _env_str("HEALIX_WHISPER_COMPUTE_TYPE", "int8")

    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _whisper_model


def set_whisper_model(model: Any | None) -> None:
    """Override the loaded Whisper model. For tests only."""
    global _whisper_model
    _whisper_model = model


def transcribe(audio_bytes: bytes, *, language: str | None = None) -> str:
    """Transcribe audio bytes to text using Whisper (faster-whisper).

    audio_bytes: raw audio file contents (webm, wav, mp3, ogg, …).
        Requires ffmpeg on PATH for formats Whisper cannot read natively
        (browser MediaRecorder webm/opus is the common case in dev UI).
    language: BCP-47 code passed to Whisper. Defaults to HEALIX_WHISPER_LANGUAGE
        (ar) — Syrian colloquial is still tagged "ar" by Whisper.

    Returns stripped transcript text, or raises SpeechUnavailable if the
    audio produced nothing usable.
    """
    if not audio_bytes:
        raise SpeechUnavailable("empty audio payload")

    lang = language or _env_str("HEALIX_WHISPER_LANGUAGE", "ar")

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _get_whisper_model()
        segments, _info = model.transcribe(
            tmp_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )
        text = "".join(segment.text for segment in segments).strip()
    except SpeechError:
        raise
    except Exception as exc:
        raise SpeechUnavailable(f"transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not text:
        raise SpeechUnavailable("transcription produced no text")
    return text


# edge-tts talks to an unofficial Microsoft backend over a websocket, and
# was observed directly during this feature's integration to occasionally
# raise edge_tts.exceptions.NoAudioReceived (the connection completes with
# no audio chunks at all) on a request that succeeds moments later on an
# identical retry — not a code defect on this side, a transient upstream
# reliability issue. One retry (two attempts total) absorbs that without
# hiding a genuinely broken configuration: a real config problem (bad
# voice name, empty text, edge-tts not installed) fails identically on
# both attempts, so the retry costs one extra round trip in that case,
# never a silently-different outcome.
_SYNTHESIS_MAX_ATTEMPTS = 2
_SYNTHESIS_RETRY_DELAY_SECONDS = 0.5

# Indirected so tests can drive it without a real wait — same pattern
# llm_client.py uses for its own retry backoff (`_sleep = time.sleep`
# there; this module is async, so asyncio.sleep here).
_async_sleep = asyncio.sleep


async def synthesize(text: str) -> bytes:
    """Synthesize text to MP3 bytes using Edge TTS.

    Voice is HEALIX_TTS_VOICE (default ar-SA-ZariyahNeural). Arabic voices
    cover Modern Standard / Gulf; Syrian colloquial in the reply still
    reads intelligibly even though no ar-SY voice exists in Edge TTS.

    Retries once (see _SYNTHESIS_MAX_ATTEMPTS) on any failure, including a
    stream that completes with zero audio chunks — a fresh edge_tts.Communicate
    per attempt, since a Communicate instance can only be streamed once.
    """
    cleaned = text.strip()
    if not cleaned:
        raise SpeechUnavailable("empty text for synthesis")

    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover - depends on env
        raise SpeechConfigError(
            "edge-tts is not installed. Install it with `pip install edge-tts`."
        ) from exc

    voice = _env_str("HEALIX_TTS_VOICE", "ar-SA-ZariyahNeural")

    last_error: SpeechUnavailable = SpeechUnavailable("synthesis produced no audio")
    for attempt in range(1, _SYNTHESIS_MAX_ATTEMPTS + 1):
        communicate = edge_tts.Communicate(cleaned, voice)
        chunks: list[bytes] = []
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
        except Exception as exc:
            last_error = SpeechUnavailable(f"synthesis failed: {exc}")
        else:
            if chunks:
                return b"".join(chunks)
            last_error = SpeechUnavailable("synthesis produced no audio")

        if attempt < _SYNTHESIS_MAX_ATTEMPTS:
            await _async_sleep(_SYNTHESIS_RETRY_DELAY_SECONDS)

    raise last_error


def synthesize_sync(text: str) -> bytes:
    """Blocking wrapper around synthesize() for sync FastAPI routes if needed."""
    return asyncio.run(synthesize(text))
