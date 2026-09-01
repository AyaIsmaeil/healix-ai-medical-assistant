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
import subprocess
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

    model_size = _env_str("HEALIX_WHISPER_MODEL", "small")
    device = _env_str("HEALIX_WHISPER_DEVICE", "cpu")
    compute_type = _env_str("HEALIX_WHISPER_COMPUTE_TYPE", "int8")

    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _whisper_model


def set_whisper_model(model: Any | None) -> None:
    """Override the loaded Whisper model. For tests only."""
    global _whisper_model
    _whisper_model = model


def _ffmpeg_normalize_to_wav(src_path: str) -> str:
    """Remux `src_path` to a 16kHz mono WAV file via ffmpeg and return the
    new file's path.

    Browser MediaRecorder audio (webm/opus on Chrome/Android, mp4/aac on
    Safari/iOS) used to be handed to Whisper as-is. Chrome's webm
    container is commonly written WITHOUT a valid Duration in its header —
    a long-standing MediaRecorder limitation on streamed, non-seekable
    output — which can throw off timestamp-dependent decoding.
    vad_filter=True below depends on correct timing, and a malformed
    duration was the identified cause of garbled/partial transcripts from
    Android specifically, while Safari's mp4 output (duration normally
    finalized) decoded cleanly. A full ffmpeg remux forces fresh, correct
    timestamps regardless of the source container's quirks, for every
    input format uniformly — not just the Android case.
    """
    wav_path = src_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SpeechUnavailable(f"audio normalization failed: {exc}") from exc
    return wav_path


# Indirected so tests can bypass real ffmpeg (see _async_sleep above for
# the same pattern) — production always goes through _ffmpeg_normalize_to_wav.
_normalize_to_wav = _ffmpeg_normalize_to_wav


def transcribe(audio_bytes: bytes, *, language: str | None = None) -> str:
    """Transcribe audio bytes to text using Whisper (faster-whisper).
    """
    if not audio_bytes:
        raise SpeechUnavailable("empty audio payload")

    lang = language or _env_str("HEALIX_WHISPER_LANGUAGE", "ar")

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    wav_path: str | None = None
    try:
        wav_path = _normalize_to_wav(tmp_path)
        model = _get_whisper_model()
        segments, _info = model.transcribe(
            wav_path,
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
        for path in (tmp_path, wav_path):
            if path is None:
                continue
            try:
                os.unlink(path)
            except OSError:
                pass

    if not text:
        raise SpeechUnavailable("transcription produced no text")
    return text


_SYNTHESIS_MAX_ATTEMPTS = 2
_SYNTHESIS_RETRY_DELAY_SECONDS = 0.5
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
