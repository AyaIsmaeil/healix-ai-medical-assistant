import logging
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import whisper

from app.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    AUDIO_DOWNLOAD_TIMEOUT_SECONDS,
    MAX_AUDIO_FILE_SIZE_MB,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)
from app.core.exceptions import (
    AudioDownloadError,
    AudioNotFoundError,
    AudioTooLargeError,
    InvalidAudioInputError,
    TranscriptionError,
    UnsupportedAudioFormatError,
)

logger = logging.getLogger(__name__)


class SpeechToTextService:
    """Transcribes patient voice recordings using a preloaded OpenAI Whisper model."""

    def __init__(self, model_name: str, language: str) -> None:
        self._model_name = model_name
        self._language = language
        self._model: whisper.Whisper | None = None

    @classmethod
    def load(cls, model_name: str | None = None, language: str | None = None) -> "SpeechToTextService":
        service = cls(
            model_name=model_name or WHISPER_MODEL,
            language=language or WHISPER_LANGUAGE,
        )
        service._load_model()
        return service

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        logger.info("Loading OpenAI Whisper model '%s' into memory...", self._model_name)
        try:
            self._model = whisper.load_model(self._model_name)
        except Exception as exc:
            logger.exception("Failed to load Whisper model '%s'", self._model_name)
            raise RuntimeError(f"Unable to load Whisper model '{self._model_name}'") from exc

        logger.info("Whisper model '%s' loaded successfully and ready for inference.", self._model_name)

    def transcribe(
        self,
        *,
        audio_path: Optional[str] = None,
        audio_url: Optional[str] = None,
    ) -> str:
        if not self._model:
            raise TranscriptionError("Whisper model is not loaded.")

        temp_file: Path | None = None

        try:
            if audio_url:
                logger.info("Downloading audio from URL for transcription.")
                temp_file = self._download_audio(audio_url)
                source_path = temp_file
            elif audio_path:
                source_path = self._resolve_local_path(audio_path)
            else:
                raise InvalidAudioInputError("Either audio_path or audio_url must be provided.")

            self._validate_audio_file(source_path)
            return self._run_transcription(source_path)
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    logger.warning("Failed to delete temporary audio file: %s", temp_file)

    def _resolve_local_path(self, audio_path: str) -> Path:
        normalized_path = audio_path.strip()
        if not normalized_path:
            raise InvalidAudioInputError("audio_path cannot be empty.")

        path = Path(normalized_path)

        if not path.is_absolute():
            raise InvalidAudioInputError("audio_path must be an absolute path.")

        if not path.exists():
            logger.warning("Audio file not found: %s", path)
            raise AudioNotFoundError(f"Audio file not found: {path}")

        if not path.is_file():
            raise InvalidAudioInputError(f"audio_path must point to a file: {path}")

        if not os.access(path, os.R_OK):
            raise AudioNotFoundError(f"Audio file is not readable: {path}")

        return path

    def _download_audio(self, audio_url: str) -> Path:
        parsed_url = urlparse(audio_url)
        extension = Path(parsed_url.path).suffix.lower()

        if extension and extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise UnsupportedAudioFormatError(
                f"Unsupported audio format '{extension}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        suffix = extension or ".wav"

        try:
            with urllib.request.urlopen(audio_url, timeout=AUDIO_DOWNLOAD_TIMEOUT_SECONDS) as response:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > MAX_AUDIO_FILE_SIZE_MB:
                        raise AudioTooLargeError(
                            f"Remote audio file exceeds the {MAX_AUDIO_FILE_SIZE_MB} MB limit."
                        )

                audio_bytes = response.read()
        except AudioTooLargeError:
            raise
        except urllib.error.HTTPError as exc:
            logger.error("HTTP error while downloading audio from %s: %s", audio_url, exc)
            raise AudioDownloadError(f"Failed to download audio (HTTP {exc.code}).") from exc
        except urllib.error.URLError as exc:
            logger.error("Network error while downloading audio from %s: %s", audio_url, exc.reason)
            raise AudioDownloadError("Failed to download audio from the provided URL.") from exc
        except TimeoutError as exc:
            logger.error("Timed out downloading audio from %s", audio_url)
            raise AudioDownloadError("Audio download timed out.") from exc

        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > MAX_AUDIO_FILE_SIZE_MB:
            raise AudioTooLargeError(
                f"Downloaded audio file exceeds the {MAX_AUDIO_FILE_SIZE_MB} MB limit."
            )

        temp_file = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
        temp_file.write_bytes(audio_bytes)
        logger.info("Downloaded audio to temporary file: %s (%.2f MB)", temp_file, size_mb)
        return temp_file

    def _validate_audio_file(self, path: Path) -> None:
        extension = path.suffix.lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise UnsupportedAudioFormatError(
                f"Unsupported audio format '{extension or 'unknown'}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_AUDIO_FILE_SIZE_MB:
            raise AudioTooLargeError(
                f"Audio file exceeds the {MAX_AUDIO_FILE_SIZE_MB} MB limit."
            )

        if path.stat().st_size == 0:
            raise InvalidAudioInputError("Audio file is empty.")

    def _run_transcription(self, path: Path) -> str:
        logger.info("Starting Whisper transcription for: %s", path.name)

        try:
            result = self._model.transcribe(
                str(path),
                language=self._language,
                task="transcribe",
            )
        except Exception as exc:
            logger.exception("Whisper transcription failed for file: %s", path)
            raise TranscriptionError("Whisper failed to transcribe the audio file.") from exc

        text = str(result.get("text", "")).strip()
        if not text:
            logger.warning("Whisper returned empty transcription for file: %s", path)
            raise TranscriptionError("Whisper returned an empty transcription.")

        logger.info("Transcription completed successfully (%d characters).", len(text))
        return text
