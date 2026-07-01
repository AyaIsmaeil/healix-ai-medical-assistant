class SpeechToTextError(Exception):
    """Base exception for speech-to-text operations."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidAudioInputError(SpeechToTextError):
    status_code = 422


class AudioNotFoundError(SpeechToTextError):
    status_code = 404


class UnsupportedAudioFormatError(SpeechToTextError):
    status_code = 415


class AudioDownloadError(SpeechToTextError):
    status_code = 502


class AudioTooLargeError(SpeechToTextError):
    status_code = 413


class TranscriptionError(SpeechToTextError):
    status_code = 500
