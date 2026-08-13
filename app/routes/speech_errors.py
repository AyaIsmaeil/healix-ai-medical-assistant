"""
Healix - Speech route error mapping.
"""

from fastapi import HTTPException, status

from app.exceptions import (
    AudioDownloadError,
    AudioNotFoundError,
    AudioTooLargeError,
    InvalidAudioInputError,
    SpeechError,
    TranscriptionError,
    UnsupportedAudioFormatError,
)


def speech_error_to_http(exc: SpeechError) -> HTTPException:
    if isinstance(exc, (InvalidAudioInputError, UnsupportedAudioFormatError)):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, AudioNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, AudioTooLargeError):
        return HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
    if isinstance(exc, AudioDownloadError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, TranscriptionError):
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
