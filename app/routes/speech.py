import asyncio
import logging
from functools import partial

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_speech_to_text_service
from app.exceptions import SpeechError
from app.routes.speech_errors import speech_error_to_http
from app.schemas.speech import SpeechToTextRequest, SpeechToTextResponse
from app.services.speech_to_text import SpeechToTextService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Speech To Text"])


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def transcribe_speech(
    payload: SpeechToTextRequest,
    service: SpeechToTextService = Depends(get_speech_to_text_service),
) -> SpeechToTextResponse:
    logger.info(
        "Received speech-to-text request via %s.",
        "audio_url" if payload.audio_url else "audio_path",
    )

    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(
            None,
            partial(
                service.transcribe,
                audio_path=payload.audio_path,
                audio_url=str(payload.audio_url) if payload.audio_url else None,
            ),
        )
    except SpeechError as exc:
        logger.error("Speech-to-text failed: %s", exc)
        raise speech_error_to_http(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected speech-to-text failure")
        raise HTTPException(status_code=500, detail="Speech transcription failed.") from exc

    return SpeechToTextResponse(success=True, text=text)
