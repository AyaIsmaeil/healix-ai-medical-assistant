import asyncio
import logging
from functools import partial

from fastapi import APIRouter, Depends

from app.dependencies import get_speech_to_text_service
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
    text = await loop.run_in_executor(
        None,
        partial(
            service.transcribe,
            audio_path=payload.audio_path,
            audio_url=str(payload.audio_url) if payload.audio_url else None,
        ),
    )

    return SpeechToTextResponse(success=True, text=text)
