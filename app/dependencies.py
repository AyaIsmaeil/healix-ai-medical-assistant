from fastapi import Request

from app.services.speech_to_text import SpeechToTextService


def get_speech_to_text_service(request: Request) -> SpeechToTextService:
    return request.app.state.speech_to_text_service
