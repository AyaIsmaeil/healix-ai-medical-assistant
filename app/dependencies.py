"""
Healix - Dependencies
مزوّدو الاعتمادية (Dependency Injection) لطبقة الـ API.

تُبنى الكائنات مرّة واحدة عند بدء التشغيل وتُخزَّن في ``app.state``،
ثمّ تُحقَن هنا عبر ``Depends`` دون أي متغيّرات عامة على مستوى الوحدة.
"""

from fastapi import Request

from app.exceptions import ConversationError, ModelNotLoadedError
from app.services.conversation_service import ConversationService
from app.services.speech_to_text import SpeechToTextService
from app.services.symptom_extractor import SymptomExtractor


def get_symptom_extractor(request: Request) -> SymptomExtractor:
    """إرجاع خدمة استخراج الأعراض المُهيّأة عند بدء التشغيل."""
    extractor = getattr(request.app.state, "symptom_extractor", None)
    if extractor is None:
        raise ModelNotLoadedError("خدمة استخراج الأعراض غير مُهيّأة.")
    return extractor


def get_speech_to_text_service(request: Request) -> SpeechToTextService:
    """إرجاع خدمة التفريغ النصّي (Whisper) المُهيّأة عند بدء التشغيل."""
    service = getattr(request.app.state, "speech_service", None)
    if service is None:
        raise ModelNotLoadedError("خدمة تفريغ الصوت (Whisper) غير مُهيّأة.")
    return service


def get_conversation_service(request: Request) -> ConversationService:
    """إرجاع محرك المحادثة المُهيّأ عند بدء التشغيل."""
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise ConversationError("محرك المحادثة غير مُهيّأ.")
    return service
