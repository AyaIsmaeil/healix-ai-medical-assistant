"""
Healix - Clinical Interview Route
نقطة الـ API لوكيل المقابلة السريرية (أخذ التاريخ المرضي + الاستخراج المنظَّم).

المدخل:  { "text": "...", "session_id": "... | null" }
المخرج:  السجل الطبي المنظَّم + سؤال عربي واحد، أو الإنهاء عند اكتفاء المعلومات.
لا تشخيص ولا احتمالات أمراض ولا توصيات في هذه المرحلة.
"""

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_conversation_service
from app.exceptions import (
    ConversationError,
    InterviewParsingError,
    LLMProviderError,
)
from app.schemas.interview import (
    InterviewTurnRequest,
    InterviewTurnResponse,
    SymptomOut,
)
from app.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["محرك المحادثة"])


@router.post(
    "/interview/turn",
    response_model=InterviewTurnResponse,
    status_code=status.HTTP_200_OK,
    summary="دور واحد في المقابلة السريرية",
    description=(
        "استدعاء LLM واحد يستخرج المعلومات الطبية المنظَّمة (أعراض، شدّة، مدّة، "
        "موضع، أدوية، حساسية، أمراض مزمنة، تاريخ عائلي، النواقص) ويعيد السؤال "
        "الطبي التالي الأهم بالعربية. لا تشخيص."
    ),
)
async def interview_turn(
    request: InterviewTurnRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> InterviewTurnResponse:
    logger.info("دور مقابلة | session=%s | %s...", request.session_id, request.text[:50])

    # استدعاء الـLLM متزامن (blocking) — نُشغّله خارج حلقة الأحداث.
    loop = asyncio.get_running_loop()
    try:
        state, decision = await loop.run_in_executor(
            None, partial(service.handle_message, request.text, request.session_id)
        )
    except (LLMProviderError, InterviewParsingError) as exc:
        logger.error("فشل الدور: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc
    except ConversationError as exc:
        logger.error("خطأ محادثة: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    record = state.record
    return InterviewTurnResponse(
        # العقد الأصلي (بلا تغيير).
        session_id=state.session_id,
        finished=decision.finished,
        next_slot=decision.next_slot,
        question=decision.question,
        turn=state.turn_count,
        status=state.status.value,
        symptoms=[
            SymptomOut(text=s.text, negated=s.negated, confidence=s.confidence)
            for s in state.symptoms
        ],
        # السجل الطبي المنظَّم (إضافة LLM-first).
        chief_complaint=record.chief_complaint,
        severity=record.severity,
        duration=record.duration,
        body_location=record.body_location,
        medications=record.medications,
        allergies=record.allergies,
        chronic_conditions=record.chronic_conditions,
        family_history=record.family_history,
        missing_fields=record.missing_fields,
        # مرآتا العقد الجديد — نفس مصدر الحقيقة، لا قيمة مستقلة.
        interview_complete=decision.finished,
        next_question=decision.question,
    )
