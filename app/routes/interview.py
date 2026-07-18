"""
Healix - Interview Route
نقطة الـ API لمحرك المحادثة (المرحلة الأولى: أخذ التاريخ المرضي فقط).

المدخل:  { "text": "...", "session_id": "... | null" }
المخرج:  سؤال عربي واحد، أو { "finished": true } عند اكتفاء المعلومات.
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
    summary="دور واحد في مقابلة أخذ التاريخ المرضي",
    description="يستخرج الأعراض، يخزّنها في الجلسة، ويعيد السؤال الطبي التالي الأهم بالعربية.",
)
async def interview_turn(
    request: InterviewTurnRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> InterviewTurnResponse:
    logger.info("دور مقابلة | session=%s | %s...", request.session_id, request.text[:50])

    # الخط (LLM + MARBERT) متزامن (blocking) — نُشغّله خارج حلقة الأحداث.
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

    return InterviewTurnResponse(
        session_id=state.session_id,
        finished=decision.finished,
        next_slot=decision.next_slot,
        question=decision.question,
        turn=state.turn_count,
        status=state.status.value,
        symptoms=[SymptomOut(text=s.text, negated=s.negated) for s in state.symptoms],
    )
