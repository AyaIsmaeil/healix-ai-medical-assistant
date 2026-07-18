"""
Healix - Conversation Service (Use Case)
محرك المحادثة — المرحلة الأولى: أخذ التاريخ المرضي فقط (بلا تشخيص).

دورة الدور الواحد:
    رسالة المريض
        → تسجيل إجابة الخانة المعلّقة (إن وُجدت)
        → استخراج الأعراض بـ MARBERT وتخزينها في الجلسة
        → استدعاء الـ LLM لاختيار السؤال التالي الأهم
        → إعادة سؤال عربي واحد، أو {"finished": true}

يعتمد فقط على المنافذ (Ports) المحقونة، فلا اقتران بنموذج ملموس، وقابل للاختبار.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.domain.conversation import (
    ConversationState,
    InterviewDecision,
    Symptom,
)
from app.domain.ports import LLMProvider, SessionStore, SymptomExtractorPort
from app.exceptions import ConversationError, InterviewParsingError, LLMProviderError
from app.parsing.interview_parser import parse_interview_decision
from app.prompts.interview_builder import InterviewPromptBuilder

logger = logging.getLogger(__name__)


class ConversationService:
    """يدير دوراً واحداً من مقابلة أخذ التاريخ المرضي."""

    def __init__(
        self,
        extractor: SymptomExtractorPort,
        provider: LLMProvider,
        prompt_builder: InterviewPromptBuilder,
        store: SessionStore,
        max_questions: int = 8,
    ) -> None:
        self._extractor = extractor
        self._provider = provider
        self._prompts = prompt_builder
        self._store = store
        self._max_questions = int(max_questions)

    def handle_message(
        self, text: str, session_id: Optional[str] = None
    ) -> Tuple[ConversationState, InterviewDecision]:
        """معالجة رسالة مريض وإرجاع (الحالة المحدَّثة، القرار)."""
        state = self._store.get_or_create(session_id)

        # 1) الرسالة الحالية هي ردّ المريض على سؤال الدور السابق — نمسح التعليق.
        #    لا نُسند نصّها إلى خانة بعينها (قد لا يُجيب المريض عن السؤال مباشرةً)؛
        #    المعلومة الكاملة تُحفظ في raw_messages وتصل للـ LLM كما هي.
        state.pending_slot = None

        # 2) حفظ الرسالة الخام كاملةً (سياق كامل للـ LLM، لا يُفقد أي شيء).
        state.record_patient_message(text)

        # 3) استخراج الأعراض (MARBERT) وتخزينها بلا تكرار.
        extracted = self._extractor.extract(text)
        state.add_symptoms(
            Symptom(text=s.text, negated=s.negated, confidence=s.confidence)
            for s in extracted
        )

        state.turn_count += 1

        # 4) قرار الدور.
        decision = self._decide(state)

        # 5) تطبيق القرار على الحالة.
        if decision.finished:
            state.mark_completed()
        else:
            state.record_question(decision.next_slot, decision.question)

        self._store.save(state)
        return state, decision

    # ------------------------------------------------------------------
    # اتخاذ القرار
    # ------------------------------------------------------------------
    def _decide(self, state: ConversationState) -> InterviewDecision:
        """حدّ أقصى للأسئلة كحماية، وإلا يختار الـ LLM السؤال التالي."""
        if len(state.asked_questions) >= self._max_questions:
            logger.info("بلوغ الحد الأقصى للأسئلة — إنهاء المقابلة.")
            return InterviewDecision(finished=True)

        system_prompt = self._prompts.system_prompt()
        user_prompt = self._prompts.turn_prompt(state)

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("فشل استدعاء مزوّد الـ LLM.")
            raise LLMProviderError(f"فشل استدعاء الـ LLM: {exc}") from exc

        decision = parse_interview_decision(completion.text)

        # منع تكرار خانة سبق السؤال عنها: نتجاهل القرار ونُنهي بدل الإعادة.
        if not decision.finished and decision.next_slot in state.asked_slots:
            logger.warning(
                "أعاد الـ LLM خانة مكرّرة (%s) — إنهاء المقابلة.", decision.next_slot
            )
            return InterviewDecision(finished=True)

        return decision
