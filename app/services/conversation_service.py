from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.domain import clinical
from app.domain.conversation import (
    ConversationState,
    InterviewDecision,
    InterviewTurnOutput,
)
from app.domain.ports import LLMProvider, SessionStore
from app.exceptions import LLMProviderError
from app.parsing.interview_parser import parse_interview_turn
from app.prompts.interview_builder import InterviewPromptBuilder

logger = logging.getLogger(__name__)


class ConversationService:
    """يدير دوراً واحداً من مقابلة أخذ التاريخ المرضي."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: InterviewPromptBuilder,
        store: SessionStore,
        max_questions: int = 8,
    ) -> None:
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

        state.turn_count += 1

        # 3) استدعاء واحد: استخراج منظَّم + قرار الدور.
        #    يُستدعى حتى عند بلوغ سقف الأسئلة (حيث سيُفرَض الإنهاء لاحقاً):
        #    رسالة المريض الأخيرة قد تحمل معلومات طبية مهمّة، وتخطّي الاستدعاء
        #    كان سيُسقطها من السجل نهائياً. سابقاً كان الاستخراج مرحلة منفصلة
        #    فحصل عليها مجاناً؛ بعد الدمج صار لزاماً استدعاء واحد لضمانها.
        turn = self._run_turn(state)

        # 4) دمج المخرجات في الجلسة قبل حرّاس المجال (فالحرّاس يقرأون الأعراض).
        state.add_symptoms(turn.symptoms)
        state.record.merge(turn.record)

        # 5) حرّاس المجال على القرار.
        decision = self._guard(state, turn.decision)

        # 6) تطبيق القرار على الحالة.
        if decision.finished:
            state.mark_completed()
        else:
            state.record_question(decision.next_slot, decision.question)

        self._store.save(state)
        return state, decision

    # استدعاء الـLLM
    def _run_turn(self, state: ConversationState) -> InterviewTurnOutput:
        """استدعاء المزوّد مرّة واحدة وتحليل العقد الموحّد."""
        system_prompt = self._prompts.system_prompt()
        user_prompt = self._prompts.turn_prompt(state)

        try:
            completion = self._provider.generate(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("فشل استدعاء مزوّد الـ LLM.")
            raise LLMProviderError(f"فشل استدعاء الـ LLM: {exc}") from exc

        return parse_interview_turn(completion.text)

    # ------------------------------------------------------------------
    # حرّاس المجال على القرار
    # ------------------------------------------------------------------
    def _guard(
        self, state: ConversationState, decision: InterviewDecision
    ) -> InterviewDecision:
        """قواعد المحرك التي تعلو على اقتراح النموذج."""
        # حدّ أقصى للأسئلة — حماية من الحلقات اللانهائية.
        if len(state.asked_questions) >= self._max_questions:
            logger.info("بلوغ الحد الأقصى للأسئلة — إنهاء المقابلة.")
            return InterviewDecision(finished=True)

        # لا تنتهي المقابلة قبل الحصول على أي عرَض مُثبَت. الـLLM قد يُنهيها بعد
        # "مرحبا/كيفك" لأنّه لم يجد ما يسأل عنه — وهذا القرار الطبي ملك المحرك
        # لا النموذج، فنُصرّ على سؤال الشكوى الرئيسية بشكل حتمي.
        #
        # الشرط ``decision.finished`` أساسي: الحارس يتدخّل **فقط** عند محاولة
        # الإنهاء. ما دام الـLLM يسأل، فسؤاله هو المعتمد حتى قبل استخراج أي
        # عرَض — هو وكيل المقابلة وصاحب صياغة الأسئلة، والمحرك لا يزاحمه.
        # (سابقاً كان التجاوز يقع كلّما غاب العرَض المُثبَت، فيستبدل سؤال
        # النموذج بسؤال ثابت يكاد يطابق تحية البداية → تكرار يراه المريض.)
        if decision.finished and not any(not s.negated for s in state.symptoms):
            item = clinical.next_missing(
                state.symptoms, " ".join(state.raw_messages), state.asked_slots
            )
            if item is not None:
                target, question = item
                logger.info("منع إنهاء المقابلة بلا أعراض — إعادة سؤال الشكوى.")
                return InterviewDecision(
                    finished=False, next_slot=target, question=question
                )

        # منع تكرار خانة سبق السؤال عنها: نتجاهل القرار ونُنهي بدل الإعادة.
        if not decision.finished and decision.next_slot in state.asked_slots:
            logger.warning(
                "أعاد الـ LLM خانة مكرّرة (%s) — إنهاء المقابلة.", decision.next_slot
            )
            return InterviewDecision(finished=True)

        return decision
