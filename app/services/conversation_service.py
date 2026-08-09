from __future__ import annotations

import logging
from copy import deepcopy
from typing import List, Optional, Tuple

from app.domain import clinical
from app.domain.clinical_priority import ClinicalPriorityEngine
from app.domain.clinical_record import FactProvenance, FactSource
from app.domain.conversation import (
    ConversationState,
    InterviewDecision,
    InterviewTurnOutput,
    Symptom,
)
from app.domain.ports import LLMProvider, SessionStore
from app.domain.red_flag_engine import RedFlagEngine, normalize_arabic
from app.domain.red_flags import RedFlagAssessment
from app.exceptions import LLMProviderError
from app.parsing.interview_parser import parse_interview_turn
from app.prompts.interview_builder import InterviewPromptBuilder

logger = logging.getLogger(__name__)

# أدنى طول للشاهد يُعتدّ به. مقطع قصير جداً ("من"، "لا") يرد في أي نصّ
# تقريباً، فإثباته بلا قيمة ويمنح تأكيداً زائفاً.
_MIN_EVIDENCE_LENGTH = 4


class ConversationService:
    """يدير دوراً واحداً من مقابلة أخذ التاريخ المرضي."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: InterviewPromptBuilder,
        store: SessionStore,
        max_questions: int = 8,
        red_flag_engine: Optional[RedFlagEngine] = None,
        priority_engine: Optional[ClinicalPriorityEngine] = None,
    ) -> None:
        self._provider = provider
        self._prompts = prompt_builder
        self._store = store
        self._max_questions = int(max_questions)
        # اختياري بالتوقيع للتوافق الخلفي مع الاختبارات القائمة، لكنّه
        # مُحقَن دائماً في الإنتاج (main.py). غيابه يعني تعطّل كشف الطوارئ،
        # ولذلك يُسجَّل تحذيراً صريحاً بدل المرور صامتاً.
        self._red_flags = red_flag_engine
        if red_flag_engine is None:
            logger.warning(
                "ConversationService بلا محرّك أعلام حمراء — كشف الطوارئ معطّل."
            )
        # اختياري بالتوقيع للتوافق الخلفي، ومُحقَن دائماً في الإنتاج. عند
        # غيابه يتدهور المحرك إلى السلوك القديم (أوّل عرَض) بدل الانهيار.
        self._priority = priority_engine

    def handle_message(
        self, text: str, session_id: Optional[str] = None
    ) -> Tuple[ConversationState, InterviewDecision]:
        """معالجة رسالة مريض وإرجاع (الحالة المحدَّثة، القرار).

        الدور **ذرّي**: كل التحوّرات تقع على نسخة عمل، ولا تُثبَّت في المخزن
        إلا عند نجاح الدور. سابقاً كان ``get_or_create`` يُعيد مرجعاً مشتركاً،
        فتُثبَّت الرسالة وعدّاد الأدوار فور تسجيلهما — وعند فشل الـLLM يعود
        الراوت بـ502 بينما التحوّر الجزئي مثبَّت أصلاً، فتُكرَّر الرسالة وينتفخ
        العدّاد مع أول إعادة محاولة من العميل.
        """
        # الجلسة معروفة سلفاً؟ يُقرَّر **قبل** الإنشاء: الاعتماد على
        # ``turn_count == 0`` كان سيرفع راية "أُعيد البدء" زوراً عند إعادة
        # محاولة الدور الأول (الجلسة أُنشئت لكن لم يُثبَّت لها دور بعد).
        known = bool(session_id) and self._store.get(session_id) is not None
        committed = self._store.get_or_create(session_id)

        # نسخة عمل معزولة — لا شيء يُرى في المخزن قبل التثبيت.
        state = deepcopy(committed)

        state.session_restarted = bool(session_id) and not known
        if state.session_restarted:
            logger.info("جلسة غير معروفة أو منتهية (%s) — بدء سجلّ جديد.", session_id)

        # 1) الرسالة الحالية هي ردّ المريض على سؤال الدور السابق — نمسح التعليق.
        #    لا نُسند نصّها إلى خانة بعينها (قد لا يُجيب المريض عن السؤال مباشرةً)؛
        #    المعلومة الكاملة تُحفظ في raw_messages وتصل للـ LLM كما هي.
        state.pending_slot = None

        # 2) حفظ الرسالة الخام كاملةً (سياق كامل للـ LLM، لا يُفقد أي شيء).
        state.record_patient_message(text)

        state.turn_count += 1

        # 3) [L0] فحص الطوارئ الحتمي **قبل** الـLLM وبمعزل عن الشبكة.
        #    هذه الخاصية المعمارية المركزية: حتى لو انهار المزوّد تماماً،
        #    يكون المريض قد حصل على حكم سلامة. لا يجوز أن تكون السلامة
        #    خلف اعتماد شبكي.
        l0 = self._screen_raw(state)

        # 4) [L1] استدعاء واحد: استخراج منظَّم + قرار الدور.
        #    يُستدعى حتى عند بلوغ سقف الأسئلة (حيث سيُفرَض الإنهاء لاحقاً):
        #    رسالة المريض الأخيرة قد تحمل معلومات طبية مهمّة، وتخطّي الاستدعاء
        #    كان سيُسقطها من السجل نهائياً.
        try:
            turn = self._run_turn(state)
        except LLMProviderError:
            # مسار متدهور: لا استخراج، لكن حكم L0 يبقى صالحاً وكافياً
            # لتحذير المريض. الإخفاق يُبتلَع **فقط** عند وجود طارئ فعلي؛
            # وإلا يُرفع ليعالجه الراوت كخطأ عادي (502).
            l0.degraded = True
            state.risk = state.risk.merged_with(l0)
            if l0.is_emergency:
                # طارئ مكتشف = نتيجة صالحة تُثبَّت: الدور انتهى بحكم سلامة
                # حقيقي، وإعادة المحاولة لا تُفيد المريض بشيء.
                logger.warning(
                    "فشل الـLLM لكن L0 كشف طارئاً — إرجاع تحذير متدهور."
                )
                state.mark_completed()
                self._store.save(state)
                return state, InterviewDecision(finished=True)
            # بلا طارئ: **لا تثبيت**. نسخة العمل تُهمَل، فتبقى الجلسة كما
            # كانت قبل الدور، وإعادة المحاولة تُسجّل الرسالة مرّة واحدة.
            raise

        # 5) التحقّق الحتمي من الشواهد قبل الدمج — لا يُمنَح النموذج صفة
        #    "مؤكَّد" بتصريحه عن نفسه، بل بوجود شاهده في كلام المريض فعلاً.
        verified = self._verify_symptoms(turn.symptoms, state)
        state.add_symptoms(verified)
        state.record.merge(
            turn.record,
            turn_number=state.turn_count,
            provenance=self._record_provenance(turn, state),
        )

        # 6) [L2] فحص حتمي ثانٍ على السجل المنظَّم، ثم [L3] تصعيد فقط.
        #    الأعراض المُصنَّفة استنتاجاً تُشارك في الفحص عمداً: التصعيد أأمن
        #    من الإسقاط — ثمن الإحالة الزائدة أهون من طارئ مُغفَل.
        state.risk = state.risk.merged_with(l0).merged_with(self._screen_record(state))

        # 7) [P1] إعادة حساب الشكوى الرئيسية حتمياً بعد كل رسالة.
        #    يقع **بعد** تقييم الخطورة كي تدخل حالة الطوارئ في الترتيب،
        #    و**قبل** الحارس كي تُبنى قائمة الأسئلة حول الشكوى الجديدة.
        #    الـLLM لا يُستشار هنا إطلاقاً — القرار للمحرّك وحده.
        self._recompute_primary(state)

        # 8) حرّاس المجال على القرار.
        decision = self._guard(state, turn.decision)

        # 6) تطبيق القرار على الحالة.
        if decision.finished:
            state.mark_completed()
        else:
            state.record_question(decision.next_slot, decision.question)

        self._store.save(state)
        return state, decision

    # ------------------------------------------------------------------
    # [P1] الأولوية السريرية
    # ------------------------------------------------------------------
    def _recompute_primary(self, state: ConversationState) -> None:
        """يُعيد تحديد الشكوى الرئيسية الحالية بعد كل رسالة.

        القرار للمحرّك حصراً: يُعاد الحساب من الصفر في كل دور بدل تثبيت
        أوّل عرَض، فيتحوّل محور المقابلة فوراً حين يظهر عرَض أعلى حدّة
        (مثال: صداع في الدور الأول ثم ألم صدر ضاغط في الدور الثالث).
        """
        if self._priority is None:
            state.primary_complaint = None
            state.symptom_priorities = []
            return

        ranked = self._priority.rank(
            state.symptoms, state.raw_messages, state.risk
        )
        previous = state.primary_complaint
        state.symptom_priorities = ranked
        state.primary_complaint = ranked[0].symptom_text if ranked else None

        if previous and state.primary_complaint != previous:
            logger.info(
                "تحوّل محور المقابلة: %r -> %r (%s)",
                previous, state.primary_complaint,
                ranked[0].rationale if ranked else "-",
            )

    def _redirect_off_topic(
        self, state: ConversationState, decision: InterviewDecision
    ) -> Optional[InterviewDecision]:
        """يُعيد توجيه السؤال إلى الشكوى الرئيسية إن كان يستهدف غيرها.

        القاعدة **ضيّقة عمداً**: يُعاد التوجيه فقط حين يذكر السؤال عرَضاً
        آخر معروفاً **ولا** يذكر الشكوى الرئيسية. سؤال عام ("متى بدأ؟")
        يُترك للنموذج، فصياغته أطبع من قالب ثابت — التدخّل عند الانحراف
        المُثبَت لا عند كل سؤال.

        يُعاد ``None`` حين لا داعي للتدخّل.
        """
        primary = state.primary_complaint
        if decision.finished or not primary or not decision.question:
            return None

        question = normalize_arabic(decision.question)
        if normalize_arabic(primary) in question:
            return None   # يستهدف الشكوى الرئيسية فعلاً

        others = [
            s.text for s in state.symptoms
            if not s.negated and s.text != primary
        ]
        if not any(normalize_arabic(o) in question for o in others):
            return None   # سؤال عام — لا انحراف مُثبَت

        item = clinical.next_missing(
            state.symptoms, " ".join(state.raw_messages), state.asked_slots,
            primary=primary,
        )
        if item is None:
            return None

        target, question_text = item
        logger.info(
            "سؤال يستهدف عرَضاً غير الشكوى الرئيسية (%r) — إعادة توجيه إلى %r.",
            primary, target,
        )
        return InterviewDecision(
            finished=False, next_slot=target, question=question_text
        )

    # ------------------------------------------------------------------
    # طبقات فحص الطوارئ الحتمية
    # ------------------------------------------------------------------
    def _screen_raw(self, state: ConversationState) -> RedFlagAssessment:
        """[L0] فحص على النصّ الخام — بلا أي اعتماد على الـLLM أو الشبكة."""
        if self._red_flags is None:
            return RedFlagAssessment(evaluated=False)
        return self._red_flags.evaluate_raw_text(state.raw_messages)

    def _screen_record(self, state: ConversationState) -> RedFlagAssessment:
        """[L2] فحص على السجل المنظَّم بعد الاستخراج.

        تُمرَّر الأعراض غير المنفية فقط (النفي مقروء بنيوياً هنا، وهو أدقّ من
        الاستنتاج اللفظي في L0). الأعراض المُصنَّفة ``LLM_INFERRED`` تُمرَّر
        أيضاً — قرار سلامة صريح: التصعيد أأمن من الإسقاط.
        """
        if self._red_flags is None:
            return RedFlagAssessment(evaluated=False)

        positives = [s.text for s in state.symptoms if not s.negated]
        record = state.record
        record_texts = [
            value for value in (
                record.chief_complaint, record.severity,
                record.duration, record.body_location,
            ) if value
        ]
        record_texts += [*record.chronic_conditions, *record.allergies]

        return self._red_flags.evaluate_record(
            symptom_texts_positive=positives,
            record_texts=record_texts,
            raw_texts=state.raw_messages,
        )

    # ------------------------------------------------------------------
    # التحقّق الحتمي من إثبات المصدر
    # ------------------------------------------------------------------
    def _verify_symptoms(
        self, symptoms: List[Symptom], state: ConversationState
    ) -> List[Symptom]:
        """يضبط ``source`` لكل عرَض بالتحقّق من شاهده في كلام المريض.

        لماذا حتمي لا بتصريح النموذج: لو سألنا النموذج "هل هذا استنتاج؟"
        لكان مصدر الحكم هو نفسه مصدر الخطأ المحتمل. البحث عن الشاهد داخل
        الرسائل الخام فحص مستقل: نموذج يختلق عرَضاً سيختلق له شاهداً لا
        يوجد في كلام المريض، فيُصنَّف ``LLM_INFERRED`` تلقائياً.

        لا يُحذف أي عرَض هنا — الاستنتاج معلومة مشروعة، لكنه يُوسَم بوضوح
        كي تزنه الطبقات اللاحقة بما يستحقّ.
        """
        haystack = normalize_arabic(" ".join(state.raw_messages))

        verified: List[Symptom] = []
        for symptom in symptoms:
            evidence = symptom.evidence
            normalized = normalize_arabic(evidence) if evidence else ""

            if normalized and len(normalized) >= _MIN_EVIDENCE_LENGTH \
                    and normalized in haystack:
                source = FactSource.PATIENT_EXPLICIT
            elif normalize_arabic(symptom.text) in haystack:
                # نصّ العرَض نفسه ورد حرفياً وإن لم يُقدَّم شاهد صالح.
                source = FactSource.PATIENT_EXPLICIT
                evidence = evidence or symptom.text
            else:
                source = FactSource.LLM_INFERRED
                if evidence:
                    logger.info(
                        "شاهد غير موجود في كلام المريض — وُسم العرَض استنتاجاً."
                    )

            verified.append(Symptom(
                text=symptom.text,
                negated=symptom.negated,
                confidence=symptom.confidence,
                evidence=evidence,
                source=source,
                turn_number=state.turn_count,
            ))
        return verified

    @staticmethod
    def _record_provenance(
        turn: InterviewTurnOutput, state: ConversationState
    ) -> dict:
        """أثر منشأ حقول السجل المفردة والقوائم.

        قيد صريح: العقد الحالي لا يطلب شاهداً لكل حقل من حقول السجل (طلبه
        لكل حقل يضخّم الرموز بلا مقابل يوازيه). لذلك يُتحقَّق من ورود قيمة
        الحقل نفسها في كلام المريض — أضعف من شاهد مخصَّص لكنه فحص مستقل
        حقيقي، لا تصريح ذاتي من النموذج.
        """
        haystack = normalize_arabic(" ".join(state.raw_messages))
        provenance = {}

        for name in ("chief_complaint", "severity", "duration", "body_location"):
            value = getattr(turn.record, name)
            if not value:
                continue
            found = normalize_arabic(value) in haystack
            provenance[name] = FactProvenance(
                source=(FactSource.PATIENT_EXPLICIT if found
                        else FactSource.LLM_INFERRED),
                evidence=value if found else None,
                turn_number=state.turn_count,
            )

        for name in ("medications", "allergies",
                     "chronic_conditions", "family_history"):
            values = getattr(turn.record, name) or []
            if not values:
                continue
            found_all = all(normalize_arabic(v) in haystack for v in values)
            provenance[name] = FactProvenance(
                source=(FactSource.PATIENT_EXPLICIT if found_all
                        else FactSource.LLM_INFERRED),
                turn_number=state.turn_count,
            )

        return provenance

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
        # [L4] الطوارئ تُنهي المقابلة فوراً — وتسبق كل حارس آخر.
        # استجواب مريض بأعراض إنذارية عن "شدّة الألم من 1 إلى 10" تأخيرٌ
        # ضارّ؛ الصواب إنهاء الجمع وتسليم التوجيه الطارئ حالاً.
        if state.risk.is_emergency:
            logger.warning(
                "علم أحمر (%s) — إنهاء المقابلة فوراً وتسليم توجيه طارئ.",
                state.risk.risk_level.value,
            )
            return InterviewDecision(finished=True)

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
                state.symptoms, " ".join(state.raw_messages), state.asked_slots,
                primary=state.primary_complaint,
            )
            if item is not None:
                target, question = item
                logger.info("منع إنهاء المقابلة بلا أعراض — إعادة سؤال الشكوى.")
                return InterviewDecision(
                    finished=False, next_slot=target, question=question
                )

        # [P1] السؤال يجب أن يستهدف الشكوى الرئيسية التي قرّرها المحرّك.
        # النموذج قد يواصل السؤال عن عرَض قديم بعد تحوّل المحور (لوحظ حيّاً:
        # المحرّك انتقل إلى ألم الصدر والنموذج ظلّ يسأل عن الصداع). سلطة
        # المحرّك لا تكتمل بتحديد المحور وحده دون فرضه على السؤال.
        redirected = self._redirect_off_topic(state, decision)
        if redirected is not None:
            return redirected

        # منع تكرار خانة سبق السؤال عنها: نتجاهل القرار ونُنهي بدل الإعادة.
        if not decision.finished and decision.next_slot in state.asked_slots:
            logger.warning(
                "أعاد الـ LLM خانة مكرّرة (%s) — إنهاء المقابلة.", decision.next_slot
            )
            return InterviewDecision(finished=True)

        return decision
