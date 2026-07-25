"""
Healix - Conversation Domain
كيانات محرك المحادثة (طبقة المجال الخالصة).

لا تعتمد هذه الوحدة على FastAPI أو torch أو أي بنية تحتية — منطق حالة
المقابلة الطبية فقط، ما يجعلها قابلة للاختبار بمعزل تام.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

from app.domain.clinical_record import ClinicalRecord


class InterviewStatus(str, Enum):
    """حالة المقابلة."""

    COLLECTING = "collecting"  # ما زلنا نجمع المعلومات
    COMPLETED = "completed"    # اكتفى المحرك من الأسئلة (finished)


@dataclass
class Symptom:
    """عرض مستخرَج مخزَّن في الجلسة (مستقل عن طبقة الاستخراج)."""

    text: str
    negated: bool
    confidence: float


@dataclass
class AskedQuestion:
    """سؤال طُرح سابقاً — يُستخدم لمنع التكرار."""

    slot: str
    question: str
    turn: int


@dataclass
class ConversationState:
    """
    حالة محادثة مريض واحد (aggregate root).

    تحتوي: كامل رسائل المريض الخام، الأعراض المستخرجة، الأسئلة المطروحة، وعدّاد
    الأدوار. كل تعديل يمرّ عبر توابع صريحة للحفاظ على الثبات (invariants).

    ملاحظة تصميمية: لا نُسند إجابة المريض إلى خانة بعينها (تجنّباً لعدم التطابق
    عندما لا يُجيب المريض عن السؤال المطروح مباشرةً). المعلومات الكاملة تبقى في
    ``raw_messages``، والخانات المطروحة في ``asked_questions`` لمنع التكرار فقط.
    """

    session_id: str
    status: InterviewStatus = InterviewStatus.COLLECTING
    turn_count: int = 0
    # كامل رسائل المريض الخام بترتيب ورودها (تُحفظ كما هي، لا تُفقَد أي معلومة).
    raw_messages: List[str] = field(default_factory=list)
    symptoms: List[Symptom] = field(default_factory=list)
    asked_questions: List[AskedQuestion] = field(default_factory=list)
    # الخانة التي طُرح سؤالها في الدور السابق (نمسحها عند وصول الرسالة التالية).
    pending_slot: Optional[str] = None
    # السجل الطبي المنظَّم المتراكم (يملؤه الـLLM دوراً بعد دور، بلا فقدان).
    record: ClinicalRecord = field(default_factory=ClinicalRecord)

    # ------------------------------------------------------------------
    # الاستعلامات
    # ------------------------------------------------------------------
    @property
    def asked_slots(self) -> List[str]:
        """قائمة الخانات التي سبق السؤال عنها (بترتيب الطرح)."""
        return [q.slot for q in self.asked_questions]

    # ------------------------------------------------------------------
    # التعديلات
    # ------------------------------------------------------------------
    def record_patient_message(self, text: str) -> None:
        """حفظ الرسالة الخام للمريض لهذا الدور (تُضاف ولا تُستبدل)."""
        self.raw_messages.append(text)

    def add_symptoms(self, new_symptoms: Iterable[Symptom]) -> None:
        """دمج أعراض جديدة مع إزالة التكرار على (النص، النفي)."""
        existing = {(s.text, s.negated) for s in self.symptoms}
        for symptom in new_symptoms:
            key = (symptom.text, symptom.negated)
            if key not in existing:
                self.symptoms.append(symptom)
                existing.add(key)

    def record_question(self, slot: str, question: str) -> None:
        """تسجيل سؤال جديد وتعيينه كخانة معلّقة تنتظر الإجابة."""
        self.asked_questions.append(
            AskedQuestion(slot=slot, question=question, turn=self.turn_count)
        )
        self.pending_slot = slot

    def mark_completed(self) -> None:
        """إنهاء المقابلة."""
        self.status = InterviewStatus.COMPLETED
        self.pending_slot = None


@dataclass
class InterviewDecision:
    """
    قرار المحرك لدور واحد (ناتج عن الـ LLM بعد التحليل).

    - ``finished=True``  → اكتفينا، لا سؤال.
    - ``finished=False`` → ``next_slot`` و ``question`` مطلوبان.
    """

    finished: bool
    next_slot: Optional[str] = None
    question: Optional[str] = None


@dataclass
class InterviewTurnOutput:
    """مخرجات استدعاء LLM واحد لدور واحد: الاستخراج + القرار معاً.

    بعد دمج الاستخراج داخل المقابلة السريرية صار استدعاء الـLLM الواحد يُنتج
    الاثنين في JSON واحد — فلا استدعاء ثانٍ ولا منطق استخراج مكرَّر. فصل
    ``decision`` عن ``record`` يُبقي منطق القرار (حراس المجال، منع التكرار)
    كما هو دون أن يعرف شيئاً عن حقول السجل.
    """

    decision: InterviewDecision
    record: ClinicalRecord = field(default_factory=ClinicalRecord)
    symptoms: List[Symptom] = field(default_factory=list)
