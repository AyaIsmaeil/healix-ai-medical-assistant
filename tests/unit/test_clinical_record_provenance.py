"""
اختبارات إثبات المصدر ومنع الاستبدال الصامت (W3 + W4).

الفكرة المركزية المُختبَرة هنا: **النموذج لا يمنح نفسه صفة "مؤكَّد"**.
التأكيد يأتي من فحص مستقل — هل يرد الشاهد فعلاً في كلام المريض؟ — لا من
تصريح النموذج عن ثقته أو عن كون المعلومة مذكورة.
"""

import json

import pytest

from app.domain.clinical_record import (
    ClinicalRecord,
    FactProvenance,
    FactSource,
)
from app.domain.conversation import Symptom
from app.domain.ports import Completion
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService


def sym(text, evidence=None, negated=False, confidence=0.9):
    return {"text": text, "negated": negated,
            "confidence": confidence, "evidence": evidence}


class Scripted:
    name = "scripted"

    def __init__(self, payloads):
        self._payloads = payloads
        self._i = 0

    def generate(self, system_prompt, user_prompt):
        payload = self._payloads[self._i]
        self._i += 1
        return Completion(json.dumps(payload, ensure_ascii=False), model="scripted")


class ScriptedSymptomExtractor:
    """بديل اختباري لـ CompositeSymptomExtractor — أعراض (مع شواهدها) محدّدة
    مسبقاً بالترتيب. الأعراض تُستخرَج بمسار مستقلّ عن ردّ الـLLM هنا، فلا
    تصل عبر حقل ``symptoms`` داخل ``turn()``."""

    def __init__(self, batches=None):
        self._batches = batches or []
        self._i = 0

    def extract(self, raw_messages, known_symptoms=None):
        if self._i >= len(self._batches):
            return []
        batch = self._batches[self._i]
        self._i += 1
        return [
            Symptom(
                text=s["text"], negated=s.get("negated", False),
                confidence=s.get("confidence", 0.9), evidence=s.get("evidence"),
            )
            for s in batch
        ]


def turn(**over):
    base = {
        "chief_complaint": None, "severity": None,
        "duration": None, "body_location": None, "medications": [],
        "allergies": [], "chronic_conditions": [], "family_history": [],
        "missing_fields": [], "finished": False,
        "next_slot": "onset", "question": "منذ متى؟",
    }
    base.update(over)
    return base


def service(payloads, symptom_batches=None):
    return ConversationService(
        provider=Scripted(payloads),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        symptom_extractor=ScriptedSymptomExtractor(symptom_batches),
    )


# ----------------------------------------------------------------------
# W3 — لا استبدال صامت
# ----------------------------------------------------------------------
def test_scalar_change_is_applied_but_recorded():
    """القيمة الجديدة تُطبَّق (المريض قد يصحّح نفسه) لكنها تُسجَّل."""
    record = ClinicalRecord(chief_complaint="ألم صدر")
    record.merge(ClinicalRecord(chief_complaint="صداع"), turn_number=2)

    assert record.chief_complaint == "صداع"
    assert len(record.revisions) == 1
    revision = record.revisions[0]
    assert revision.old_value == "ألم صدر"
    assert revision.new_value == "صداع"
    assert revision.turn_number == 2
    assert revision.is_contradiction is True


def test_first_value_is_not_a_contradiction():
    record = ClinicalRecord()
    record.merge(ClinicalRecord(chief_complaint="صداع"), turn_number=1)

    assert record.revisions[0].is_contradiction is False
    assert record.contradictions == []


def test_empty_incoming_never_erases():
    record = ClinicalRecord(chief_complaint="صداع", severity="شديد")
    record.merge(ClinicalRecord(chief_complaint=None, severity=""), turn_number=2)

    assert record.chief_complaint == "صداع"
    assert record.severity == "شديد"
    assert record.revisions == []      # لا تغيير ⇒ لا مراجعة


def test_contradictions_are_queryable():
    record = ClinicalRecord(duration="يومين")
    record.merge(ClinicalRecord(duration="شهر"), turn_number=3)

    assert [r.field_name for r in record.contradictions] == ["duration"]


# ----------------------------------------------------------------------
# W4 — إثبات المصدر
# ----------------------------------------------------------------------
def test_verified_evidence_marks_symptom_as_patient_stated():
    svc = service([turn()], symptom_batches=[
        [sym("صداع", evidence="عندي صداع شديد")],
    ])
    state, _ = svc.handle_message("عندي صداع شديد من امبارح", None)

    symptom = state.symptoms[0]
    assert symptom.source is FactSource.PATIENT_EXPLICIT
    assert symptom.is_patient_stated


def test_fabricated_evidence_is_demoted_to_inferred():
    """الاختبار الأهم: شاهد لا يرد في كلام المريض لا يمنح تأكيداً."""
    svc = service([turn()], symptom_batches=[
        [sym("ألم صدر", evidence="المريض ذكر ألماً شديداً في الصدر")],
    ])
    state, _ = svc.handle_message("عندي صداع من امبارح", None)

    symptom = state.symptoms[0]
    assert symptom.source is FactSource.LLM_INFERRED
    assert not symptom.is_patient_stated


def test_inferred_symptom_is_kept_not_deleted():
    """الاستنتاج معلومة مشروعة — يُوسَم ولا يُحذف."""
    svc = service([turn()], symptom_batches=[[sym("ألم صدر", evidence=None)]])
    state, _ = svc.handle_message("عندي صداع", None)

    assert [s.text for s in state.symptoms] == ["ألم صدر"]
    assert state.symptoms[0].source is FactSource.LLM_INFERRED


def test_symptom_text_present_in_message_verifies_without_evidence():
    """إن ورد نصّ العرَض حرفياً فلا حاجة لشاهد منفصل."""
    svc = service([turn()], symptom_batches=[[sym("صداع", evidence=None)]])
    state, _ = svc.handle_message("عندي صداع", None)

    assert state.symptoms[0].source is FactSource.PATIENT_EXPLICIT


def test_trivially_short_evidence_does_not_verify():
    """مقطع قصير جداً يرد في أي نصّ — إثباته بلا قيمة."""
    svc = service([turn()], symptom_batches=[[sym("ألم صدر", evidence="من")]])
    state, _ = svc.handle_message("عندي صداع من امبارح", None)

    assert state.symptoms[0].source is FactSource.LLM_INFERRED


def test_evidence_verification_ignores_diacritics_and_spelling_variants():
    svc = service([turn()], symptom_batches=[
        [sym("صداع", evidence="عندي صُداع شديد")],
    ])
    state, _ = svc.handle_message("عندي صداع شديد", None)

    assert state.symptoms[0].source is FactSource.PATIENT_EXPLICIT


def test_invented_medication_is_flagged_unverified():
    svc = service([turn(medications=["أسبرين"])])
    state, _ = svc.handle_message("عندي صداع", None)

    assert state.record.medications == ["أسبرين"]     # لا يُحذف
    assert "medications" in state.record.unverified_fields()


def test_patient_stated_medication_is_verified():
    svc = service([turn(medications=["بنادول"])])
    state, _ = svc.handle_message("بآخذ بنادول للصداع", None)

    assert "medications" not in state.record.unverified_fields()


def test_provenance_carries_turn_number():
    svc = service([
        turn(),
        turn(next_slot="severity", question="ما الشدة؟"),
    ], symptom_batches=[
        [sym("صداع", evidence="عندي صداع")],
        [sym("حرارة", evidence="وصار عندي حرارة")],
    ])
    state, _ = svc.handle_message("عندي صداع", None)
    state, _ = svc.handle_message("وصار عندي حرارة", state.session_id)

    by_text = {s.text: s for s in state.symptoms}
    assert by_text["صداع"].turn_number == 1
    assert by_text["حرارة"].turn_number == 2


# ----------------------------------------------------------------------
# عدم الكسر
# ----------------------------------------------------------------------
def test_legacy_symptom_construction_still_works():
    """المُنشئ القديم بثلاثة معاملات يبقى صالحاً."""
    from app.domain.conversation import Symptom

    symptom = Symptom("صداع", False, 0.9)
    assert symptom.evidence is None
    assert symptom.source is FactSource.UNKNOWN


def test_record_scalar_fields_remain_plain_values():
    """الحقول لم تُغلَّف في كائنات — كل قارئ حالي يبقى يعمل."""
    record = ClinicalRecord()
    record.merge(ClinicalRecord(chief_complaint="صداع"), turn_number=1)

    assert isinstance(record.chief_complaint, str)


def test_merge_without_turn_number_still_works():
    """التوقيع القديم merge(incoming) يبقى صالحاً."""
    record = ClinicalRecord()
    record.merge(ClinicalRecord(severity="شديد"))
    assert record.severity == "شديد"
