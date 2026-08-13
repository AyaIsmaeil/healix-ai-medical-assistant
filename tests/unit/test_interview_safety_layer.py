"""
اختبارات تكامل طبقة السلامة داخل وكيل المقابلة (W1).

تُغطّي الخصائص المعمارية الأربع:
  L0 — الفحص يسبق الـLLM ولا يعتمد عليه.
  L2 — فحص ثانٍ على السجل المنظَّم بعد الاستخراج.
  L3 — التصعيد فقط، لا تخفيض من أي طبقة.
  L4 — الطوارئ تُنهي المقابلة فوراً بدل مواصلة الاستجواب.
"""

import json

import pytest

from app.domain.conversation import Symptom
from app.domain.ports import Completion
from app.domain.red_flag_engine import RedFlagEngine
from app.domain.red_flags import RiskLevel
from app.exceptions import LLMProviderError
from app.infrastructure.dictionary_loader import DictionaryLoader
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService


@pytest.fixture(scope="module")
def flags() -> RedFlagEngine:
    return RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())


def payload(**over):
    base = {
        "chief_complaint": None, "symptoms": [], "severity": None,
        "duration": None, "body_location": None, "medications": [],
        "allergies": [], "chronic_conditions": [], "family_history": [],
        "missing_fields": [], "finished": False,
        "next_slot": "onset", "question": "منذ متى؟",
    }
    base.update(over)
    return base


class Scripted:
    name = "scripted"

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self._i = 0

    def generate(self, system_prompt, user_prompt):
        data = self._payloads[min(self._i, len(self._payloads) - 1)]
        self._i += 1
        return Completion(json.dumps(data, ensure_ascii=False), model="scripted")


class Broken:
    """يحاكي انهيار OpenRouter/الشبكة بالكامل."""

    name = "broken"

    def generate(self, system_prompt, user_prompt):
        raise RuntimeError("OpenRouter unavailable")


class ScriptedSymptomExtractor:
    """بديل اختباري لـ CompositeSymptomExtractor — أعراض محدّدة مسبقاً بالترتيب.

    الأعراض تُستخرَج بمسار مستقلّ عن ردّ الـLLM هنا، فلا تصل عبر حقل
    ``symptoms`` داخل ``payload()``."""

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


def service(provider, flags, max_questions=8, symptom_batches=None):
    return ConversationService(
        provider=provider,
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=max_questions,
        red_flag_engine=flags,
        symptom_extractor=ScriptedSymptomExtractor(symptom_batches),
    )


ACS_MESSAGE = "عندي ألم ضاغط بصدري وضيق نفس وعرق بارد"


# ----------------------------------------------------------------------
# L4 — الطوارئ تُنهي المقابلة
# ----------------------------------------------------------------------
def test_emergency_ends_interview_immediately(flags):
    """لا يجوز استجواب مريض بأعراض إنذارية عن شدّة الألم."""
    svc = service(Scripted(payload(next_slot="severity", question="ما شدّة الألم؟")), flags)
    state, decision = svc.handle_message(ACS_MESSAGE, None)

    assert state.risk.is_emergency
    assert decision.finished is True          # فُرض الإنهاء رغم اقتراح النموذج
    assert state.status.value == "completed"


def test_emergency_supplies_actionable_guidance(flags):
    svc = service(Scripted(payload()), flags)
    state, _ = svc.handle_message(ACS_MESSAGE, None)

    action = state.risk.primary_action_ar
    assert action and "طوارئ" in action
    # حدّ المجال: توجيه لرعاية، لا تسمية مرض.
    assert "احتشاء" not in action and "تشخيص" not in action


def test_ordinary_complaint_does_not_trigger_emergency(flags):
    svc = service(Scripted(payload()), flags)
    state, decision = svc.handle_message("عندي رشح خفيف من يومين", None)

    assert not state.risk.is_emergency
    assert state.risk.risk_level is RiskLevel.NONE
    assert decision.finished is False          # المقابلة تكمل طبيعياً


# ----------------------------------------------------------------------
# L0 — الفحص يعمل رغم انهيار الـLLM (الخاصية الحاسمة)
# ----------------------------------------------------------------------
def test_emergency_detected_even_when_llm_completely_fails(flags):
    """أهمّ اختبار في الملف: السلامة لا تتوقّف على نجاح استدعاء شبكي."""
    svc = service(Broken(), flags)
    state, decision = svc.handle_message(ACS_MESSAGE, None)   # لا يرفع خطأ

    assert state.risk.is_emergency
    assert state.risk.risk_level is RiskLevel.IMMEDIATE
    assert state.risk.degraded is True          # يُعلَن أنّ الحكم من L0 وحده
    assert decision.finished is True
    assert state.risk.primary_action_ar


def test_non_emergency_llm_failure_still_raises(flags):
    """بلا طارئ، يبقى فشل الـLLM خطأً عادياً (502) لا نجاحاً زائفاً."""
    svc = service(Broken(), flags)
    with pytest.raises(LLMProviderError):
        svc.handle_message("عندي رشح خفيف", None)


def test_degraded_flag_is_false_on_healthy_path(flags):
    svc = service(Scripted(payload()), flags)
    state, _ = svc.handle_message(ACS_MESSAGE, None)
    assert state.risk.degraded is False


# ----------------------------------------------------------------------
# L2 — الاستخراج المنظَّم يكشف ما تفوته المطابقة اللفظية
# ----------------------------------------------------------------------
def test_structured_extraction_can_raise_risk_alone(flags):
    """صياغة غير مغطّاة لفظياً، لكن الـLLM طبّعها إلى مصطلح معروف."""
    svc = service(Scripted(payload()), flags, symptom_batches=[[
        {"text": "ألم صدر", "negated": False, "confidence": 0.9, "evidence": None},
        {"text": "ضيق تنفس", "negated": False, "confidence": 0.9, "evidence": None},
    ]])
    state, _ = svc.handle_message("حاسس بشي غريب بقفصي الصدري ونفسي تقيل", None)

    assert state.risk.is_emergency


def test_inferred_symptoms_still_trigger_red_flags(flags):
    """قرار سلامة صريح: التصعيد أأمن من الإسقاط."""
    svc = service(Scripted(payload()), flags, symptom_batches=[[
        {"text": "ألم صدر", "negated": False, "confidence": 0.5,
         "evidence": "شاهد مختلق لا يرد في كلام المريض"},
        {"text": "ضيق تنفس", "negated": False, "confidence": 0.5, "evidence": None},
    ]])
    state, _ = svc.handle_message("ما بعرف شو فيني", None)

    from app.domain.clinical_record import FactSource
    assert all(s.source is FactSource.LLM_INFERRED for s in state.symptoms)
    assert state.risk.is_emergency        # وُسمت استنتاجاً لكنها صعّدت


def test_negated_symptoms_do_not_trigger(flags):
    svc = service(Scripted(payload(symptoms=[
        {"text": "ألم صدر", "negated": True, "confidence": 0.9, "evidence": None},
        {"text": "ضيق تنفس", "negated": True, "confidence": 0.9, "evidence": None},
    ])), flags)
    state, _ = svc.handle_message("ما عندي ألم صدر ولا ضيق تنفس", None)

    assert not state.risk.is_emergency


# ----------------------------------------------------------------------
# L3 — تصعيد فقط، عبر الأدوار
# ----------------------------------------------------------------------
def test_risk_persists_across_turns_and_never_downgrades(flags):
    """طارئ ظهر بدور سابق لا يزول لأنّ الدور التالي هادئ."""
    svc = service(Scripted(payload(), payload()), flags)

    state, _ = svc.handle_message(ACS_MESSAGE, None)
    assert state.risk.is_emergency

    state, _ = svc.handle_message("شكراً", state.session_id)
    assert state.risk.is_emergency        # لم يُخفَّض
    assert state.risk.risk_level is RiskLevel.IMMEDIATE


def test_pregnancy_bleeding_combines_across_turns(flags):
    svc = service(Scripted(payload(), payload()), flags)

    state, _ = svc.handle_message("أنا حامل بالشهر السابع", state_id := None)
    first_risk = state.risk.risk_level
    state, decision = svc.handle_message("صار عندي نزيف", state.session_id)

    assert first_risk is RiskLevel.NONE       # الحمل وحده ليس طارئاً
    assert state.risk.is_emergency            # التركيب عبر الأدوار كشفه
    assert decision.finished is True


# ----------------------------------------------------------------------
# التوافق الخلفي
# ----------------------------------------------------------------------
def test_service_without_engine_still_works(flags):
    """المحرّك اختياري بالتوقيع — الاختبارات القائمة لا تنكسر."""
    svc = ConversationService(
        provider=Scripted(payload()),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
    )
    state, decision = svc.handle_message(ACS_MESSAGE, None)

    assert state.risk.evaluated is False      # يُعلَن أنّ الفحص لم يجرِ
    assert not state.risk.is_emergency
    assert decision.finished is False


def test_api_shows_each_rule_once_but_audit_keeps_all_layers(flags):
    """المريض يرى تنبيهاً واحداً؛ التدقيق يحتفظ بأثر كل طبقة."""
    svc = service(Scripted(payload(symptoms=[
        {"text": "ألم صدر", "negated": False, "confidence": 0.9, "evidence": None},
        {"text": "ضيق تنفس", "negated": False, "confidence": 0.9, "evidence": None},
    ])), flags)
    state, _ = svc.handle_message(ACS_MESSAGE, None)

    assert len(state.risk.matches) > len(state.risk.unique_matches)
    assert len({m.rule_id for m in state.risk.unique_matches}) == \
        len(state.risk.unique_matches)
    layers = {m.layer for m in state.risk.matches}
    assert len(layers) == 2      # L0 و L2 كلاهما مسجَّل
