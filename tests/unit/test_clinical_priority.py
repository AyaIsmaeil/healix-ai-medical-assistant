"""
اختبارات محرّك الأولوية السريرية (P1) — **بصفر استدعاء LLM وصفر شبكة**.

الفكرة المُختبَرة: الشكوى الرئيسية قرار **المحرّك** لا النموذج، ويُعاد
حسابه بعد كل رسالة — فيتحوّل محور المقابلة فوراً عند ظهور عرَض أعلى حدّة.
"""

import json

import pytest

from app.domain import clinical
from app.domain.clinical_priority import ClinicalPriorityEngine
from app.domain.conversation import Symptom
from app.domain.ports import Completion
from app.domain.red_flag_engine import RedFlagEngine
from app.domain.red_flags import RedFlagAssessment, RiskLevel
from app.infrastructure.dictionary_loader import DictionaryLoader
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService


@pytest.fixture(scope="module")
def engine() -> ClinicalPriorityEngine:
    """المحرّك مبنيّ من جدول الإنتاج نفسه — لا بيانات اختبار مصطنعة."""
    return ClinicalPriorityEngine.from_dict(DictionaryLoader.load_clinical_priority())


@pytest.fixture(scope="module")
def flags() -> RedFlagEngine:
    return RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())


def s(text, negated=False, turn=0):
    return Symptom(text, negated, 0.9, turn_number=turn)


# ----------------------------------------------------------------------
# جوهر P1 — إعادة الترتيب لا ترتيب الظهور
# ----------------------------------------------------------------------
def test_higher_acuity_symptom_wins_regardless_of_order(engine):
    """السيناريو المرجعي: صداع أولاً ثم ألم صدر — الأولوية تتحوّل."""
    assert engine.primary([s("صداع", turn=1), s("ألم صدر", turn=3)]) == "ألم صدر"


def test_order_of_mention_does_not_change_the_winner(engine):
    """الترتيب العكسي يعطي النتيجة نفسها — لا أثر لأوّل عرَض."""
    assert engine.primary([s("ألم صدر", turn=1), s("صداع", turn=3)]) == "ألم صدر"


def test_first_symptom_is_never_privileged(engine):
    """أوّل عرَض شائع لا يهزم عرَضاً حادّاً لاحقاً مهما تراكمت النقاط."""
    ranked = engine.rank([s("رشح", turn=1), s("ضيق تنفس", turn=2)])
    assert ranked[0].symptom_text == "ضيق تنفس"
    assert ranked[0].score > ranked[1].score


def test_equal_acuity_falls_back_to_recency_then_text(engine):
    """التعادل يُحسم حتمياً — لا عشوائية."""
    first = engine.rank([s("صداع", turn=1), s("صداع", turn=1)])
    second = engine.rank([s("صداع", turn=1), s("صداع", turn=1)])
    assert [p.symptom_text for p in first] == [p.symptom_text for p in second]


def test_negated_symptoms_are_excluded(engine):
    """عرَض نفاه المريض لا يصلح محوراً للتاريخ المرضي."""
    assert engine.primary([s("ألم صدر", negated=True), s("صداع")]) == "صداع"


def test_no_positive_symptoms_yields_none(engine):
    assert engine.primary([]) is None
    assert engine.primary([s("صداع", negated=True)]) is None


def test_unknown_symptom_gets_default_acuity_not_dropped(engine):
    """عرَض غير مُدرَج بالجدول يبقى مرشّحاً بدرجة افتراضية."""
    ranked = engine.rank([s("عرَض غير معروف إطلاقاً")])
    assert len(ranked) == 1
    assert ranked[0].concept is None
    assert ranked[0].acuity == 30


# ----------------------------------------------------------------------
# مكوّنات الدرجة
# ----------------------------------------------------------------------
def test_emergency_dominates_ranking(engine):
    """علم أحمر مُطلَق يرفع العرَض الحادّ فوق أي تراكم نقاط."""
    calm = engine.rank([s("ألم صدر"), s("صداع")])
    urgent = engine.rank(
        [s("ألم صدر"), s("صداع")],
        risk=RedFlagAssessment(risk_level=RiskLevel.IMMEDIATE),
    )
    assert urgent[0].symptom_text == "ألم صدر"
    assert urgent[0].score > calm[0].score
    assert urgent[0].red_flag_boost == 100.0


def test_emergency_boost_does_not_flatten_low_acuity_symptoms(engine):
    """التعزيز للأعراض الحادّة فقط — وإلا تساوى كل شيء وفقد الترتيب معناه."""
    ranked = engine.rank(
        [s("ألم صدر"), s("رشح")],
        risk=RedFlagAssessment(risk_level=RiskLevel.IMMEDIATE),
    )
    by_text = {p.symptom_text: p for p in ranked}
    assert by_text["ألم صدر"].red_flag_boost == 100.0
    assert by_text["رشح"].red_flag_boost == 0.0


def test_patient_stated_severity_raises_score(engine):
    plain = engine.rank([s("صداع")], ["عندي صداع"])[0]
    severe = engine.rank([s("صداع")], ["عندي صداع شديد جداً ما بحتمل"])[0]
    assert severe.score > plain.score
    assert severe.severity_boost > 0


def test_patient_emphasis_raises_score(engine):
    plain = engine.rank([s("صداع")], ["عندي صداع"])[0]
    emphatic = engine.rank([s("صداع")], ["عندي صداع وأنا خايف مش طبيعي"])[0]
    assert emphatic.emphasis_boost > 0
    assert emphatic.score > plain.score


def test_newest_symptom_gets_recency_boost(engine):
    ranked = engine.rank([s("صداع", turn=1), s("صداع خفيف", turn=5)])
    by_text = {p.symptom_text: p for p in ranked}
    assert by_text["صداع خفيف"].recency_boost > 0
    assert by_text["صداع"].recency_boost == 0


def test_severity_never_outweighs_acuity_gap(engine):
    """حدّة عالية تغلب شدّة معلنة على عرَض خفيف — فواصل الأوزان مقصودة."""
    ranked = engine.rank(
        [s("رشح", turn=1), s("ألم صدر", turn=1)],
        ["عندي رشح شديد جداً ما بحتمل"],
    )
    assert ranked[0].symptom_text == "ألم صدر"


def test_rationale_explains_the_score(engine):
    p = engine.rank([s("ألم صدر")], ["ألم شديد جداً"],
                    risk=RedFlagAssessment(risk_level=RiskLevel.IMMEDIATE))[0]
    assert "acuity=" in p.rationale
    assert "red_flag=" in p.rationale
    assert "severity=" in p.rationale


# ----------------------------------------------------------------------
# التطبيع والمطابقة
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "ألم صدر", "وجع بصدري", "ألم ضاغط بصدري", "ثقل بصدري",
])
def test_chest_pain_variants_all_match_the_concept(engine, text):
    ranked = engine.rank([s(text)])
    assert ranked[0].concept == "chest_pain"
    assert ranked[0].acuity == 95


def test_longest_term_wins_on_overlap(engine):
    """مصطلح أدقّ لا تبتلعه كلمة عامّة."""
    assert engine.rank([s("ألم بصدري")])[0].concept == "chest_pain"


def test_ranking_is_deterministic(engine):
    """نفس المدخلات ⇒ نفس الترتيب دائماً (لا عشوائية، لا حالة داخلية)."""
    syms = [s("صداع", turn=1), s("حرارة", turn=2), s("ألم صدر", turn=3)]
    runs = [[p.symptom_text for p in engine.rank(syms, ["نص"])] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


# ----------------------------------------------------------------------
# قائمة الأسئلة تتبع قرار المحرّك
# ----------------------------------------------------------------------
def test_checklist_reanchors_to_engine_primary(engine):
    syms = [s("صداع", turn=1), s("ألم صدر", turn=3)]
    primary = engine.primary(syms)

    targets = [t for t, _ in clinical.build_checklist(syms, "", primary=primary)]
    assert targets[0] == "onset@ألم صدر"
    assert not targets[0].endswith("@صداع")


def test_checklist_without_primary_keeps_legacy_behaviour():
    """التدهور الآمن: بلا محرّك يبقى السلوك القديم فلا ينكسر مستهلك."""
    syms = [s("صداع", turn=1), s("ألم صدر", turn=3)]
    targets = [t for t, _ in clinical.build_checklist(syms, "")]
    assert targets[0] == "onset@صداع"


def test_stale_primary_is_rejected():
    """شكوى لم تعد ضمن الأعراض لا تُبنى حولها أسئلة."""
    syms = [s("صداع")]
    targets = [t for t, _ in clinical.build_checklist(syms, "", primary="ألم صدر")]
    assert targets[0] == "onset@صداع"


def test_next_missing_follows_primary(engine):
    syms = [s("صداع", turn=1), s("ألم صدر", turn=3)]
    target, question = clinical.next_missing(
        syms, "", asked=[], primary=engine.primary(syms))
    assert target == "onset@ألم صدر"
    assert "ألم صدر" in question


# ----------------------------------------------------------------------
# التكامل داخل ConversationService
# ----------------------------------------------------------------------
def sym_payload(text, evidence):
    return {"text": text, "negated": False, "confidence": 0.9, "evidence": evidence}


def turn_payload(symptoms, **over):
    base = {
        "chief_complaint": None, "symptoms": symptoms, "severity": None,
        "duration": None, "body_location": None, "medications": [],
        "allergies": [], "chronic_conditions": [], "family_history": [],
        "missing_fields": [], "finished": False,
        "next_slot": "onset", "question": "س؟",
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


def build(provider, engine, flags):
    return ConversationService(
        provider=provider, prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(), red_flag_engine=flags,
        priority_engine=engine, max_questions=40,
    )


def test_service_reanchors_mid_interview(engine, flags):
    """السيناريو المرجعي كاملاً عبر الخدمة."""
    svc = build(Scripted(
        turn_payload([sym_payload("صداع", "عندي صداع")]),
        turn_payload([sym_payload("صداع", "عندي صداع"),
                      sym_payload("ألم صدر", "ألم ضاغط بصدري")],
                     next_slot="severity", question="س٢؟"),
    ), engine, flags)

    state, _ = svc.handle_message("عندي صداع", None)
    assert state.primary_complaint == "صداع"

    state, _ = svc.handle_message("وصار عندي ألم ضاغط بصدري", state.session_id)
    assert state.primary_complaint == "ألم صدر"


def test_service_recomputes_every_turn_not_once(engine, flags):
    """القرار يُعاد حسابه لا يُثبَّت — يعود للصداع لو زال ألم الصدر مستحيل،
    لكن ترتيب الدرجات يجب أن يُحدَّث في كل دور."""
    svc = build(Scripted(
        turn_payload([sym_payload("صداع", "عندي صداع")]),
        turn_payload([sym_payload("صداع", "عندي صداع"),
                      sym_payload("حرارة", "وحرارة")],
                     next_slot="severity", question="س؟"),
    ), engine, flags)

    state, _ = svc.handle_message("عندي صداع", None)
    first = list(state.symptom_priorities)
    state, _ = svc.handle_message("وحرارة", state.session_id)

    assert len(state.symptom_priorities) == 2 > len(first)
    assert state.primary_complaint == "حرارة"      # حدّة أعلى من الصداع


def test_service_without_priority_engine_degrades_safely(flags):
    """بلا محرّك: لا انهيار، ولا ادّعاء بوجود قرار."""
    svc = ConversationService(
        provider=Scripted(turn_payload([sym_payload("صداع", "عندي صداع")])),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(), red_flag_engine=flags,
    )
    state, _ = svc.handle_message("عندي صداع", None)

    assert state.primary_complaint is None
    assert state.symptom_priorities == []


def test_llm_never_decides_the_primary_complaint(engine, flags):
    """النموذج يزعم شكوى مخالفة — المحرّك لا يعبأ بها."""
    svc = build(Scripted(turn_payload(
        [sym_payload("صداع", "عندي صداع"),
         sym_payload("ألم صدر", "ألم ضاغط بصدري")],
        chief_complaint="صداع",          # ادّعاء النموذج
    )), engine, flags)

    state, _ = svc.handle_message("عندي صداع وألم ضاغط بصدري", None)

    assert state.record.chief_complaint == "صداع"     # ما استخرجه النموذج
    assert state.primary_complaint == "ألم صدر"       # ما قرّره المحرّك


def test_prompt_exposes_engine_primary_to_the_model(engine, flags):
    """التعليمات تُبلّغ النموذج بقرار المحرّك كتأريض لا كاقتراح."""
    svc = build(Scripted(turn_payload(
        [sym_payload("صداع", "عندي صداع"),
         sym_payload("ألم صدر", "ألم ضاغط بصدري")])), engine, flags)
    state, _ = svc.handle_message("عندي صداع وألم ضاغط بصدري", None)

    prompt = InterviewPromptBuilder().turn_prompt(state)
    assert '"ألم صدر"' in prompt
    assert "PRIMARY_COMPLAINT" in prompt
    # والاقتراحات صارت حول الشكوى الجديدة.
    assert "@ألم صدر" in prompt


# ----------------------------------------------------------------------
# سلطة المحرّك على استهداف السؤال
# ----------------------------------------------------------------------
def test_question_targeting_another_symptom_is_redirected(engine, flags):
    """المحرّك حوّل المحور، والنموذج ظلّ يسأل عن العرَض القديم — يُعاد توجيهه."""
    svc = build(Scripted(turn_payload(
        [sym_payload("صداع", "عندي صداع"),
         sym_payload("ألم صدر", "ألم ضاغط بصدري")],
        next_slot="severity", question="كم شدّة الصداع بالنسبة لك؟",
    )), engine, flags)

    state, decision = svc.handle_message("عندي صداع وألم ضاغط بصدري", None)

    assert state.primary_complaint == "ألم صدر"
    assert "ألم صدر" in decision.next_slot
    assert "صداع" not in decision.question


def test_question_already_targeting_primary_is_left_alone(engine, flags):
    """لا تدخّل بلا داعٍ: سؤال يستهدف الشكوى الرئيسية يمرّ كما هو."""
    original = "هل ألم صدر يزداد مع المجهود؟"
    svc = build(Scripted(turn_payload(
        [sym_payload("صداع", "عندي صداع"),
         sym_payload("ألم صدر", "ألم ضاغط بصدري")],
        next_slot="exertion", question=original,
    )), engine, flags)

    _, decision = svc.handle_message("عندي صداع وألم ضاغط بصدري", None)
    assert decision.question == original


def test_generic_question_is_not_redirected(engine, flags):
    """سؤال عام لا يذكر أي عرَض يُترك للنموذج — صياغته أطبع من قالب."""
    original = "منذ متى بدأ ذلك بالضبط؟"
    svc = build(Scripted(turn_payload(
        [sym_payload("صداع", "عندي صداع"),
         sym_payload("ألم صدر", "ألم ضاغط بصدري")],
        next_slot="onset", question=original,
    )), engine, flags)

    _, decision = svc.handle_message("عندي صداع وألم ضاغط بصدري", None)
    assert decision.question == original


def test_single_symptom_interview_is_never_redirected(engine, flags):
    """بعرَض واحد لا يوجد انحراف ممكن — لا تدخّل."""
    original = "كم شدّة الصداع؟"
    svc = build(Scripted(turn_payload(
        [sym_payload("صداع", "عندي صداع")],
        next_slot="severity", question=original,
    )), engine, flags)

    _, decision = svc.handle_message("عندي صداع", None)
    assert decision.question == original
