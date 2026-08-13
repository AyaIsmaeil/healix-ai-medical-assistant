"""اختبار تكامل C-1/C-4 (Phase 1) + C-1 الإغلاق الكامل (Phase 1.1):
نتيجة فحص الأعلام الحمراء يجب أن تصل فعلاً لمُصنِّف الاستعجال عبر
/api/assessment/run — سواء وصلت صريحة من المقابلة (``interview_risk``)، أو
غابت كلياً واحتاج البنّاء لتشغيل نفس ``RedFlagEngine`` الحقيقي مباشرة على
``raw_messages`` (BUG مُثبَت بـHEALIX_INTERVIEW_ASSESSMENT_AUDIT.md، والتحقّق
اللاحق أثبت أنّ الاعتماد على ``interview_risk`` وحدها غير مكتمل من طرف لطرف
لأنّ Laravel خارج هذا المستودع وقد لا يرسله).

يُنفَّذ المسار الحقيقي حرفياً كما يُركَّبه ``app/routes/assessment.py``:
AssessmentRequest (Pydantic حقيقي) → اشتقاق known_has_red_flag (نفس سطر
الراوت) → AssessmentFeatureBuilder.build() الحقيقي (مع RedFlagEngine حقيقي
محقون، كما في main.py) → RuleBasedUrgencyClassifier الحقيقي. لا FastAPI
TestClient (غير مُستخدَم بهذا المشروع أصلاً) ولا محاكاة لأي من الأصناف
الحقيقية — فقط مزوّد LLM وهمي بلا شبكة (نفس روح باقي اختبارات الوحدة).
"""

from app.domain.clinical_record import ClinicalRecord
from app.domain.conversation import Symptom
from app.domain.feature_extraction_rules import RuleBasedFeatureExtractor
from app.domain.ports import Completion
from app.domain.red_flag_engine import RedFlagEngine
from app.domain.rule_based_urgency_classifier import RuleBasedUrgencyClassifier
from app.domain.urgency import UrgencyLevel
from app.infrastructure.dictionary_loader import DictionaryLoader
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.assessment_extraction_builder import AssessmentExtractionPromptBuilder
from app.prompts.interview_builder import InterviewPromptBuilder
from app.schemas.assessment import AssessmentRequest, InterviewRiskIn
from app.schemas.symptom import SymptomOut
from app.services.assessment_feature_builder import AssessmentFeatureBuilder
from app.services.conversation_service import ConversationService
from app.services.llm_feature_extractor import LLMFeatureExtractor


class EmptyProvider:
    """لا يُرجع أي حقول — يحاكي عدم توفّر معلومة إضافية، بلا شبكة حقيقية."""

    name = "empty"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        return Completion("{}", model="empty")


def _real_red_flag_engine() -> RedFlagEngine:
    """محرّك مبنيّ من جدول الإنتاج نفسه — لا بيانات اختبار مصطنعة (C-1)."""
    return RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())


def _builder(red_flag_engine=None) -> AssessmentFeatureBuilder:
    return AssessmentFeatureBuilder(
        rule_extractor=RuleBasedFeatureExtractor(),
        llm_extractor=LLMFeatureExtractor(
            provider=EmptyProvider(),
            prompt_builder=AssessmentExtractionPromptBuilder(),
        ),
        red_flag_engine=red_flag_engine,
    )


def _run_real_assessment_path(request: AssessmentRequest, red_flag_engine=None) -> UrgencyLevel:
    """يُعيد إنتاج تسلسل app/routes/assessment.py حرفياً (بلا طبقة HTTP):
    اشتقاق known_has_red_flag من الطلب → البنّاء الحقيقي (مع/بلا محرّك أعلام
    حمراء حقيقي، كما يُحدَّد بكل اختبار) → المُصنِّف الحقيقي.
    """
    symptoms = [
        Symptom(text=s.text, negated=s.negated, confidence=s.confidence)
        for s in request.symptoms
    ]
    interview_record = (
        ClinicalRecord(**request.interview_record.model_dump())
        if request.interview_record is not None
        else None
    )
    # --- نفس سطر الراوت الحقيقي بالضبط (app/routes/assessment.py) ---
    known_has_red_flag = (
        request.interview_risk.emergency_detected
        if request.interview_risk is not None
        else None
    )

    features = _builder(red_flag_engine).build(
        request.session_id, request.raw_messages, symptoms,
        interview_record, known_has_red_flag,
    )
    result = RuleBasedUrgencyClassifier().classify(features)
    return result.level


# ----------------------------------------------------------------------
# السيناريو المرجعي (نفس تركيبة ACS من HEALIX_INTERVIEW_ASSESSMENT_AUDIT.md):
# المقابلة اكتشفت علماً أحمر فعلياً — يجب أن يصل هذا الحكم للتقييم النهائي.
# ----------------------------------------------------------------------
def test_interview_red_flag_reaches_final_urgency_as_emergency():
    """Test 1 — interview_risk.emergency_detected=True صراحةً → EMERGENCY."""
    request = AssessmentRequest(
        session_id="sid-acs-1",
        raw_messages=[
            "عندي ألم وضغط في الصدر من أسبوعين، يزيد بالمجهود ويخف بالراحة",
            "نعم، عندي ضيق نفس وتعرّق بارد معه",
        ],
        symptoms=[
            SymptomOut(text="ألم صدر", negated=False, confidence=0.95),
            SymptomOut(text="ضيق نفس", negated=False, confidence=0.9),
        ],
        interview_risk=InterviewRiskIn(emergency_detected=True, risk_level="immediate"),
    )

    # لا محرّك أعلام حمراء محقون هنا عمداً: الحكم الصريح له الأولوية دوماً،
    # فلا يجوز أن يعتمد نجاح هذا الاختبار على الاحتياطي إطلاقاً.
    assert _run_real_assessment_path(request, red_flag_engine=None) == UrgencyLevel.EMERGENCY


# ----------------------------------------------------------------------
# Test 2 (الأهمّ) — C-1 الإغلاق الكامل: interview_risk غائب كلياً، لكنّ
# raw_messages تُطلق فعلياً RedFlagEngine الحقيقي (لا True مكتوبة يدوياً).
# ----------------------------------------------------------------------
def test_fallback_derives_emergency_from_real_red_flag_engine_without_interview_risk():
    """يفشل هذا الاختبار قبل إغلاق C-1: بلا interview_risk كانت النتيجة
    تسقط دوماً إلى has_red_flag=False بصرف النظر عن محتوى raw_messages.
    الآن: AssessmentFeatureBuilder يُشغّل RedFlagEngine الحقيقي مباشرة على
    raw_messages نفسها التي تحمل نمط ACS كاملاً (ألم صدر + ضيق نفس + تعرّق
    بارد)، فيستقرّ الحكم على EMERGENCY من فحص حقيقي، لا قيمة مُلقَّنة."""
    request = AssessmentRequest(
        session_id="sid-acs-fallback-1",
        raw_messages=[
            "عندي ألم وضغط في الصدر من أسبوعين، يزيد بالمجهود ويخف بالراحة",
            "نعم، عندي ضيق نفس وتعرّق بارد معه",
        ],
        symptoms=[
            SymptomOut(text="ألم صدر", negated=False, confidence=0.95),
            SymptomOut(text="ضيق نفس", negated=False, confidence=0.9),
        ],
        # لا interview_risk إطلاقاً.
    )

    assert request.interview_risk is None
    result = _run_real_assessment_path(request, red_flag_engine=_real_red_flag_engine())
    assert result == UrgencyLevel.EMERGENCY


# ----------------------------------------------------------------------
# Test 3 — التوافق الخلفي: بلا علم أحمر حقيقي بالرسائل وبلا interview_risk،
# السلوك السابق (NON_URGENT) يبقى كما هو — لكن الآن عبر فحص حقيقي وجد
# لا شيء، لا سقوط تلقائي لقيمة ثابتة.
# ----------------------------------------------------------------------
def test_no_red_flag_in_messages_and_no_interview_risk_stays_non_urgent():
    request = AssessmentRequest(
        session_id="sid-legacy-1",
        raw_messages=["عندي صداع خفيف"],
        symptoms=[SymptomOut(text="صداع", negated=False, confidence=0.8)],
        # لا interview_risk إطلاقاً — الحقل اختياري، القيمة الافتراضية None.
    )

    assert request.interview_risk is None
    # المحرّك الحقيقي محقون (كما بالإنتاج) ويُشغَّل فعلياً — النتيجة NON_URGENT
    # لأنّه لم يجد أي نمط علم أحمر بالرسائل، لا لأنّه لم يُفحَص أصلاً.
    result = _run_real_assessment_path(request, red_flag_engine=_real_red_flag_engine())
    assert result == UrgencyLevel.NON_URGENT


def test_interview_risk_present_but_not_emergency_does_not_force_escalation():
    """وصول الحقل بحدّ ذاته لا يرفع الاستعجال — ``emergency_detected=False``
    مع رسائل بلا نمط طارئ يبقى NON_URGENT."""
    request = AssessmentRequest(
        session_id="sid-no-emergency-1",
        raw_messages=["عندي صداع خفيف"],
        symptoms=[SymptomOut(text="صداع", negated=False, confidence=0.8)],
        interview_risk=InterviewRiskIn(emergency_detected=False, risk_level="none"),
    )

    assert _run_real_assessment_path(request, red_flag_engine=_real_red_flag_engine()) == UrgencyLevel.NON_URGENT


def test_interview_risk_false_does_not_suppress_engine_emergency():
    """C-1 تصعيد فقط: ``emergency_detected=False`` من العميل لا يُخفّض طارئاً
    يكتشفه ``RedFlagEngine`` على ``raw_messages`` (BUG P0 من التدقيق)."""
    request = AssessmentRequest(
        session_id="sid-acs-client-false-1",
        raw_messages=[
            "عندي ألم وضغط في الصدر من أسبوعين، يزيد بالمجهود ويخف بالراحة",
            "نعم، عندي ضيق نفس وتعرّق بارد معه",
        ],
        symptoms=[
            SymptomOut(text="ألم صدر", negated=False, confidence=0.95),
            SymptomOut(text="ضيق نفس", negated=False, confidence=0.9),
        ],
        interview_risk=InterviewRiskIn(emergency_detected=False, risk_level="none"),
    )

    assert _run_real_assessment_path(request, red_flag_engine=_real_red_flag_engine()) == UrgencyLevel.EMERGENCY


# ----------------------------------------------------------------------
# Test 4 — إثبات أنّ الاحتياطي يستخدم نفس قواعد /api/interview/turn حرفياً:
# نفس نسخة RedFlagEngine الواحدة تُستهلَك أولاً بمقابلة حقيقية عبر
# ConversationService (فتكتشف الطوارئ من تلقاء نفسها)، ثم يُعاد استخدامها
# مباشرة باحتياطي التقييم على نفس الرسائل الخام — بلا أي True مكتوبة يدوياً
# بأي من الخطوتين.
# ----------------------------------------------------------------------
class _ScriptedInterviewProvider:
    """لا يُستدعى فعلياً بهذا الاختبار (L0 يُنهي الدور قبل استهلاكه) — موجود
    فقط لأنّ ConversationService يتطلّب مزوّداً، بنفس نمط بقيّة اختبارات
    الوحدة (test_interview_safety_layer.py)."""

    name = "scripted"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        return Completion(
            '{"chief_complaint": null, "symptoms": [], "severity": null, '
            '"duration": null, "body_location": null, "medications": [], '
            '"allergies": [], "chronic_conditions": [], "family_history": [], '
            '"missing_fields": [], "finished": false, "next_slot": "onset", '
            '"question": "?"}',
            model="scripted",
        )


ACS_MESSAGE = "عندي ألم ضاغط بصدري وضيق نفس وعرق بارد"


def test_fallback_uses_the_exact_same_engine_instance_as_the_interview():
    shared_engine = _real_red_flag_engine()

    # (1) /api/interview/turn الحقيقي (عبر ConversationService) يكتشف
    #     الطوارئ من تلقاء نفسه — بلا أي تلقين.
    svc = ConversationService(
        provider=_ScriptedInterviewProvider(),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=8,
        red_flag_engine=shared_engine,
    )
    state, _decision = svc.handle_message(ACS_MESSAGE, None)
    assert state.risk.is_emergency, "المقابلة نفسها لم تكتشف الطوارئ — الرسالة المرجعية خاطئة لا الكود"

    # (2) نفس نسخة RedFlagEngine ونفس الرسالة تُمرَّران لاحتياطي التقييم —
    #     بلا interview_risk، وبلا استنساخ حكم True يدوياً.
    request = AssessmentRequest(
        session_id=state.session_id,
        raw_messages=list(state.raw_messages),
        symptoms=[SymptomOut(text="ألم صدر", negated=False, confidence=0.9)],
    )
    assert request.interview_risk is None
    result = _run_real_assessment_path(request, red_flag_engine=shared_engine)
    assert result == UrgencyLevel.EMERGENCY
