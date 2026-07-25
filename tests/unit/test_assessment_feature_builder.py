"""اختبارات وحدة لـAssessmentFeatureBuilder — الدمج، إعادة استخدام أعراض المقابلة،
والحقول المحسوبة. مزوّد LLM وهمي بلا شبكة (نفس روح باقي اختبارات الوحدة)."""

from app.domain.clinical_record import ClinicalRecord
from app.domain.conversation import Symptom
from app.domain.feature_extraction_rules import RuleBasedFeatureExtractor
from app.domain.ports import Completion
from app.prompts.assessment_extraction_builder import AssessmentExtractionPromptBuilder
from app.services.assessment_feature_builder import AssessmentFeatureBuilder
from app.services.llm_feature_extractor import LLMFeatureExtractor


class EmptyProvider:
    """لا يُرجع أي حقول (يحاكي فشلاً ناعماً أو عدم توفّر معلومة)."""

    name = "empty"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        return Completion("{}", model="empty")


def _builder(provider=None):
    return AssessmentFeatureBuilder(
        rule_extractor=RuleBasedFeatureExtractor(),
        llm_extractor=LLMFeatureExtractor(
            provider=provider or EmptyProvider(),
            prompt_builder=AssessmentExtractionPromptBuilder(),
        ),
    )


def test_reuses_interview_symptoms_without_recomputation():
    symptoms = [Symptom("صداع شديد", False, 0.97), Symptom("حرارة", False, 0.9)]
    features = _builder().build("sid-1", ["عندي صداع شديد وحرارة"], symptoms)

    names = {s.name for s in features.symptoms}
    assert names == {"صداع شديد", "حرارة"}
    assert all(not s.negated for s in features.symptoms)


def test_negated_symptom_recorded_separately():
    symptoms = [Symptom("صداع", False, 0.9), Symptom("ضيق تنفس", True, 0.8)]
    features = _builder().build("sid-2", ["عندي صداع بلا ضيق تنفس"], symptoms)

    assert features.negated_symptoms == ["ضيق تنفس"]
    assert features.derived.symptom_count == 1  # المُثبَت فقط


def test_first_non_negated_symptom_is_primary():
    symptoms = [Symptom("صداع", False, 0.9), Symptom("دوخة", False, 0.8)]
    features = _builder().build("sid-3", ["عندي صداع ودوخة"], symptoms)

    primary = [s for s in features.symptoms if s.is_primary]
    assert len(primary) == 1
    assert primary[0].name == "صداع"


def test_rule_based_fields_merged_into_demographics_and_primary_descriptors():
    symptoms = [Symptom("صداع", False, 0.9)]
    features = _builder().build(
        "sid-4",
        ["عمري 30 سنة، أنا رجل، عندي صداع منذ يومين شدته 8 من 10"],
        symptoms,
    )

    assert features.demographics.age == 30
    assert features.demographics.gender == "male"
    primary = next(s for s in features.symptoms if s.is_primary)
    assert primary.descriptors.severity_0_10 == 8
    assert primary.descriptors.onset_days_ago == 2


def test_unresolved_fields_reported_when_nothing_found():
    symptoms = [Symptom("صداع", False, 0.9)]
    features = _builder().build("sid-5", ["عندي صداع"], symptoms)

    assert set(features.unresolved_fields) == {
        "age", "gender", "smoking", "temperature", "duration", "severity",
    }


def test_llm_fills_field_rule_extractor_missed():
    class AgeProvider:
        name = "age-provider"

        def generate(self, system_prompt: str, user_prompt: str) -> Completion:
            import json
            return Completion(json.dumps({"age": 45}), model="age-provider")

    symptoms = [Symptom("صداع", False, 0.9)]
    features = _builder(AgeProvider()).build("sid-6", ["عندي صداع"], symptoms)

    assert features.demographics.age == 45
    assert "age" not in features.unresolved_fields


def test_validation_report_is_empty_in_phase_3_1():
    features = _builder().build("sid-7", ["عندي صداع"], [Symptom("صداع", False, 0.9)])
    assert features.validation.corrected_fields == []
    assert features.validation.rejected_fields == []
    assert features.validation.warnings == []
    assert features.validation.validity_score is None


def test_no_symptoms_produces_empty_symptom_list_not_error():
    features = _builder().build("sid-8", ["مرحبا"], [])
    assert features.symptoms == []
    assert features.derived.symptom_count == 0
    assert features.derived.positive_negative_ratio is None


# ----------------------------------------------------------------------
# دمج سجل المقابلة (المصدر الوحيد للاستخراج) — إلغاء الازدواج
# ----------------------------------------------------------------------
class CountingProvider:
    """يعدّ استدعاءات الـLLM لإثبات تخطّيها عند اكتفاء سجل المقابلة."""

    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        self.calls += 1
        return Completion("{}", model="counting")


def _record(**kwargs) -> ClinicalRecord:
    return ClinicalRecord(**kwargs)


def test_interview_record_fills_medical_history():
    """التاريخ المرضي كان فجوة موثَّقة — سجل المقابلة يغلقها بلا مستخرج جديد."""
    record = _record(
        medications=["بنادول"],
        allergies=["بنسلين"],
        chronic_conditions=["سكري"],
        family_history=["ضغط"],
    )
    features = _builder().build("sid-h", ["نص"], [Symptom("صداع", False, 0.9)], record)

    assert features.medical_history.medications == ["بنادول"]
    assert features.medical_history.allergies == ["بنسلين"]
    assert features.medical_history.chronic_diseases == ["سكري"]
    assert features.medical_history.family_history == ["ضغط"]


def test_interview_record_severity_and_duration_take_priority():
    record = _record(severity="الشدة 8 من 10", duration="ثلاثة أيام")
    features = _builder().build("sid-s", ["نص"], [Symptom("صداع", False, 0.9)], record)

    primary = features.symptoms[0]
    assert primary.descriptors.severity_0_10 == 8
    # البنّاء يُسند نصّ المدّة إلى onset (اصطلاح قائم سابقاً، غير مُعدَّل هنا).
    assert primary.descriptors.onset == "ثلاثة أيام"
    assert "duration" not in features.unresolved_fields
    assert "severity" not in features.unresolved_fields


def test_unparsable_severity_is_not_guessed():
    """نصّ شدّة بلا رقم لا يُخمَّن — يبقى ناقصاً بشفافية."""
    record = _record(severity="شديد جداً")
    features = _builder().build("sid-u", ["نص"], [Symptom("صداع", False, 0.9)], record)

    assert features.symptoms[0].descriptors.severity_0_10 is None
    assert "severity" in features.unresolved_fields


def test_no_interview_record_keeps_previous_behaviour():
    """الطلبات القديمة (بلا سجل) تعمل تماماً كما كانت."""
    features = _builder().build("sid-o", ["نص"], [Symptom("صداع", False, 0.9)])

    assert features.medical_history.medications == []
    assert features.symptoms[0].name == "صداع"
