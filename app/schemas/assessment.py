"""
Healix - Assessment Schemas
نماذج الطلب والاستجابة لمحرك التقييم (طبقة الـAPI فقط — Pydantic).

``AssessmentRequest`` يعيد استخدام ``SymptomOut`` من ``schemas.symptom`` كمدخل
للأعراض الجاهزة من المقابلة (لا تكرار للشكل). نماذج الاستجابة مرآة لـ
``domain.assessment`` — التحويل عبر ``dataclasses.asdict`` + ``model_validate``
بالراوت، لا حقول مُعاد كتابتها يدوياً.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.symptom import SymptomOut


class InterviewRiskIn(BaseModel):
    """نتيجة فحص الأعلام الحمراء من ``RedFlagEngine`` كما أنتجتها المقابلة.

    نسخ حرفي من حقلَي ``emergency_detected``/``risk_level`` الموجودَين أصلاً
    باستجابة ``POST /api/interview/turn`` — **لا حساب جديد هنا ولا محرّك
    أعلام حمراء ثانٍ**؛ هذا الحقل ينقل حكماً حتمياً حُسِب سلفاً بمحرّك واحد
    (نفس مبدأ ``InterviewRecordIn``). اختياري بالكامل؛ إرساله يُغني عن إعادة
    فحص ``raw_messages`` (أسرع). غيابه **لا يعني** ``has_red_flag=False``
    بعد الآن (C-1، Phase 1.1) — ``AssessmentFeatureBuilder`` يُشغّل حينها
    نفس ``RedFlagEngine`` الحقيقي مباشرة على ``raw_messages`` بدل الاعتماد
    حصراً على هذا الحقل، فتبقى الصحّة مستقلّة عن أي وسيط خارجي (Laravel)
    يُحدَّث لإرساله أم لا.
    """

    emergency_detected: bool = False
    risk_level: str = "none"


class InterviewRecordIn(BaseModel):
    """السجل الطبي المنظَّم كما أنتجه وكيل المقابلة السريرية.

    اختياري بالكامل وكل حقوله اختيارية: الطلبات القديمة التي لا تُرسله تعمل
    تماماً كما كانت (سلوك مطابق للسابق). عند إرساله يصبح مصدر الحقيقة الأعلى
    أولوية، فيتخطّى محرك التقييم استخراج ما ورد فيه بدل إعادة استخراجه بالـLLM.
    يُنسخ حرفياً من استجابة ``POST /api/interview/turn``.
    """

    chief_complaint: Optional[str] = None
    severity: Optional[str] = None
    duration: Optional[str] = None
    body_location: Optional[str] = None
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)


class AssessmentRequest(BaseModel):
    """لقطة كاملة من المحادثة المُنتهية — الخدمة مستقلة (stateless)، لا تُقرأ
    من جلسة داخلية بالذاكرة."""

    session_id: str = Field(..., min_length=1)
    raw_messages: List[str] = Field(
        ...,
        min_length=1,
        description="كامل رسائل المريض الخام بترتيب ورودها.",
    )
    symptoms: List[SymptomOut] = Field(
        default_factory=list,
        description="الأعراض المُستخرَجة مسبقاً أثناء المقابلة — تُعاد "
        "استخدامها هنا، لا تُعاد حسبتها.",
    )
    interview_record: Optional[InterviewRecordIn] = Field(
        default=None,
        description="السجل المنظَّم من المقابلة. إرساله يمنع إعادة استخراج ما "
        "ورد فيه (أسرع وأرخص)، ويملأ التاريخ المرضي. حذفه لا يكسر شيئاً.",
    )
    interview_risk: Optional[InterviewRiskIn] = Field(
        default=None,
        description="نتيجة فحص الأعلام الحمراء من /api/interview/turn إن "
        "وُجدت (نسخ حرفي من emergency_detected/risk_level) — يُغني عن إعادة "
        "الفحص. حذفها لا يُسقِط الاستعجال إلى قيمة ثابتة (C-1، Phase 1.1): "
        "الخادم يُعيد فحص raw_messages بنفس محرّك الأعلام الحمراء الحقيقي "
        "تلقائياً.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "b1e7f9b0-...",
                "raw_messages": ["أعاني من صداع شديد وحرارة منذ ثلاثة أيام"],
                "symptoms": [
                    {"text": "صداع شديد", "negated": False, "confidence": 0.97},
                    {"text": "حرارة", "negated": False, "confidence": 0.93},
                ],
            }
        }
    }


class DemographicsOut(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    pregnancy_possible: Optional[bool] = None


class SymptomDescriptorsOut(BaseModel):
    onset: Optional[str] = None
    onset_days_ago: Optional[int] = None
    duration: Optional[str] = None
    severity_0_10: Optional[int] = None
    progression: Optional[str] = None
    location: Optional[str] = None
    quality: Optional[str] = None
    aggravating: List[str] = Field(default_factory=list)
    relieving: List[str] = Field(default_factory=list)


class SymptomFeatureOut(BaseModel):
    """عرَض في مخرجات التقييم.

    ``marbert_confidence`` اسم قديم من زمن MARBERT بقي **في الاستجابة فقط**
    كمرآة لـ``extraction_confidence`` — حذفه كان سيكسر مستهلكاً حالياً
    (تكامل Laravel). الاسم المعياري الجديد هو ``extraction_confidence``؛
    القديم مهجور (deprecated) ويُحذف في نسخة عقد لاحقة مُعلَنة.
    """

    name: str
    negated: bool
    extraction_confidence: float
    marbert_confidence: float = Field(
        default=0.0, deprecated=True,
        description="مهجور — استخدم extraction_confidence. مرآة لنفس القيمة.",
    )
    is_primary: bool = False

    @model_validator(mode="before")
    @classmethod
    def _mirror_legacy_confidence(cls, data):
        """يملأ الاسم القديم من الجديد تلقائياً.

        المجال لم يعد يعرف ``marbert_confidence`` إطلاقاً؛ المرآة تُشتقّ هنا
        عند التسلسل فقط، فيبقى مصدر الحقيقة واحداً ولا ينحرف الاسمان.
        """
        if isinstance(data, dict) and "extraction_confidence" in data:
            data = {**data, "marbert_confidence": data["extraction_confidence"]}
        return data
    descriptors: SymptomDescriptorsOut = Field(default_factory=SymptomDescriptorsOut)
    associated_symptom_flags: Dict[str, bool] = Field(default_factory=dict)


class MedicalHistoryOut(BaseModel):
    chronic_diseases: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)


class LifestyleOut(BaseModel):
    smoking: Optional[bool] = None
    alcohol: Optional[bool] = None
    occupation: Optional[str] = None


class DerivedFeaturesOut(BaseModel):
    symptom_count: int = 0
    positive_negative_ratio: Optional[float] = None
    primary_symptom_severity: Optional[int] = None
    has_red_flag: bool = False
    interview_completeness_ratio: Optional[float] = None


class ValidationReportOut(BaseModel):
    """فارغ بمرحلة ٣.١ — يُملأ فعلياً بـFeatureValidator بمرحلة ٣.٢."""

    corrected_fields: List[str] = Field(default_factory=list)
    rejected_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validity_score: Optional[float] = None


class ClinicalFeatureSetOut(BaseModel):
    session_id: str
    demographics: DemographicsOut = Field(default_factory=DemographicsOut)
    symptoms: List[SymptomFeatureOut] = Field(default_factory=list)
    negated_symptoms: List[str] = Field(default_factory=list)
    temperature_c: Optional[float] = None
    medical_history: MedicalHistoryOut = Field(default_factory=MedicalHistoryOut)
    lifestyle: LifestyleOut = Field(default_factory=LifestyleOut)
    derived: DerivedFeaturesOut = Field(default_factory=DerivedFeaturesOut)
    validation: ValidationReportOut = Field(default_factory=ValidationReportOut)
    unresolved_fields: List[str] = Field(default_factory=list)


class DiseasePredictionOut(BaseModel):
    """احتمال مرض واحد (Phase 3.4) — مرآة لـ``domain.prediction.DiseasePrediction``."""

    disease: str
    score: float
    explanation: str


class DiseasePredictionResultOut(BaseModel):
    """ناتج التنبؤ الكامل — مرآة لـ``domain.prediction.DiseasePredictionResult``."""

    predictions: List[DiseasePredictionOut] = Field(default_factory=list)
    predictor_version: str = ""


class UrgencyAssessmentOut(BaseModel):
    """نتيجة تقييم الاستعجال (Phase 3.5) — مرآة لـ``domain.urgency.UrgencyAssessment``.
    ``level`` نصّي (قيمة UrgencyLevel) — EMERGENCY | URGENT | SEMI_URGENT | NON_URGENT."""

    level: str
    score: float
    explanation: str


class SpecialtyRecommendationOut(BaseModel):
    """توصية التخصّص الطبي (Phase 3.6) — مرآة لـ``domain.specialty.SpecialtyRecommendation``."""

    specialty: str
    confidence: float
    explanation: str


class ConfidenceAssessmentOut(BaseModel):
    """تقدير موثوقية التقييم الكامل (Phase 3.7) — مرآة لـ
    ``domain.confidence.ConfidenceAssessment``. ``requires_human_review`` علم
    صريح لطبقة التطبيق بأنّ الحالة تحتاج مراجعة بشرية."""

    overall_confidence: float
    requires_human_review: bool
    explanation: str


class AssessmentExplanationOut(BaseModel):
    """التفسير النهائي للتقييم بالعربية (Phase 3.8) — مرآة لـ
    ``domain.explanation.AssessmentExplanation``. شرح لما حُسِب سلفاً فقط —
    لا تشخيص ولا تعديل لأي مخرَج سابق."""

    summary: str
    medical_reasoning: str
    recommendation: str
    disclaimer: str


class RagSourceOut(BaseModel):
    """مصدر PubMed مسترجَع (RAG) — للشفافية والاستشهاد."""

    doc_id: str
    pmid: str
    disease_name: str = ""
    medical_specialty: str = ""
    triage_level: str = ""
    snippet: str = ""
    pubmed_url: str = ""
    relevance_score: Optional[float] = None


class AssessmentResponse(BaseModel):
    """استجابة محرّك التقييم الهجين: قواعد + ML + RAG (اختياري)."""

    success: bool = True
    status: str = "features_extracted"
    hybrid_mode: str = Field(
        default="rules_ml",
        description="rules_ml | rules_ml_rag — يبيّن إن RAG شارك في الشرح",
    )
    features: ClinicalFeatureSetOut
    predictions: DiseasePredictionResultOut = Field(default_factory=DiseasePredictionResultOut)
    urgency: UrgencyAssessmentOut
    specialty: SpecialtyRecommendationOut
    confidence: ConfidenceAssessmentOut
    explanation: AssessmentExplanationOut
    rag_sources: List[RagSourceOut] = Field(default_factory=list)
