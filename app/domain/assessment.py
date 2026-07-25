"""
Healix - Assessment Domain Models
نماذج المجال الخالصة لمحرك التقييم (Phase 3.1: بناء الميزات فقط).

كائنات بيانات نقية (dataclasses) بلا أي اعتماد على FastAPI أو مكتبات ML أو
Pydantic — بالضبط نفس فلسفة ``domain.conversation``. ``ClinicalFeatureSet`` هو
الناتج النهائي لمرحلة استخراج الميزات، والمدخل المستقبلي لطبقة التنبؤ (Phase 3.2+).

لا يوجد هنا أي منطق تحقّق (FeatureValidator) ولا ترميز (FeatureEncoder) —
تلك مسؤولية مراحل لاحقة. ``ValidationReport`` مُعرَّف هنا فارغاً فقط لتثبيت
الشكل الذي ستملأه Phase 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Demographics:
    """معلومات ديموغرافية — تُملأ من الاستخراج القاعدي أو الـLLM."""

    age: Optional[int] = None
    gender: Optional[str] = None
    pregnancy_possible: Optional[bool] = None


@dataclass
class SymptomDescriptors:
    """واصفات OLDCARTS لعرَض واحد. فارغة افتراضياً؛ تُملأ جزئياً بمرحلة ٣.١
    للعرَض الرئيسي فقط (onset/severity)، والباقي محجوز لمراحل لاحقة."""

    onset: Optional[str] = None
    onset_days_ago: Optional[int] = None
    duration: Optional[str] = None
    severity_0_10: Optional[int] = None
    progression: Optional[str] = None
    location: Optional[str] = None
    quality: Optional[str] = None
    aggravating: List[str] = field(default_factory=list)
    relieving: List[str] = field(default_factory=list)


@dataclass
class SymptomFeature:
    """عرَض واحد كما يظهر بمجموعة الميزات — يُبنى من الأعراض المُعاد
    استخدامها (لا يُعاد استخراجها هنا)."""

    name: str
    negated: bool
    extraction_confidence: float
    is_primary: bool = False
    descriptors: SymptomDescriptors = field(default_factory=SymptomDescriptors)
    associated_symptom_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class MedicalHistory:
    """التاريخ المرضي. فارغ بمرحلة ٣.١ — لا يوجد استخراج له بعد (فجوة معروفة
    وموثَّقة، خارج نطاق هذه المرحلة)."""

    chronic_diseases: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)


@dataclass
class Lifestyle:
    """نمط الحياة. ``smoking`` فقط مُستخرَج بمرحلة ٣.١."""

    smoking: Optional[bool] = None
    alcohol: Optional[bool] = None
    occupation: Optional[str] = None


@dataclass
class DerivedFeatures:
    """حقول محسوبة من البيانات الموجودة (بلا سؤال إضافي للمريض)."""

    symptom_count: int = 0
    positive_negative_ratio: Optional[float] = None
    primary_symptom_severity: Optional[int] = None
    has_red_flag: bool = False
    interview_completeness_ratio: Optional[float] = None


@dataclass
class ValidationReport:
    """تقرير تحقّق فارغ — يُملأ فعلياً بـFeatureValidator بمرحلة ٣.٢.
    موجود هنا فقط لتثبيت الشكل المستقبلي لـClinicalFeatureSet."""

    corrected_fields: List[str] = field(default_factory=list)
    rejected_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validity_score: Optional[float] = None


@dataclass
class ClinicalFeatureSet:
    """الناتج النهائي لمرحلة استخراج الميزات (Phase 3.1) — مدخل مستقبلي
    لـFeatureValidator ثم FeatureEncoder ثم طبقة التنبؤ (لا شيء من ذلك هنا)."""

    session_id: str
    demographics: Demographics = field(default_factory=Demographics)
    symptoms: List[SymptomFeature] = field(default_factory=list)
    negated_symptoms: List[str] = field(default_factory=list)
    temperature_c: Optional[float] = None
    medical_history: MedicalHistory = field(default_factory=MedicalHistory)
    lifestyle: Lifestyle = field(default_factory=Lifestyle)
    derived: DerivedFeatures = field(default_factory=DerivedFeatures)
    validation: ValidationReport = field(default_factory=ValidationReport)
    # حقول لم يُحلّها لا الاستخراج القاعدي ولا الـLLM — شفافية للمراحل القادمة.
    unresolved_fields: List[str] = field(default_factory=list)
