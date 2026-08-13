"""
Healix - Rule-Based Specialty Recommender (Phase 3.6)
أول تنفيذ فعلي لـ``SpecialtyRecommenderPort`` — قواعد حتمية بسيطة
(Placeholder)، بلا أي مكتبة ML. يحقّق نفس العقد الذي ستستخدمه لاحقاً
adapters حقيقية — استبدال هذا الصنف فقط مستقبلاً، بلا تغيير على بقية النظام.

يعتمد حصراً على ``ClinicalFeatureSet`` (+ ``DiseasePredictionResult``
اختياري — انظر توثيق القرار المعماري بـ``domain.ports.SpecialtyRecommenderPort``)
— لا يعرف شيئاً عن AssessmentFeatureBuilder أو FeatureValidator أو
FeatureEncoder أو UrgencyClassifier أو أي مزوّد LLM.

⚠️ ملاحظة تصميم (تكرار مقصود، لا اقتران): مطابقة الكلمات المفتاحية العربية
لأسماء الأعراض (``symptoms[].name`` نص خام، لا فئة إنجليزية معيارية) مستقلّة
تماماً عن ``domain.clinical`` (محرّك المقابلة المجمَّد) — قوائم كلمات مفتاحية
خاصة مُعرَّفة محلياً أدناه، لا استيراد، بنفس مبدأ الاستقلال المُتّبَع
بـRuleBasedUrgencyClassifier.

⚠️ ملاحظة واقعية: أسماء أمراض RuleBasedDiseasePredictor الحالية ("Febrile
Illness", "Severe Condition") لا تطابق أي مفتاح بـspecialty_lookup.yaml
الحالي (أمراض حقيقية: influenza, pneumonia...) — فالأولوية ٢ (specialty_lookup)
عملياً تتخطّى دائماً بالوضع الراهن وتستقرّ التوصية على الاستدلال السريري أو
الاحتياطي عند استخدام المُتنبِّئ القاعدي. هذا متوقَّع (كل مرحلة Placeholder
مستقلة) لا خلل، ومُختبَر صراحة أدناه.

الأولوية ١ (ADR-04، إضافة لا استبدال): ``disease_metadata`` — قاموس واحد
مُحقَن (``infrastructure.dictionary_loader.load_disease_metadata``) يربط
أسماء أمراض DDXPlus الحقيقية الـ٤٩ (مخرجات المُتنبِّئ ML الفعلي) بتخصّص
مُراجَع مسبقاً، لا حساب ICD->تخصّص حيّ هنا. التدفّق: Disease -> disease_metadata
-> Specialty. لا specialty فارغ أبداً — الحالات الغامضة (تعارض فصل/بلوك
ICD-10 مع الممارسة السريرية الفعلية) مُعلَّمة ``requires_review=true`` مع
``review_reason`` موثَّق، وتُعاد كـ"General Medicine" بثقة أخفض، لا بصمت.

لا يخترع بيانات غائبة: أي حقل None/فارغ يُتخطّى بلا افتراض قيمة بديلة —
تستقرّ التوصية على "General Medicine" إن لم تُطابَق أي قاعدة.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.domain.assessment import ClinicalFeatureSet
from app.domain.prediction import DiseasePredictionResult
from app.domain.specialty import SpecialtyRecommendation

_CONFIDENCE_METADATA_CLEAN = 0.90
_CONFIDENCE_METADATA_REVIEW = 0.55
_CONFIDENCE_LOOKUP = 0.90
_CONFIDENCE_CLINICAL_INFERENCE = 0.70
_CONFIDENCE_FALLBACK = 0.50

_FALLBACK_SPECIALTY = "General Medicine"
_FALLBACK_EXPLANATION = "No specific specialty identified; recommending General Medicine."

# كلمات مفتاحية عربية → (التخصّص، التفسير) — بترتيب الأولوية (أوّل فحص
# يُطابَق يفوز). مستقلّة تماماً عن domain.clinical (لا استيراد).
_SYMPTOM_KEYWORD_RULES: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (("صدر", "ذبحة"), "Cardiology", "Chief complaint is chest pain."),
    (("صداع", "راس", "رأس"), "Neurology", "Chief complaint is headache."),
    (("سعال", "كحة"), "Pulmonology", "Persistent cough presentation."),
    (("جلد", "طفح", "حكة"), "Dermatology", "Skin rash presentation."),
    (("بطن", "معدة", "مغص"), "General Surgery", "Abdominal pain presentation."),
    (("مفصل", "مفاصل", "ركبة"), "Orthopedics", "Joint pain presentation."),
    (("بول", "تبول", "مثانة"), "Urology", "Urinary symptoms presentation."),
)

_FEVER_KEYWORDS = ("حرار", "حمى", "حمّى", "سخون")


class RuleBasedSpecialtyRecommender:
    """يطبّق قواعد حتمية بسيطة (Placeholder): بحث بالمرض ← استدلال سريري ← احتياطي."""

    def __init__(
        self,
        specialty_lookup: Dict[str, Dict[str, str]],
        disease_metadata: Optional[Dict[str, Dict[str, object]]] = None,
    ):
        self._lookup = specialty_lookup
        # {disease_name: {icd10, severity, specialty, requires_review, review_reason, ...}}
        # اختياري (افتراضي فارغ) لبقاء التوافق مع أي موضع بناء قديم للصنف.
        self._disease_metadata: Dict[str, Dict[str, object]] = (
            disease_metadata.get("diseases", {}) if disease_metadata else {}
        )

    def recommend(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: Optional[DiseasePredictionResult] = None,
    ) -> SpecialtyRecommendation:
        """يُعيد ``SpecialtyRecommendation`` دائماً — احتياطي General Medicine."""
        by_metadata = self._recommend_from_disease_metadata(prediction_result)
        if by_metadata is not None:
            return by_metadata

        by_disease = self._recommend_from_top_disease(prediction_result)
        if by_disease is not None:
            return by_disease

        by_clinical = self._recommend_from_clinical_features(clinical_features)
        if by_clinical is not None:
            return by_clinical

        return SpecialtyRecommendation(
            specialty=_FALLBACK_SPECIALTY,
            confidence=_CONFIDENCE_FALLBACK,
            explanation=_FALLBACK_EXPLANATION,
        )

    # ------------------------------------------------------------------
    # الأولوية ١ (ADR-04): Disease -> disease_metadata -> Specialty
    # ------------------------------------------------------------------
    def _recommend_from_disease_metadata(
        self, prediction_result: Optional[DiseasePredictionResult]
    ) -> Optional[SpecialtyRecommendation]:
        if prediction_result is None or not prediction_result.predictions:
            return None
        if not self._disease_metadata:
            return None

        top = max(prediction_result.predictions, key=lambda prediction: prediction.score)
        entry = self._disease_metadata.get(top.disease)
        if entry is None:
            return None

        specialty = entry["specialty"]
        if entry.get("requires_review"):
            return SpecialtyRecommendation(
                specialty=specialty,
                confidence=_CONFIDENCE_METADATA_REVIEW,
                explanation=(
                    f"Highest ranked predicted disease is {top.disease}. "
                    f"No single specialty is defensible from ICD-10 classification alone "
                    f"(reason: {entry.get('review_reason')}) -- recommending {specialty} "
                    f"pending clinical review."
                ),
            )

        return SpecialtyRecommendation(
            specialty=specialty,
            confidence=_CONFIDENCE_METADATA_CLEAN,
            explanation=(
                f"Highest ranked predicted disease is {top.disease} "
                f"(ICD-10 {entry.get('icd10')}, {entry.get('icd10_block_name_who')})."
            ),
        )

    # ------------------------------------------------------------------
    # الأولوية ٢: specialty_lookup.yaml القديم (احتياطي، لأمراض غير موجودة
    # بـdisease_metadata — مثلاً مخرجات RuleBasedDiseasePredictor Placeholder)
    # ------------------------------------------------------------------
    def _recommend_from_top_disease(
        self, prediction_result: Optional[DiseasePredictionResult]
    ) -> Optional[SpecialtyRecommendation]:
        if prediction_result is None or not prediction_result.predictions:
            return None

        top = max(prediction_result.predictions, key=lambda prediction: prediction.score)
        entry = self._lookup.get(top.disease.strip().lower())
        if entry is None:
            return None

        return SpecialtyRecommendation(
            specialty=entry["specialty"],
            confidence=_CONFIDENCE_LOOKUP,
            explanation=f"Highest ranked predicted disease is {top.disease}.",
        )

    # ------------------------------------------------------------------
    # الأولوية ٣: استدلال سريري من ClinicalFeatureSet مباشرة
    # ------------------------------------------------------------------
    def _recommend_from_clinical_features(
        self, clinical_features: ClinicalFeatureSet
    ) -> Optional[SpecialtyRecommendation]:
        if clinical_features.demographics.pregnancy_possible is True:
            return SpecialtyRecommendation(
                specialty="Obstetrics and Gynecology",
                confidence=_CONFIDENCE_CLINICAL_INFERENCE,
                explanation="Pregnancy-related presentation.",
            )

        positive_symptom_names = [
            symptom.name for symptom in clinical_features.symptoms if not symptom.negated
        ]

        for keywords, specialty, explanation in _SYMPTOM_KEYWORD_RULES:
            if self._any_symptom_matches(positive_symptom_names, keywords):
                return SpecialtyRecommendation(
                    specialty=specialty,
                    confidence=_CONFIDENCE_CLINICAL_INFERENCE,
                    explanation=explanation,
                )

        if self._any_symptom_matches(positive_symptom_names, _FEVER_KEYWORDS):
            return SpecialtyRecommendation(
                specialty="Family Medicine",
                confidence=_CONFIDENCE_CLINICAL_INFERENCE,
                explanation="Fever-only presentation.",
            )

        return None

    @staticmethod
    def _any_symptom_matches(symptom_names: List[str], keywords: Tuple[str, ...]) -> bool:
        return any(keyword in name for name in symptom_names for keyword in keywords)
