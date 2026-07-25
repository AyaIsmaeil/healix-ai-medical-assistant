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
الحالي (أمراض حقيقية: influenza, pneumonia...) — فالأولوية ١ عملياً تتخطّى
دائماً بالوضع الراهن وتستقرّ التوصية على الاستدلال السريري أو الاحتياطي.
هذا متوقَّع (كل مرحلة Placeholder مستقلة) لا خلل، ومُختبَر صراحة أدناه.

لا يخترع بيانات غائبة: أي حقل None/فارغ يُتخطّى بلا افتراض قيمة بديلة —
تستقرّ التوصية على "General Medicine" إن لم تُطابَق أي قاعدة.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.domain.assessment import ClinicalFeatureSet
from app.domain.prediction import DiseasePredictionResult
from app.domain.specialty import SpecialtyRecommendation

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

    def __init__(self, specialty_lookup: Dict[str, Dict[str, str]]):
        self._lookup = specialty_lookup

    def recommend(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: Optional[DiseasePredictionResult] = None,
    ) -> SpecialtyRecommendation:
        """يُعيد ``SpecialtyRecommendation`` دائماً — احتياطي General Medicine."""
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
    # الأولوية ١: أعلى مرض مُتنبَّأ به (إن وُجد وطابق قاموس التخصّصات)
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
    # الأولوية ٢: استدلال سريري من ClinicalFeatureSet مباشرة
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
