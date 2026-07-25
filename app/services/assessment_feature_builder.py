"""
Healix - Assessment Feature Builder (Orchestrator)
ينسّق مرحلة بناء الميزات كاملة: سجل المقابلة → استخراج قاعدي → تحديد الناقص
→ استخراج LLM (عند اللزوم فقط) → دمج → أعراض المقابلة → ``ClinicalFeatureSet``.

لا تحقّق (FeatureValidator) ولا ترميز (FeatureEncoder) ولا تنبؤ هنا —
تلك مراحل ٣.٢+. يعتمد فقط على منافذ/كائنات محقونة بالمُنشئ، بنفس أسلوب
``ConversationService``.

يستقبل أعراض المقابلة كمُدخَل جاهز (``List[Symptom]`` من ``domain.conversation``
المُجمَّد) — لا يُعاد استدعاء الاستخراج هنا إطلاقاً، تماشياً مع مبدأ "بلا
استدعاء مكرّر" المُثبَت بمحرّك المقابلة.

أولوية المصادر (بعد جعل وكيل المقابلة مصدر الحقيقة الوحيد للاستخراج):
    سجل المقابلة  >  الاستخراج القاعدي  >  استخراج الـLLM
سجل المقابلة يأتي أولاً لأنّه نتاج حوار كامل مع المريض، لا استنباط من نصّ.
وكلّما ملأ حقلاً سقط ذلك الحقل من ``unresolved``، فلا يُستدعى الـLLM لاستخراجه
مجدداً — هذا هو إلغاء الازدواج عملياً، لا مجرّد إعادة ترتيب.

``medical_history`` كان فجوة موثَّقة (فارغ دائماً بمرحلة ٣.١ لعدم وجود مستخرج
له). وكيل المقابلة صار يستخرج الأدوية والحساسية والأمراض المزمنة والتاريخ
العائلي، فتُملأ منه مباشرةً هنا — إغلاق للفجوة بلا مستخرج جديد.
"""

from __future__ import annotations

import re
from typing import List, Optional

from app.domain.assessment import (
    ClinicalFeatureSet,
    Demographics,
    DerivedFeatures,
    Lifestyle,
    MedicalHistory,
    SymptomDescriptors,
    SymptomFeature,
    ValidationReport,
)
from app.domain.clinical_record import ClinicalRecord
from app.domain.conversation import Symptom
from app.domain.feature_extraction_rules import RuleBasedFeatureExtractor, RuleExtractionResult
from app.services.llm_feature_extractor import LLMFeatureExtractor

# أوّل عدد صحيح 0..10 داخل نصّ الشدّة ("7/10"، "شدة 8"، "٦" لا تُدعم عمداً).
_SEVERITY_RE = re.compile(r"\b(10|[0-9])\b")


class AssessmentFeatureBuilder:
    """ينسّق بناء ``ClinicalFeatureSet`` من رسائل خام وأعراض مقابلة جاهزة."""

    def __init__(self, rule_extractor: RuleBasedFeatureExtractor, llm_extractor: LLMFeatureExtractor):
        self._rules = rule_extractor
        self._llm = llm_extractor

    def build(
        self,
        session_id: str,
        raw_messages: List[str],
        symptoms: List[Symptom],
        interview_record: Optional[ClinicalRecord] = None,
    ) -> ClinicalFeatureSet:
        record = interview_record or ClinicalRecord()

        # 1. استخراج قاعدي حتمي.
        rule_result = self._rules.extract(raw_messages)

        # 2. سجل المقابلة يعلو على القاعدي (حوار مباشر مع المريض).
        seeded = self._apply_interview_record(rule_result, record)

        # 3. الحقول التي بقيت ناقصة بعد المقابلة + القاعدي.
        unresolved = self._rules.unresolved_fields(seeded)

        # 4. استخراج LLM — يُستدعى فقط لو بقي شيء ناقص (يتخطّى تلقائياً غير ذلك).
        #    ما ملأته المقابلة لا يصل هنا أبداً → لا استخراج مكرَّر.
        llm_result = self._llm.extract(raw_messages, unresolved)

        # 5. دمج: ما سبق يبقى، والـLLM يملأ الباقي فقط.
        merged = self._merge(seeded, llm_result)

        # 6. إعادة استخدام أعراض المقابلة الجاهزة (لا إعادة استخراج).
        symptom_features, negated_texts = self._build_symptom_features(symptoms, merged)

        # 7. إنتاج ClinicalFeatureSet.
        return ClinicalFeatureSet(
            session_id=session_id,
            demographics=Demographics(age=merged.age, gender=merged.gender),
            symptoms=symptom_features,
            negated_symptoms=negated_texts,
            temperature_c=merged.temperature,
            medical_history=MedicalHistory(
                chronic_diseases=list(record.chronic_conditions),
                medications=list(record.medications),
                allergies=list(record.allergies),
                family_history=list(record.family_history),
            ),
            lifestyle=Lifestyle(smoking=merged.smoking),
            derived=self._compute_derived(symptom_features, negated_texts, merged),
            validation=ValidationReport(),  # فارغ عمداً — Phase 3.2
            unresolved_fields=self._still_unresolved(merged),
        )

    # ------------------------------------------------------------------
    # سجل المقابلة → الحقول الستّة
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_interview_record(
        rule: RuleExtractionResult, record: ClinicalRecord
    ) -> RuleExtractionResult:
        """يُسقط قيم المقابلة فوق الناتج القاعدي (المقابلة لها الأولوية).

        الحقول المتقاطعة الوحيدة هي ``duration`` و``severity``؛ الباقي
        (العمر/الجنس/التدخين/الحرارة) لا يستخرجه وكيل المقابلة حالياً فيبقى
        للقاعدي ثمّ الـLLM. الشدّة نصّ حرّ بالمقابلة ورقم هنا: تُحوَّل بأول
        عدد 0..10، ويبقى None عند تعذّر التحويل — لا تخمين.
        """
        severity = rule.severity
        if record.severity:
            match = _SEVERITY_RE.search(record.severity)
            if match:
                severity = int(match.group(1))

        return RuleExtractionResult(
            age=rule.age,
            gender=rule.gender,
            smoking=rule.smoking,
            temperature=rule.temperature,
            duration_text=record.duration or rule.duration_text,
            duration_days=rule.duration_days,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # الدمج
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(rule: RuleExtractionResult, llm: RuleExtractionResult) -> RuleExtractionResult:
        """القاعدي يبقى كما هو لأي حقل حلّه؛ الـLLM يملأ الباقي فقط."""
        return RuleExtractionResult(
            age=rule.age if rule.age is not None else llm.age,
            gender=rule.gender if rule.gender is not None else llm.gender,
            smoking=rule.smoking if rule.smoking is not None else llm.smoking,
            temperature=rule.temperature if rule.temperature is not None else llm.temperature,
            duration_text=rule.duration_text if rule.duration_text is not None else llm.duration_text,
            duration_days=rule.duration_days,  # الـLLM لا يُنتج قيمة رقمية بمرحلة ٣.١
            severity=rule.severity if rule.severity is not None else llm.severity,
        )

    @staticmethod
    def _still_unresolved(merged: RuleExtractionResult) -> List[str]:
        names = []
        if merged.age is None:
            names.append("age")
        if merged.gender is None:
            names.append("gender")
        if merged.smoking is None:
            names.append("smoking")
        if merged.temperature is None:
            names.append("temperature")
        if merged.duration_text is None:
            names.append("duration")
        if merged.severity is None:
            names.append("severity")
        return names

    # ------------------------------------------------------------------
    # بناء الأعراض (العرَض الرئيسي = أوّل عرَض مُثبَت، بنفس اصطلاح clinical.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_symptom_features(
        symptoms: List[Symptom], merged: RuleExtractionResult
    ):
        features: List[SymptomFeature] = []
        negated_texts: List[str] = []
        primary_assigned = False

        for symptom in symptoms:
            if symptom.negated:
                negated_texts.append(symptom.text)
                features.append(
                    SymptomFeature(
                        name=symptom.text,
                        negated=True,
                        extraction_confidence=symptom.confidence,
                    )
                )
                continue

            is_primary = not primary_assigned
            descriptors = SymptomDescriptors()
            if is_primary:
                primary_assigned = True
                descriptors = SymptomDescriptors(
                    onset=merged.duration_text,
                    onset_days_ago=merged.duration_days,
                    severity_0_10=merged.severity,
                )

            features.append(
                SymptomFeature(
                    name=symptom.text,
                    negated=False,
                    extraction_confidence=symptom.confidence,
                    is_primary=is_primary,
                    descriptors=descriptors,
                )
            )

        return features, negated_texts

    # ------------------------------------------------------------------
    # الحقول المحسوبة
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_derived(
        symptom_features: List[SymptomFeature],
        negated_texts: List[str],
        merged: RuleExtractionResult,
    ) -> DerivedFeatures:
        positive_count = sum(1 for s in symptom_features if not s.negated)
        negated_count = len(negated_texts)
        ratio = round(positive_count / negated_count, 2) if negated_count else None

        return DerivedFeatures(
            symptom_count=positive_count,
            positive_negative_ratio=ratio,
            primary_symptom_severity=merged.severity,
            has_red_flag=False,  # غير مُنفَّذ بمرحلة ٣.١ — يحتاج قاموس قواعد (Phase 3.2)
            interview_completeness_ratio=None,  # يحتاج asked_slots/turn_count — خارج نطاق ٣.١
        )
