"""
Healix - Feature Encoder (Phase 3.3)
يحوّل ``ClinicalFeatureSet`` (المُتحقَّق منه بمرحلة ٣.٢) إلى ``EncodedFeatures``
— قاموس مسطّح، حتمي، مسمّى، جاهز لأي مُتنبِّئ ML لاحق (Phase 3.4+: قاعدي،
XGBoost، Random Forest، CatBoost...).

منطق مجال خالص — بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على
وكيل المقابلة أو Whisper أو مزوّدي الـLLM. مخطّط الترميز (أسماء الحقول، الترتيب،
الفئات المسموحة) يصله محقوناً بالمُنشئ من ``infrastructure.dictionary_loader``
— لا قيم ثابتة بالكود؛ "النقاء" هنا يعني صفر I/O وقت الترميز نفسه، بنفس مبدأ
حقن ``FeatureValidator``.

FeatureEncoder هو المصدر الوحيد للترميز — لا مُتنبِّئ لاحق يُعيد تنفيذ منطق
ترميزه الخاص؛ الجميع يستهلك ``EncodedFeatures`` كما هي.

FeatureEncoder ليس منفذاً (Port) وليس قابلاً للاستبدال. تطوّر منطق الترميز
يكون عبر نسخة مخطّط جديدة (v2.json) لا استبدال هذا الصنف — لذا يحمل كل
``EncodedFeatures`` دائماً ``feature_schema_version`` الذي أُنتِج به.

قاعدتان حاسمتان لصحّة البيانات الطبية:
1. القيمة الفئوية المفقودة (None) → كل أعمدة one-hot المقابلة تصير null، لا
   0 — تصفيرها كان سيَختلق تأكيداً كاذباً ("مؤكَّد ليس X") من غياب معرفة.
2. كل حقل مُعرَّف بالمخطّط يظهر دائماً بـ``features`` (حتى لو null) — لا حذف
   مفاتيح شرطياً، لأنّ المخطّط عقد ثابت بين التدريب والاستدلال؛ أي مُتنبِّئ
   لاحق يعتمد على شكل قاموس ثابت المفاتيح، لا وجود شرطي.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.domain.assessment import ClinicalFeatureSet, SymptomFeature


@dataclass
class EncodedFeatures:
    """ناتج الترميز — قاموس مسطّح فقط (لا numpy، لا pandas)، حتمي الترتيب."""

    feature_schema_version: str
    features: Dict[str, Any] = field(default_factory=dict)
    categorical_index: Dict[str, List[str]] = field(default_factory=dict)


class FeatureEncoder:
    """يحوّل ``ClinicalFeatureSet`` إلى ``EncodedFeatures`` وفق مخطّط محقون."""

    def __init__(self, schema: Dict[str, Any]):
        self._schema = schema
        self._schema_version: str = schema["schema_version"]
        self._feature_order: List[str] = list(schema["feature_order"])
        self._numeric_fields: Dict[str, Any] = schema.get("numeric_fields", {})
        self._boolean_fields: Dict[str, Any] = schema.get("boolean_fields", {})
        self._categorical_fields: Dict[str, Any] = schema.get("categorical_fields", {})

    def encode(self, features: ClinicalFeatureSet) -> EncodedFeatures:
        """يرمّز ``features`` إلى ``EncodedFeatures`` — بلا أي جانب تأثير."""
        raw_values = self._extract_raw_values(features)

        columns: Dict[str, Any] = {}

        for name in self._numeric_fields:
            columns[name] = self._encode_numeric(raw_values.get(name))

        for name in self._boolean_fields:
            columns[name] = self._encode_boolean(raw_values.get(name))

        for name, spec in self._categorical_fields.items():
            columns.update(
                self._encode_categorical(raw_values.get(name), spec["values"], spec["one_hot_prefix"])
            )

        # ترتيب حتمي مضمون — نفس المدخل يُنتج نفس ترتيب المفاتيح دائماً.
        ordered_features = {name: columns[name] for name in self._feature_order}

        categorical_index = {
            name: list(spec["values"]) for name, spec in self._categorical_fields.items()
        }

        return EncodedFeatures(
            feature_schema_version=self._schema_version,
            features=ordered_features,
            categorical_index=categorical_index,
        )

    # ------------------------------------------------------------------
    # استخراج القيم الخام من ClinicalFeatureSet (تعيين صريح، لا انعكاس ديناميكي)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_raw_values(features: ClinicalFeatureSet) -> Dict[str, Any]:
        primary = FeatureEncoder._find_primary_symptom(features.symptoms)
        return {
            "age": features.demographics.age,
            "gender": features.demographics.gender,
            # عمداً من وصف العرَض الرئيسي، لا من derived.primary_symptom_severity:
            # FeatureValidator (Phase 3.2) يتحقّق من symptoms[primary].descriptors
            # .severity_0_10 ويصحّحه (قصّ/null) عند الحاجة، لكنه لا يُزامن النسخة
            # الملخَّصة بـderived — قراءتها هنا كانت ستتجاوز التحقّق بصمت.
            "severity": primary.descriptors.severity_0_10 if primary else None,
            "smoking": features.lifestyle.smoking,
            "temperature_c": features.temperature_c,
            "progression": primary.descriptors.progression if primary else None,
        }

    @staticmethod
    def _find_primary_symptom(symptoms: List[SymptomFeature]) -> Optional[SymptomFeature]:
        return next((symptom for symptom in symptoms if symptom.is_primary), None)

    # ------------------------------------------------------------------
    # قواعد الترميز العامة (بلا أي اسم/فئة ثابتة بالكود)
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_numeric(value: Any) -> Optional[Any]:
        """رقمي يبقى رقمياً كما هو؛ null يبقى null — لا اختلاق قيم."""
        return value if value is not None else None

    @staticmethod
    def _encode_boolean(value: Any) -> Optional[int]:
        """منطقي → 0/1؛ null يبقى null."""
        if value is None:
            return None
        return 1 if value else 0

    @staticmethod
    def _encode_categorical(
        value: Any, allowed_values: List[str], prefix: str
    ) -> Dict[str, Optional[int]]:
        """فئوي → one-hot. القيمة المفقودة أو غير المعروفة بالمخطّط → كل
        الأعمدة null (لا تُصفَّر، ولا تُختلَق فئة جديدة)."""
        columns: Dict[str, Optional[int]] = {f"{prefix}_{v}": None for v in allowed_values}

        if value is None or value not in allowed_values:
            return columns

        for candidate in allowed_values:
            columns[f"{prefix}_{candidate}"] = 1 if candidate == value else 0
        return columns
