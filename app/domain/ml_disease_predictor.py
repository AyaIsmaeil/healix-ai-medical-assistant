"""
Healix - ML Disease Predictor (Phase 3.4, مسار موازٍ)
تنفيذ ثانٍ لـ``DiseasePredictorPort`` بجانب ``RuleBasedDiseasePredictor`` —
لا يحذفه ولا يستبدله؛ يُحقَن بدلاً منه فقط عند تفعيل ``config.USE_ML_PREDICTOR``
(انظر ``main.py``). نفس العقد بالضبط: يستهلك ``EncodedFeatures`` فقط، لا يقرأ
أي ملف من القرص بنفسه — النموذج و``feature_names`` و``label_encoder`` تصله
محقونة بالمُنشئ (نفس نمط ``FeatureEncoder(schema=...)``)، محمَّلة سلفاً عبر
``infrastructure.model_loader.ModelLoader``.

⚠️ فجوة معماريّة مُكتشَفة وموثَّقة (بنفس روح ``rule_based_predictor.py``):
مخطّط ترميز الميزات الحالي (v1.json) لا يُصدِّر أي عمود برمز ``E_*`` — هذه
الأعمدة (208 ثنائي + 764 one-hot/multi-hot من DDXPlus) لا تظهر إطلاقاً في
``EncodedFeatures.features`` حتى يُضاف استخراجها بمخطّط مستقبلي (v2.json).
لذا هذا المُتنبِّئ حالياً **يُعيد نتيجة فارغة دائماً** بلا خطأ — ليس عطلاً، بل
سلوك مُتوقَّع وموثَّق حتى تُسَدّ الفجوة، بدل اختلاق قيم افتراضية لأدلة غائبة.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.domain.feature_encoder import EncodedFeatures
from app.domain.prediction import DiseasePrediction, DiseasePredictionResult
from app.exceptions import InferenceError

PREDICTOR_VERSION = "ml-logreg-v1"

_EVIDENCE_PREFIX = "E_"
_TOP_N = 5


class MLDiseasePredictor:
    """يستنتج احتمالات الأمراض عبر نموذج ML مدرَّب مُحقَن (لا يقرأ القرص بنفسه)."""

    predictor_version: str = PREDICTOR_VERSION

    def __init__(self, model: Any, feature_names: List[str], label_encoder: Any) -> None:
        self._model = model
        self._feature_names = feature_names
        self._label_encoder = label_encoder
        # فهرس بحث O(1) — نفس أسلوب FeatureEncoder._feature_order/_categorical_fields.
        self._feature_index: Dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    def predict(self, encoded_features: EncodedFeatures) -> DiseasePredictionResult:
        """يُعيد ``DiseasePredictionResult`` دائماً — حتى لو لم يُطابَق أي دليل."""
        values = encoded_features.features

        vector, matched_evidence_count = self._build_vector(values)
        if matched_evidence_count == 0:
            # لا رموز E_* مطابقة (متوقَّع حالياً — انظر توثيق الفجوة أعلى الملف)
            # — نتيجة فارغة بلا خطأ، بدل استنتاج على متجه صفري لا معنى له.
            return DiseasePredictionResult(predictions=[], predictor_version=self.predictor_version)

        try:
            proba = self._model.predict_proba(vector.reshape(1, -1))[0]
        except Exception as exc:  # noqa: BLE001 - أي فشل استنتاج يُغلَّف بنوع خطأ الدومين
            raise InferenceError(f"فشل استنتاج MLDiseasePredictor: {exc}") from exc

        predictions = self._top_predictions(proba, matched_evidence_count)
        return DiseasePredictionResult(predictions=predictions, predictor_version=self.predictor_version)

    # ------------------------------------------------------------------
    # بناء متجه الميزات
    # ------------------------------------------------------------------
    def _build_vector(self, values: Dict[str, Any]) -> tuple[np.ndarray, int]:
        """متجه بطول ``feature_names``، مبنيّ من مفاتيح ``E_*`` الموجودة فعلياً
        في ``values`` + AGE/SEX_M. لا قيمة ثلاثية: مفتاح غائب أو ``None`` يبقى
        0 (نفس منطق ترميز DDXPlus الأصلي — بلا تمييز "لم يُسأل" عن "نُفي")."""
        vector = np.zeros(len(self._feature_names))
        matched_evidence_count = 0

        for key, value in values.items():
            if not key.startswith(_EVIDENCE_PREFIX):
                continue
            index = self._feature_index.get(key)
            if index is None:
                # رمز E_* غير معروف لهذا النموذج (مثلاً مخطّط مستقبلي أوسع) — يُتجاهَل بصمت.
                continue
            vector[index] = 1 if value else 0
            matched_evidence_count += 1

        age_index = self._feature_index.get("AGE")
        if age_index is not None:
            age = values.get("age")
            vector[age_index] = age if age is not None else 0

        sex_index = self._feature_index.get("SEX_M")
        if sex_index is not None:
            gender_male = values.get("gender_male")
            vector[sex_index] = 1 if gender_male else 0

        return vector, matched_evidence_count

    # ------------------------------------------------------------------
    # أعلى N احتمال — ترتيب حتمي
    # ------------------------------------------------------------------
    def _top_predictions(self, proba: np.ndarray, matched_evidence_count: int) -> List[DiseasePrediction]:
        """أعلى ``_TOP_N`` احتمالات، مُرتَّبة تنازلياً. ``np.argsort`` مستقرّ
        (ترتيب ``label_encoder.classes_`` الأبجدي يفصل التعادل حتمياً)."""
        top_indices = np.argsort(-proba)[:_TOP_N]
        return [
            DiseasePrediction(
                disease=str(self._label_encoder.classes_[i]),
                score=float(proba[i]),
                explanation=(
                    f"احتمال {proba[i] * 100:.1f}% وفق النموذج المدرَّب ({self.predictor_version})، "
                    f"بناءً على {matched_evidence_count} دليلاً مطابقاً."
                ),
            )
            for i in top_indices
        ]
