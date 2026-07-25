"""
Healix - Disease Prediction Domain Models (Phase 3.4)
نماذج المجال الخالصة لنتائج التنبؤ بالمرض — العقد الثابت الذي يستهلكه أي
adapter مستقبلي (قاعدي الآن، XGBoost/RandomForest/CatBoost لاحقاً) عبر
``DiseasePredictorPort`` (انظر ``domain.ports``). تبديل الـadapter لا يغيّر
هذا العقد إطلاقاً — هذا هو المقصود بـ"العقد الثابت بين المُتنبِّئات".

بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على ML حقيقي —
dataclasses خالصة فقط، بنفس فلسفة ``domain.assessment`` و``domain.feature_encoder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DiseasePrediction:
    """احتمال مرض واحد مُتنبَّأ به."""

    disease: str
    score: float
    explanation: str


@dataclass
class DiseasePredictionResult:
    """ناتج التنبؤ الكامل — يُعاد دائماً من أي adapter، حتى لو ``predictions``
    فارغة (لم يُطابَق أي قاعدة/نموذج)."""

    predictions: List[DiseasePrediction] = field(default_factory=list)
    predictor_version: str = ""
