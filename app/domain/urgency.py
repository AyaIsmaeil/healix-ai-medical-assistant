"""
Healix - Urgency (Triage) Domain Models (Phase 3.5)
نماذج المجال الخالصة لتقييم درجة الاستعجال — العقد الثابت الذي يستهلكه أي
adapter مستقبلي (قاعدي الآن، نموذج ML لاحقاً) عبر ``UrgencyClassifierPort``
(انظر ``domain.ports``). تبديل الـadapter لا يغيّر هذا العقد إطلاقاً.

بلا FastAPI، بلا شبكة، بلا نظام ملفات، بلا أي اعتماد على ML حقيقي — Enum
+ dataclass خالصان فقط، بنفس فلسفة ``domain.prediction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UrgencyLevel(str, Enum):
    """مستوى الاستعجال (Triage) — من الأخطر إلى الأقل خطورة. ``str, Enum``
    بنفس نمط ``InterviewStatus`` بـ``domain.conversation`` (تسلسل JSON طبيعي)."""

    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    SEMI_URGENT = "SEMI_URGENT"
    NON_URGENT = "NON_URGENT"


@dataclass
class UrgencyAssessment:
    """نتيجة تقييم الاستعجال الكاملة — تُعاد دائماً من أي adapter."""

    level: UrgencyLevel
    score: float
    explanation: str
