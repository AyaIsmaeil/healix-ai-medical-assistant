"""
Healix - Clinical Priority Engine (P1)
تحديد **الشكوى الرئيسية الحالية** حتمياً بعد كل رسالة مريض.

المشكلة التي يحلّها
--------------------
كان المحرّك يعتمد ``positives[0]`` — أوّل عرَض ذُكر — كشكوى رئيسية **إلى
الأبد**. مريض يقول "صداع" ثم "ألم ضاغط بصدري" في دور لاحق كانت المقابلة
تستمرّ في استجوابه عن الصداع. هذا عكس المنطق السريري: التاريخ المرضي
يتمحور حول الشكوى الأعلى حدّة، لا الأسبق ذكراً.

مَن يقرّر
---------
**المحرّك وحده.** الـLLM يستخرج معلومات منظَّمة فقط؛ لا يُقرأ منه أي حكم
عن الشكوى الرئيسية. الترتيب هنا دوال خالصة حتمية: نفس المدخلات تُعطي نفس
الترتيب دائماً، وبلا أي استدعاء شبكة.

مكوّنات الدرجة
---------------
    score = acuity              (0..100)  الأهمية السريرية من الجدول
          + red_flag_boost      (0/60/100) مستوى الطوارئ المُطلَق فعلاً
          + severity_boost      (0..25)   شدّة بكلام المريض نفسه
          + emphasis_boost      (0..15)   تأكيد/إلحاح المريض
          + recency_boost       (0..10)   معلومة جديدة تستحق التفاتاً

فواصل الأوزان مقصودة: الطوارئ تغلب كل شيء، والحدّة تغلب الأقدمية — فلا
يستطيع عرَض قديم شائع أن يهزم عرَضاً إنذارياً بتراكم نقاط صغيرة.

حدّ المجال
-----------
لا تشخيص هنا. ``acuity`` تعني "كم يستحق هذا العرَض تصدّر الأسئلة"، لا
احتمال مرض ولا خطورة نهائية — تلك مسؤولية طبقات لاحقة خلف منافذها.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.domain.red_flag_engine import normalize_arabic
from app.domain.red_flags import RedFlagAssessment, RiskLevel

# سقوف مكوّنات الدرجة — ثوابت مسمّاة لا أرقام سحرية.
SEVERITY_HIGH_BOOST = 25.0
SEVERITY_MODERATE_BOOST = 12.0
EMPHASIS_BOOST = 15.0
RECENCY_MAX_BOOST = 10.0
RED_FLAG_IMMEDIATE_BOOST = 100.0
RED_FLAG_URGENT_BOOST = 60.0

# الحدّ الأقصى لفارق acuity الذي يجوز لـrecency_boost تجاوزه (C-2).
# دون هذا القيد كان recency_boost (+10 ثابتة) يقلب الشكوى الرئيسية بمجرّد
# تأخّر ذكرها بدور واحد، حتى لو كانت أقلّ حدّة فعلياً بفارق معتبر — مثال
# مُثبَت: ألم الصدر (acuity=95) يخسر الصدارة لضيق التنفّس (acuity=92) بمجرّد
# ذكر الأخير بدور أحدث (92+10=102 > 95+0=95)، رغم فارق 3 نقاط فقط لصالح
# ألم الصدر (انظر HEALIX_INTERVIEW_ASSESSMENT_AUDIT.md، القسم C-2).
# القيمة 2.0 مُشتقّة من تكتّل القيم العليا الفعلي بجدول الإنتاج
# (chest_pain=95, syncope=94, focal_neuro_deficit=96, seizure=93,
# dyspnea=92 — كلّها ضمن مجال 4 نقاط) لا رقماً مُختلَقاً: تكفي لترك
# recency_boost يحسم بين أعراض متقاربة الحدّة فعلاً (فارق ≤2)، بينما تمنعه
# من تجاوز فارق 3 نقاط فأكثر بين عرَضين مختلفَي الحدّة سريرياً بوضوح.
RECENCY_ACUITY_TOLERANCE = 2.0


@dataclass(frozen=True)
class SymptomPriority:
    """درجة عرَض واحد مع تفكيكها — قابلة للتفسير لا صندوقاً أسود."""

    symptom_text: str
    concept: Optional[str]
    acuity: float
    red_flag_boost: float = 0.0
    severity_boost: float = 0.0
    emphasis_boost: float = 0.0
    recency_boost: float = 0.0
    turn_number: int = 0

    @property
    def score(self) -> float:
        return (
            self.acuity
            + self.red_flag_boost
            + self.severity_boost
            + self.emphasis_boost
            + self.recency_boost
        )

    @property
    def rationale(self) -> str:
        """سبب الترتيب بصيغة مقروءة — للتدقيق والتفسير، لا للمريض."""
        parts = [f"acuity={self.acuity:g}"]
        if self.red_flag_boost:
            parts.append(f"red_flag=+{self.red_flag_boost:g}")
        if self.severity_boost:
            parts.append(f"severity=+{self.severity_boost:g}")
        if self.emphasis_boost:
            parts.append(f"emphasis=+{self.emphasis_boost:g}")
        if self.recency_boost:
            parts.append(f"recency=+{self.recency_boost:g}")
        return " ".join(parts)


class ClinicalPriorityEngine:
    """يرتّب الأعراض المُثبَتة سريرياً. عديم الحالة وآمن للمشاركة."""

    def __init__(
        self,
        concepts: Dict[str, Dict[str, Any]],
        severity_terms: Dict[str, Sequence[str]],
        emphasis_terms: Sequence[str],
        default_acuity: float = 30.0,
        version: str = "unknown",
    ) -> None:
        self.version = version
        self._default_acuity = float(default_acuity)

        # تُطبَّع المصطلحات مرّة واحدة عند البناء لا مع كل طلب.
        # الأطول أولاً: "الم بصدري" تُفضَّل على "الم" عند التداخل.
        self._concepts: Dict[str, Tuple[float, Tuple[str, ...], Tuple[Tuple[str, ...], ...]]] = {}
        for name, spec in concepts.items():
            terms = tuple(sorted(
                {normalize_arabic(t) for t in spec.get("terms", ()) if normalize_arabic(t)},
                key=len, reverse=True,
            ))
            # ``all_tokens``: مجموعات رموز يجب حضور واحد من كلٍّ منها. تُطابِق
            # بصرف النظر عن ترتيب الكلمات أو ما يتوسّطها — وهو ما تعجز عنه
            # مطابقة العبارة الكاملة. ("ألم ضاغط في الصدر" لا يطابق أي عبارة
            # مُعدَّدة، لكنه يحقّق [ألم] + [صدر] فوراً.)
            token_groups = tuple(
                tuple(sorted(
                    {normalize_arabic(tok) for tok in group if normalize_arabic(tok)},
                    key=len, reverse=True,
                ))
                for group in (spec.get("all_tokens") or ())
            )
            self._concepts[name] = (
                float(spec.get("acuity", default_acuity)), terms, token_groups,
            )

        self._severity_high = self._normalize_all(severity_terms.get("high", ()))
        self._severity_moderate = self._normalize_all(severity_terms.get("moderate", ()))
        self._emphasis = self._normalize_all(emphasis_terms)

    @staticmethod
    def _normalize_all(terms: Sequence[str]) -> Tuple[str, ...]:
        return tuple(sorted(
            {normalize_arabic(t) for t in terms if normalize_arabic(t)},
            key=len, reverse=True,
        ))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalPriorityEngine":
        return cls(
            concepts=data.get("concepts", {}),
            severity_terms=data.get("severity_terms", {}) or {},
            emphasis_terms=data.get("emphasis_terms", ()) or (),
            default_acuity=float(data.get("default_acuity", 30)),
            version=str(data.get("version", "unknown")),
        )

    # ------------------------------------------------------------------
    # الواجهة العامة
    # ------------------------------------------------------------------
    def rank(
        self,
        symptoms: Sequence[Any],
        raw_messages: Sequence[str] = (),
        risk: Optional[RedFlagAssessment] = None,
    ) -> List[SymptomPriority]:
        """ترتيب الأعراض المُثبَتة تنازلياً حسب الأولوية السريرية.

        الأعراض المنفية تُستبعد: المريض نفاها، فلا يصحّ بناء التاريخ حولها.
        """
        positives = [s for s in symptoms if not getattr(s, "negated", False)]
        if not positives:
            return []

        context = normalize_arabic(" ".join(raw_messages))
        red_flag_boost = self._red_flag_boost(risk)
        max_turn = max((getattr(s, "turn_number", 0) or 0) for s in positives)
        # أعلى acuity بين أعراض هذا الدور المُثبَتة — يُستخدَم فقط لضبط سقف
        # recency_boost (C-2)، لا يُخزَّن ولا يُعاد. حساب مسبق لأنّ acuity
        # عرَض واحد لا يكفي وحده لمعرفة موقعه من بقية أعراض الجلسة.
        max_acuity = max(
            self._match_concept(getattr(symptom, "text", "") or "")[1]
            for symptom in positives
        )

        scored = [
            self._score(symptom, context, red_flag_boost, max_turn, max_acuity)
            for symptom in positives
        ]

        # ترتيب مستقرّ ومُتوقَّع: الدرجة، ثم الأحدث، ثم النصّ أبجدياً.
        # الحسم بالنصّ يمنع أي عشوائية في التعادل التام.
        scored.sort(key=lambda p: (-p.score, -p.turn_number, p.symptom_text))
        return scored

    def primary(
        self,
        symptoms: Sequence[Any],
        raw_messages: Sequence[str] = (),
        risk: Optional[RedFlagAssessment] = None,
    ) -> Optional[str]:
        """نصّ الشكوى الرئيسية الحالية، أو None إن لا عرَض مُثبَت بعد."""
        ranked = self.rank(symptoms, raw_messages, risk)
        return ranked[0].symptom_text if ranked else None

    # ------------------------------------------------------------------
    # داخلي
    # ------------------------------------------------------------------
    def _score(
        self,
        symptom: Any,
        context: str,
        red_flag_boost: float,
        max_turn: int,
        max_acuity: float,
    ) -> SymptomPriority:
        text = getattr(symptom, "text", "") or ""
        concept, acuity = self._match_concept(text)
        turn = int(getattr(symptom, "turn_number", 0) or 0)

        # التعزيز الطارئ يُمنح للأعراض عالية الحدّة فقط: علم أحمر أُطلق عن
        # تركيب لا يعني أن كل عرَض بالجلسة صار طارئاً، وإلا لتساوت الأعراض
        # جميعاً وفقد الترتيب معناه.
        applies = red_flag_boost if acuity >= 80 else 0.0

        # C-2: recency_boost يحسم فقط بين أعراض متقاربة الحدّة فعلاً (فارق
        # acuity عن الأعلى بالجلسة ≤ RECENCY_ACUITY_TOLERANCE) — لا يجوز أن
        # يقلب شكوى أعلى حدّة بمجرّد ذكر أخرى أقلّ حدّة بدور أحدث. أعلى
        # عرَض حدّةً بالجلسة (max_acuity) فارقه عن نفسه صفر، فيبقى مؤهَّلاً
        # دائماً — القيد يستبعد فقط ما يتخلّف عنه بفارق معتبر.
        is_recent = turn >= max_turn > 0
        within_recency_tolerance = (max_acuity - acuity) <= RECENCY_ACUITY_TOLERANCE
        recency = RECENCY_MAX_BOOST if is_recent and within_recency_tolerance else 0.0

        return SymptomPriority(
            symptom_text=text,
            concept=concept,
            acuity=acuity,
            red_flag_boost=applies,
            severity_boost=self._severity_boost(text, context),
            emphasis_boost=self._emphasis_boost(context),
            recency_boost=recency,
            turn_number=turn,
        )

    def _match_concept(self, text: str) -> Tuple[Optional[str], float]:
        """مطابقة المفهوم بمرحلتين.

        ١) عبارة كاملة (``terms``) — الأدقّ، وأطول عبارة تفوز فتمنع كلمة
           عامّة من ابتلاع مفهوم أخصّ.
        ٢) رموز مجتمعة (``all_tokens``) — احتياط أمتن حين تفشل العبارة بسبب
           ترتيب الكلمات أو كلمة متوسّطة. أثبت اختبار حيّ أنّ "ألم ضاغط في
           الصدر" كان يسقط إلى الدرجة الافتراضية فيهبط تحت الصداع — وهو خطأ
           ترتيب ذو أثر سريري.
        """
        haystack = normalize_arabic(text)
        if not haystack:
            return None, self._default_acuity

        best_name: Optional[str] = None
        best_acuity = self._default_acuity
        best_len = 0

        for name, (acuity, terms, _groups) in self._concepts.items():
            for term in terms:
                if term in haystack and len(term) > best_len:
                    best_name, best_acuity, best_len = name, acuity, len(term)
                    break
        if best_name is not None:
            return best_name, best_acuity

        # لا عبارة طابقت — نجرّب الرموز المجتمعة، والأعلى حدّة يفوز عند
        # تحقّق أكثر من مفهوم (تصعيد أأمن من إسقاط).
        for name, (acuity, _terms, groups) in self._concepts.items():
            if not groups:
                continue
            if all(any(tok in haystack for tok in group) for group in groups):
                if acuity > best_acuity or best_name is None:
                    best_name, best_acuity = name, acuity
        return best_name, best_acuity

    def _severity_boost(self, text: str, context: str) -> float:
        """الشدّة تُقرأ من نصّ العرَض أو من كامل كلام المريض."""
        blob = f"{normalize_arabic(text)} {context}"
        if any(term in blob for term in self._severity_high):
            return SEVERITY_HIGH_BOOST
        if any(term in blob for term in self._severity_moderate):
            return SEVERITY_MODERATE_BOOST
        return 0.0

    def _emphasis_boost(self, context: str) -> float:
        return EMPHASIS_BOOST if any(t in context for t in self._emphasis) else 0.0

    @staticmethod
    def _red_flag_boost(risk: Optional[RedFlagAssessment]) -> float:
        if risk is None:
            return 0.0
        if risk.risk_level is RiskLevel.IMMEDIATE:
            return RED_FLAG_IMMEDIATE_BOOST
        if risk.risk_level is RiskLevel.URGENT:
            return RED_FLAG_URGENT_BOOST
        return 0.0
