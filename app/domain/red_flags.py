"""
Healix - Red Flags Domain
أنواع مجال كشف الحالات الطارئة (طبقة المجال الخالصة).

لا تعتمد هذه الوحدة على FastAPI ولا على أي مزوّد LLM ولا على أي بنية تحتية —
بنية معطيات + قواعد ترتيب فقط، فتبقى قابلة للاختبار بمعزل تام وبصفر استدعاء
شبكة.

حدّ المجال — ما هذه الطبقة وما ليست
------------------------------------
هذه الطبقة **لا تُشخّص**. لا تقول "هذا احتشاء" بل "هذا النمط يستوجب رعاية
عاجلة". مخرجها توصية بمستوى الاستعجال وإجراء مقترح، لا اسم مرض ولا احتمال
مرض — وهو نفس حدّ المجال المفروض على وكيل المقابلة.

لماذا التقييم حتمي لا بالـLLM
------------------------------
سلامة المريض لا يجوز أن تتوقّف على نجاح استدعاء شبكي. هذه القواعد دوال خالصة
تعمل على النصّ الخام وعلى السجل المنظَّم، فتُنتج حكماً حتى لو فشل الـLLM أو
انقطع المزوّد تماماً. الـLLM قد **يرفع** الخطورة لاحقاً (تعبير غير مباشر)
ولكنه لا يستطيع خفضها أبداً — انظر ``escalate``.

قيد معروف وموثَّق (شريحة أولى)
-------------------------------
``ClinicalRecord`` لا يحمل حالياً حقولاً للعمر أو الجنس أو الحمل، لذلك قواعد
السياق الديموغرافي (نزيف الحمل، حمى الرضيع) تُقيَّم **لفظياً من نصّ المريض
الخام** لا من سجل منظَّم. هذا أضعف من التقييم المنظَّم ويُعدّ قيداً صريحاً؛
إضافة حقول ديموغرافية منظَّمة عمل شريحة لاحقة.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class RiskLevel(str, Enum):
    """مستوى الاستعجال. ``str`` ليُسلسَل إلى JSON مباشرةً بلا تحويل."""

    NONE = "none"
    LOW = "low"
    URGENT = "urgent"
    IMMEDIATE = "immediate"

    @property
    def rank(self) -> int:
        """رتبة عددية للمقارنة — التسلسل هو ما يجعل التصعيد أحادي الاتجاه."""
        return _RISK_RANK[self]

    def __ge__(self, other: "RiskLevel") -> bool:  # type: ignore[override]
        return self.rank >= other.rank

    def __gt__(self, other: "RiskLevel") -> bool:  # type: ignore[override]
        return self.rank > other.rank


_RISK_RANK = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.URGENT: 2,
    RiskLevel.IMMEDIATE: 3,
}


def escalate(*levels: RiskLevel) -> RiskLevel:
    """أعلى مستوى بين المعطيات — **تصعيد فقط، لا تخفيض أبداً**.

    هذه الدالة هي التجسيد البرمجي لقاعدة الأمان المركزية: أي طبقة تستطيع رفع
    الخطورة، ولا طبقة تستطيع خفض ما رفعته طبقة أخرى. استُخدمت ``max`` على
    الرتبة بدل منطق شرطي حتى يستحيل تمثيل "التخفيض" أصلاً.
    """
    best = RiskLevel.NONE
    for level in levels:
        if level.rank > best.rank:
            best = level
    return best


class TriggerType(str, Enum):
    """نوع المُطلِق — يوثّق *لماذا* أطلقت القاعدة، ويظهر في التفسير."""

    SINGLE_TERM = "single_term"            # مصطلح واحد كافٍ بذاته
    COMBINATION = "combination"            # تركيب مصطلحات (all_of / any_of)
    DEMOGRAPHIC_CONTEXT = "demographic_context"  # سياق (حمل، رضيع...) + عرَض


class DetectionLayer(str, Enum):
    """أي طبقة أطلقت القاعدة — للتدقيق ولإثبات عمل L0 عند فشل الـLLM."""

    RAW_TEXT = "raw_text"                  # L0: قبل الـLLM، على النصّ الخام
    STRUCTURED_RECORD = "structured_record"  # L2: بعد الاستخراج، على السجل


@dataclass(frozen=True)
class RedFlagRule:
    """قاعدة علم أحمر واحدة.

    التصميم يعتمد **مجموعات مصطلحات** (``term_groups``) لا مصطلحات مفردة،
    فتُعاد المجموعة الواحدة (مثل "ضيق تنفس" بكل لهجاتها) عبر عدّة قواعد بلا
    تكرار. ونفس تعريف القاعدة يُقيَّم في الطبقتين L0 و L2 — فلا يوجد تعريفان
    قد ينحرفان عن بعضهما.

    الدلالة: تُطلق القاعدة عندما تتحقّق **كل** مجموعات ``all_of`` **و**
    (إن وُجدت ``any_of``) **واحدة على الأقل** منها.
    """

    rule_id: str
    name_ar: str
    name_en: str
    risk_level: RiskLevel
    trigger_type: TriggerType
    # أسماء مجموعات مصطلحات (مفاتيح في كتالوج المجموعات)
    all_of: Tuple[str, ...] = ()
    any_of: Tuple[str, ...] = ()
    recommended_action_ar: str = ""
    recommended_action_en: str = ""
    evidence_source: str = ""
    notes: str = ""

    def is_satisfied_by(self, present_groups: set) -> bool:
        """هل تتحقّق القاعدة على مجموعة المصطلحات الحاضرة؟ (دالة خالصة)"""
        if not all(group in present_groups for group in self.all_of):
            return False
        if self.any_of and not any(group in present_groups for group in self.any_of):
            return False
        # قاعدة بلا شروط إطلاقاً لا تُطلق — حماية من قاعدة مشوّهة بالملف.
        return bool(self.all_of or self.any_of)


@dataclass(frozen=True)
class RedFlagMatch:
    """قاعدة أطلقت فعلاً، مع ما يكفي لتفسير سبب الإطلاق."""

    rule_id: str
    name_ar: str
    name_en: str
    risk_level: RiskLevel
    trigger_type: TriggerType
    layer: DetectionLayer
    matched_groups: Tuple[str, ...]
    # النصّ الشاهد كما ورد من المريض — إثبات المصدر، لا إعادة صياغة.
    evidence: Optional[str] = None
    recommended_action_ar: str = ""
    recommended_action_en: str = ""


@dataclass
class RedFlagAssessment:
    """حصيلة تقييم الأعلام الحمراء لدور واحد.

    ``risk_level`` هو حاصل ``escalate`` على كل ما أُطلق في كل الطبقات، فلا
    يمكن لطبقة متأخّرة أن تُنقص ما قرّرته طبقة سابقة.
    """

    risk_level: RiskLevel = RiskLevel.NONE
    matches: List[RedFlagMatch] = field(default_factory=list)
    # هل نُفِّذ الفحص أصلاً؟ يميّز "فُحص ولم يُوجد شيء" عن "لم يُفحص".
    # الافتراض False عمداً: كائن لم يمرّ بمقيّم لا يجوز أن يدّعي أنه فُحص،
    # وإلا بدت جلسة بلا محرّك أعلام حمراء كأنها فُحصت وخرجت سليمة.
    evaluated: bool = False
    # هل جرى التقييم بينما كان الـLLM معطّلاً؟ (يظهر بالتدقيق لا للمريض)
    degraded: bool = False

    @property
    def is_emergency(self) -> bool:
        """يُعدّ طارئاً عند URGENT فما فوق — عتبة واحدة بمكان واحد."""
        return self.risk_level.rank >= RiskLevel.URGENT.rank

    @property
    def unique_matches(self) -> List[RedFlagMatch]:
        """مطابقة واحدة لكل قاعدة — للعرض على المريض.

        ``matches`` يحتفظ بكل طبقة أطلقت القاعدة (أثر تدقيق يثبت أنّ L0 عمل
        مستقلاً عن L2)، لكن عرض التنبيه ذاته مرّتين على المريض ضجيج بلا
        معنى. تُفضَّل مطابقة الطبقة الأعلى خطورة، ثم الأسبق ترتيباً.
        """
        best: dict = {}
        for match in self.matches:
            current = best.get(match.rule_id)
            if current is None or match.risk_level.rank > current.risk_level.rank:
                best[match.rule_id] = match
        return list(best.values())

    @property
    def primary_action_ar(self) -> Optional[str]:
        """إجراء أعلى قاعدة خطورة أُطلقت (لا دمج نصوص، لا اختلاق)."""
        if not self.matches:
            return None
        top = max(self.matches, key=lambda m: m.risk_level.rank)
        return top.recommended_action_ar or None

    def merged_with(self, other: "RedFlagAssessment") -> "RedFlagAssessment":
        """دمج تقييمي طبقتين — تصعيد فقط، وتوحيد المطابقات بلا تكرار."""
        seen = {(m.rule_id, m.layer) for m in self.matches}
        matches = list(self.matches)
        for match in other.matches:
            if (match.rule_id, match.layer) not in seen:
                matches.append(match)
                seen.add((match.rule_id, match.layer))
        return RedFlagAssessment(
            risk_level=escalate(self.risk_level, other.risk_level),
            matches=matches,
            evaluated=self.evaluated or other.evaluated,
            degraded=self.degraded or other.degraded,
        )
