"""
Healix - Red Flag Engine
محرّك كشف الحالات الطارئة — **حتمي بالكامل، بصفر استدعاء شبكة**.

لماذا حتمي
-----------
سلامة المريض لا يجوز أن تتوقّف على نجاح استدعاء LLM. هذا المحرّك دوال خالصة
تعمل على النصّ الخام وعلى السجل المنظَّم، فيُنتج حكماً حتى لو فشل المزوّد
تماماً أو انقطعت الشبكة. هذه هي الخاصية المعمارية المركزية للطبقة كلّها.

الطبقتان
---------
* ``evaluate_raw_text``  (L0) — يعمل على رسائل المريض الخام **قبل** استدعاء
  الـLLM. يلتقط سياق الحمل/الرضيع الذي لا يحمله السجل المنظَّم بعد.
* ``evaluate_record``    (L2) — يعمل على الأعراض المستخرَجة والسجل المنظَّم
  **بعد** الاستخراج. أدقّ في النفي لأنّه يقرأ ``negated`` المنظَّم بدل
  استنتاجه لفظياً.

تُدمج النتيجتان بـ``RedFlagAssessment.merged_with`` الذي يصعّد ولا يخفّض.

تراكم عبر الأدوار (قرار مقصود)
--------------------------------
L0 يُقيَّم على **كل** رسائل المريض لا على الرسالة الأخيرة وحدها: المريض قد
يقول "أنا حامل" في دور، و"عندي نزيف" في دور لاحق — وتقييم الرسالة الأخيرة
منفردة كان سيُفوّت التركيب عبر الأدوار تماماً.

النفي (قيد صريح)
-----------------
L0 يكتشف النفي **لفظياً** بنافذة قصيرة قبل المصطلح ("ما في ألم صدر").
هذا أفضل جهد لا ضمان: النفي البعيد أو المعقّد قد يفوته. L2 أدقّ لأنّه يقرأ
راية ``negated`` المنظَّمة. عند الشك تبقى القاعدة العامة: **التصعيد أأمن من
الإسقاط** — لذلك النفي يُطبَّق فقط عند وجود صيغة نفي صريحة قريبة.

حدّ المجال
-----------
لا تشخيص هنا إطلاقاً — المخرَج مستوى استعجال وإجراء مقترح فقط.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from app.domain.red_flags import (
    DetectionLayer,
    RedFlagAssessment,
    RedFlagMatch,
    RedFlagRule,
    RiskLevel,
    TriggerType,
    escalate,
)
# مصدر التطبيع الوحيد الآن ``domain.text_preprocessing`` (المبني على CAMeL
# Tools) — يُعاد تصديره هنا حفاظاً على واردات ``from ... red_flag_engine
# import normalize_arabic`` القائمة (clinical_priority، conversation_service،
# الاختبارات) دون تغييرها.
from app.domain.text_preprocessing import normalize_arabic

# ----------------------------------------------------------------------
# النفي
# ----------------------------------------------------------------------
# صيغ النفي التي تُبطل مطابقة تليها مباشرةً (نافذة قصيرة).
_NEGATION_CUES: Tuple[str, ...] = (
    "ما في", "مافي", "ما عندي", "ماعندي", "لا يوجد", "لايوجد",
    "بدون", "بلا", "من غير", "ليس", "ما بحس", "ما اشعر", "نفيت",
    "no ", "not ", "without ", "denies",
)
# نافذة النفي بالمحارف — قصيرة عمداً: نافذة طويلة تُسقط طوارئ حقيقية
# (مثال: "ما عندي سكري بس عندي الم صدر" — النفي يخصّ السكري لا الألم).
_NEGATION_WINDOW = 18


def _is_negated(haystack: str, index: int) -> bool:
    """هل تسبق المطابقةَ صيغةُ نفي صريحة ضمن نافذة قصيرة؟"""
    window = haystack[max(0, index - _NEGATION_WINDOW):index]
    return any(cue in window for cue in _NEGATION_CUES)


# ----------------------------------------------------------------------
# المحرّك
# ----------------------------------------------------------------------
class RedFlagEngine:
    """مقيّم حتمي لقواعد الأعلام الحمراء. عديم الحالة وآمن للمشاركة."""

    def __init__(
        self,
        term_groups: Dict[str, Sequence[str]],
        rules: Sequence[RedFlagRule],
        version: str = "unknown",
    ) -> None:
        self.version = version
        self._rules: Tuple[RedFlagRule, ...] = tuple(rules)
        # تُطبَّع المصطلحات مرّة واحدة عند البناء لا مع كل طلب.
        self._groups: Dict[str, Tuple[str, ...]] = {
            name: tuple(sorted(
                {normalize_arabic(t) for t in terms if normalize_arabic(t)},
                key=len, reverse=True,   # الأطول أولاً: مطابقة أدقّ
            ))
            for name, terms in term_groups.items()
        }

    # ------------------------------------------------------------------
    # البناء من بيانات الملف
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedFlagEngine":
        """بناء المحرّك من بنية ``red_flags.yaml`` المُحمَّلة."""
        rules = [
            RedFlagRule(
                rule_id=raw["rule_id"],
                name_ar=raw["name_ar"],
                name_en=raw["name_en"],
                risk_level=RiskLevel(raw["risk_level"]),
                trigger_type=TriggerType(raw["trigger_type"]),
                all_of=tuple(raw.get("all_of") or ()),
                any_of=tuple(raw.get("any_of") or ()),
                recommended_action_ar=raw.get("recommended_action_ar", ""),
                recommended_action_en=raw.get("recommended_action_en", ""),
                evidence_source=raw.get("evidence_source", ""),
                notes=raw.get("notes", ""),
            )
            for raw in data.get("rules", ())
        ]
        return cls(
            term_groups=data.get("term_groups", {}),
            rules=rules,
            version=str(data.get("version", "unknown")),
        )

    # ------------------------------------------------------------------
    # L0 — النصّ الخام (قبل الـLLM)
    # ------------------------------------------------------------------
    def evaluate_raw_text(self, texts: Sequence[str]) -> RedFlagAssessment:
        """تقييم على رسائل المريض الخام. يعمل حتى لو تعذّر استدعاء الـLLM.

        يُقيَّم على **كل** الرسائل مجتمعةً لالتقاط التركيبات عبر الأدوار.
        """
        present, evidence_by_group = self._match_groups(texts, apply_negation=True)
        return self._fire(present, evidence_by_group, DetectionLayer.RAW_TEXT)

    # ------------------------------------------------------------------
    # L2 — السجل المنظَّم (بعد الاستخراج)
    # ------------------------------------------------------------------
    def evaluate_record(
        self,
        symptom_texts_positive: Sequence[str],
        record_texts: Sequence[str] = (),
        raw_texts: Sequence[str] = (),
    ) -> RedFlagAssessment:
        """تقييم على الأعراض المُثبَتة والسجل المنظَّم.

        ``symptom_texts_positive`` أعراض **غير منفية** فقط — النفي هنا مقروء
        من راية ``negated`` المنظَّمة، فلا حاجة لاستنتاجه لفظياً (وهذا أدقّ
        من L0).

        ``raw_texts`` يُمرَّر أيضاً لأنّ السياق الديموغرافي (حمل، رضيع) لا
        يوجد له حقل منظَّم في ``ClinicalRecord`` بعد — قيد موثَّق.
        """
        # الأعراض والسجل: بلا نفي لفظي (النفي عولج بنيوياً قبل الوصول هنا).
        present, evidence = self._match_groups(
            list(symptom_texts_positive) + list(record_texts), apply_negation=False
        )
        # السياق الديموغرافي من الخام: هنا يلزم النفي اللفظي.
        ctx_present, ctx_evidence = self._match_groups(raw_texts, apply_negation=True)
        present |= ctx_present
        for group, ev in ctx_evidence.items():
            evidence.setdefault(group, ev)

        return self._fire(present, evidence, DetectionLayer.STRUCTURED_RECORD)

    # ------------------------------------------------------------------
    # C-3 — استعلام أولوية الأسئلة (لا يُطلق أي قاعدة، لا يُنتج تقييماً)
    # ------------------------------------------------------------------
    def is_red_flag_pathway_evidence(self, text: str) -> bool:
        """هل يقدّم هذا النصّ وحده دليلاً كافياً ليكون عرَضه "على مسار"
        قاعدة علم أحمر حالية — لا مجرّد مطابقة عابرة لجزء منها؟

        استعلام قراءة فقط لأولوية أسئلة المقابلة (``domain.clinical``) —
        **لا يُطلق قاعدة ولا يُنتج ``RedFlagAssessment``**، ولا يخترع أي
        علم أحمر جديد ولا مطابقة جديدة: يعيد استخدام ``_match_groups``
        وكتالوج ``self._rules`` الحاليَين حرفياً. الفارق عن الاسم القديم
        (``is_potential_trigger``، محذوف): ذاك كان يكتفي بمطابقة **مجموعة
        واحدة فقط** من ``all_of`` قاعدةٍ قد تتطلّب أكثر من مجموعة معاً — فكان
        "صداع" وحده يُعَدّ مُطلِقاً لقاعدة "الصداع الرعدي" رغم غياب
        ``thunderclap_quality`` كلياً، وكذلك "حرارة" وحدها لقاعدة حمى الرضيع
        رغم غياب السياق الديموغرافي. هذا خطأ دلالي مُثبَت (انظر تدقيق
        Phase 0)، لا مجرّد اسم غير دقيق.

        القاعدة الصحيحة المُطبَّقة هنا لكل قاعدة علم أحمر:
          * لها ``all_of``: يجب أن يحضر هذا النصّ **كل** مجموعاته معاً (لا
            مجموعة واحدة فقط) — فيكفي هذا وحده (بصرف النظر عن ``any_of``،
            لأنّ استكمال ``any_of`` بالتحديد هو ما تفيد فيه أولوية الأسئلة).
            قاعدة بمجموعة ``all_of`` وحيدة (كألم الصدر) تتحقّق بهذا النصّ
            وحده تلقائياً — لا تغيير هنا عن السلوك القديم لهذه الحالة.
          * ``all_of`` فارغة (مثل قاعدة FAST): تُطلَق أصلاً بأيّ عضو من
            ``any_of`` وحده، فحضور عضو واحد منها بهذا النصّ كافٍ — حالة كانت
            الآلية القديمة تُغفلها كلياً لأنّها فحصت ``all_of`` فقط.
        """
        present, _ = self._match_groups([text], apply_negation=False)
        if not present:
            return False
        for rule in self._rules:
            if rule.all_of:
                if all(group in present for group in rule.all_of):
                    return True
                continue
            if rule.any_of and any(group in present for group in rule.any_of):
                return True
        return False

    # ------------------------------------------------------------------
    # داخلي
    # ------------------------------------------------------------------
    def _match_groups(
        self, texts: Iterable[str], apply_negation: bool
    ) -> Tuple[Set[str], Dict[str, str]]:
        """أي مجموعات مصطلحات حاضرة؟ مع النصّ الشاهد لكل مجموعة."""
        present: Set[str] = set()
        evidence: Dict[str, str] = {}

        normalized = [(raw, normalize_arabic(raw)) for raw in texts if raw]
        for group, terms in self._groups.items():
            for raw, hay in normalized:
                hit = False
                for term in terms:
                    start = hay.find(term)
                    while start != -1:
                        if not (apply_negation and _is_negated(hay, start)):
                            hit = True
                            break
                        start = hay.find(term, start + 1)
                    if hit:
                        break
                if hit:
                    present.add(group)
                    # الشاهد = رسالة المريض الأصلية كما وردت (لا إعادة صياغة).
                    evidence.setdefault(group, raw)
                    break
        return present, evidence

    def _fire(
        self,
        present: Set[str],
        evidence: Dict[str, str],
        layer: DetectionLayer,
    ) -> RedFlagAssessment:
        """تطبيق كل القواعد على المجموعات الحاضرة وبناء الحصيلة."""
        matches: List[RedFlagMatch] = []
        level = RiskLevel.NONE

        for rule in self._rules:
            if not rule.is_satisfied_by(present):
                continue
            matched = tuple(
                g for g in (*rule.all_of, *rule.any_of) if g in present
            )
            matches.append(RedFlagMatch(
                rule_id=rule.rule_id,
                name_ar=rule.name_ar,
                name_en=rule.name_en,
                risk_level=rule.risk_level,
                trigger_type=rule.trigger_type,
                layer=layer,
                matched_groups=matched,
                evidence=next(
                    (evidence[g] for g in matched if g in evidence), None
                ),
                recommended_action_ar=rule.recommended_action_ar,
                recommended_action_en=rule.recommended_action_en,
            ))
            level = escalate(level, rule.risk_level)

        return RedFlagAssessment(risk_level=level, matches=matches, evaluated=True)
