"""
Healix - Clinical Interview Knowledge
معرفة أخذ التاريخ المرضي (طبقة المجال الخالصة) — لا تشخيص هنا إطلاقاً.

مصدر وحيد للحقيقة يحدّد "ما الذي يجب سؤاله" في مقابلة ذكية:
- خانات OLDCARTS الأساسية لكل عرَض رئيسي (onset/duration/severity/...).
- أسئلة خاصة بأعراض بعينها (حرارة، ألم بطن، ألم صدر، صداع، سعال...).
- خانات سياقية (عمر، جنس، حمل، أمراض مزمنة، أدوية، حساسية، تدخين) عند اللزوم.

يوفّر أيضاً مُخطِّطاً (planner) يحسب "الخانة التالية الأعلى قيمة" غير المُغطّاة،
فيقود التساؤل الديناميكي. يستهلكه كلٌّ من بانِي التعليمات ومزوّد الـ LLM الوهمي
دون تكرار للمنطق.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class SymptomView(Protocol):
    """الشكل المتوقَّع لعرَض (يطابقه ``domain.conversation.Symptom``)."""

    text: str
    negated: bool


@dataclass(frozen=True)
class SlotSpec:
    """خانة معلومات وسؤالها العربي. ``{s}`` يُستبدل باسم العرَض الرئيسي."""

    slot: str
    question: str


@dataclass(frozen=True)
class SymptomRule:
    """قاعدة أسئلة خاصة بعرَض: كلمات مفتاحية → خانات إضافية."""

    keywords: Tuple[str, ...]
    slots: Tuple[SlotSpec, ...]


# ملاحظة تفريغية (target) = "<slot>@<symptom>" لخانات العرَض، و"context:<slot>" للسياق.
ChecklistItem = Tuple[str, str]  # (target, question)

# خانات OLDCARTS الأساسية للعرَض الرئيسي.
CORE_SLOTS: Tuple[SlotSpec, ...] = (
    SlotSpec("onset", "منذ متى بدأت {s}؟"),
    SlotSpec("duration", "كم تستمر {s} في كل مرة؟"),
    SlotSpec("severity", "ما شدّة {s} من 1 إلى 10؟"),
    SlotSpec("progression", "هل {s} تزداد سوءاً أم تتحسّن أم بقيت ثابتة؟"),
    SlotSpec("location", "أين تشعر بـ{s} بالتحديد؟"),
    SlotSpec("quality", "كيف تصف طبيعة {s} (نابض، حارق، ضاغط، طاعن...)؟"),
    SlotSpec("associated_symptoms", "هل توجد أعراض أخرى مصاحبة لـ{s}؟"),
    SlotSpec("aggravating", "ما الذي يزيد {s} سوءاً؟"),
    SlotSpec("relieving", "ما الذي يخفّف {s}؟"),
)

# أسئلة خاصة بأعراض محدّدة (تساؤل ديناميكي حسب العرَض).
SYMPTOM_SPECIFIC: Tuple[SymptomRule, ...] = (
    SymptomRule(
        ("حرار", "حمى", "حمّى", "سخون"),
        (
            SlotSpec("temperature", "ما أعلى درجة حرارة سجّلتها؟"),
            SlotSpec("chills", "هل ترافقها قشعريرة أو رعشة؟"),
            SlotSpec("cough", "هل يوجد سعال؟"),
            SlotSpec("sore_throat", "هل يوجد ألم أو احتقان في الحلق؟"),
        ),
    ),
    SymptomRule(
        ("بطن", "معدة", "مغص", "أمعاء"),
        (
            SlotSpec("migration", "هل انتقل الألم من مكانه الأول إلى مكان آخر؟"),
            SlotSpec("vomiting", "هل يوجد تقيّؤ؟"),
            SlotSpec("diarrhea", "هل يوجد إسهال؟"),
            SlotSpec("constipation", "هل يوجد إمساك؟"),
        ),
    ),
    SymptomRule(
        ("صدر", "ذبحة"),
        (
            SlotSpec("radiation", "هل يمتدّ الألم إلى الذراع أو الفك أو الظهر؟"),
            SlotSpec("dyspnea", "هل يترافق مع ضيق في التنفّس؟"),
            SlotSpec("sweating", "هل يترافق مع تعرّق؟"),
            SlotSpec("exertion", "هل يظهر أو يزداد مع المجهود؟"),
        ),
    ),
    SymptomRule(
        ("صداع", "رأس"),
        (
            SlotSpec("nausea", "هل يترافق مع غثيان أو تقيّؤ؟"),
            SlotSpec("photophobia", "هل يزعجك الضوء؟"),
            SlotSpec("neck_stiffness", "هل تشعر بتيبّس في الرقبة؟"),
        ),
    ),
    SymptomRule(
        ("سعال", "كحة"),
        (
            SlotSpec("sputum", "هل يترافق السعال مع بلغم؟"),
            SlotSpec("hemoptysis", "هل لاحظت دماً مع السعال؟"),
        ),
    ),
)

# خانات سياقية عامة عالية القيمة (تُطرح بعد خانات العرَض عند اللزوم).
CONTEXT_SLOTS: Tuple[SlotSpec, ...] = (
    SlotSpec("age", "كم عمرك؟"),
    SlotSpec("gender", "ما جنسك (ذكر / أنثى)؟"),
    SlotSpec("pregnancy", "هل هناك احتمال للحمل حالياً؟"),
    SlotSpec("chronic_diseases", "هل لديك أمراض مزمنة (سكري، ضغط، ربو...)؟"),
    SlotSpec("medications", "هل تتناول أي أدوية بشكل منتظم حالياً؟"),
    SlotSpec("allergies", "هل لديك أي حساسية معروفة تجاه دواء أو غيره؟"),
    SlotSpec("smoking", "هل تدخّن؟"),
)

_MALE_HINTS = ("ذكر", "رجل", "صبي", "male", "ولد")
_FEMALE_HINTS = ("أنثى", "انثى", "امرأة", "مرأة", "بنت", "سيدة", "female")


def _positive_symptoms(symptoms: Sequence[SymptomView]) -> List[SymptomView]:
    """الأعراض المُثبَتة فقط (نتجاهل المنفية — لا نأخذ تاريخ عرَض نفاه المريض)."""
    return [s for s in symptoms if not getattr(s, "negated", False)]


def _resolve_primary(
    positives: Sequence[SymptomView], primary: Optional[str]
) -> str:
    """الشكوى الرئيسية المعتمدة لبناء قائمة التساؤل.

    تُقبل ``primary`` القادمة من محرّك الأولوية **فقط** إن كانت تطابق عرَضاً
    مُثبَتاً حاضراً فعلاً: قيمة قديمة أو مُختلَقة كانت ستبني قائمة أسئلة عن
    عرَض لا وجود له في الجلسة.
    """
    if primary:
        for symptom in positives:
            if symptom.text == primary:
                return primary
    return positives[0].text


def _is_male(context_text: str) -> bool:
    """هل دلّ كلام المريض على أنه ذكر؟ (الأنثى لها الأولوية لتجنّب الالتباس)."""
    if any(hint in context_text for hint in _FEMALE_HINTS):
        return False
    return any(hint in context_text for hint in _MALE_HINTS)


def build_checklist(
    symptoms: Sequence[SymptomView],
    context_text: str = "",
    primary: Optional[str] = None,
) -> List[ChecklistItem]:
    """
    بناء قائمة التساؤل المرتّبة (target, question) للحالة الراهنة.

    الترتيب: OLDCARTS للعرَض الرئيسي → الأسئلة الخاصة بكل عرَض مُثبَت → السياق.

    ``context_text`` = كل ما قاله المريض (raw messages مدموجة). منه تُستنتج
    خانة الحمل: تُستبعد إذا دلّ كلامه على أنه ذكر. (سابقاً كانت تُقرأ من خريطة
    خانة→قيمة التي أُزيلت، فصارت لا تُستبعد أبداً — وهو سبب سؤال الذكر عن الحمل.)

    وإذا لم يُذكر أي عرَض بعد، تُعاد خانة الشكوى الرئيسية **وحدها**: لا يصحّ
    الانتقال لأسئلة العمر/الجنس قبل معرفة سبب المراجعة أصلاً.
    """
    items: List[ChecklistItem] = []
    seen: set = set()

    positives = _positive_symptoms(symptoms)

    if not positives:
        return [("chief_complaint", "هل يمكنك وصف الأعراض التي تشعر بها بالتفصيل؟")]
    else:
        # ``primary`` يُحدَّد بمحرّك الأولوية السريرية (ClinicalPriorityEngine)
        # ويُمرَّر من الأعلى. الرجوع إلى أوّل عرَض عند غيابه ليس اختياراً
        # سريرياً بل تدهور آمن للسلوك السابق حين لا يكون المحرّك محقوناً
        # (اختبارات قديمة، مسارات لا تملكه) — انظر توثيق الوحدة.
        primary = _resolve_primary(positives, primary)
        for spec in CORE_SLOTS:
            target = f"{spec.slot}@{primary}"
            if target in seen:
                continue
            seen.add(target)
            items.append((target, spec.question.format(s=primary)))

        for symptom in positives:
            for rule in SYMPTOM_SPECIFIC:
                if any(keyword in symptom.text for keyword in rule.keywords):
                    for spec in rule.slots:
                        target = f"{spec.slot}@{symptom.text}"
                        if target in seen:
                            continue
                        seen.add(target)
                        items.append((target, spec.question))

    is_male = _is_male(context_text)

    for spec in CONTEXT_SLOTS:
        if spec.slot == "pregnancy" and is_male:
            continue
        target = f"context:{spec.slot}"
        if target in seen:
            continue
        seen.add(target)
        items.append((target, spec.question))

    return items


def next_missing(
    symptoms: Sequence[SymptomView],
    context_text: str,
    asked: Sequence[str],
    primary: Optional[str] = None,
) -> Optional[ChecklistItem]:
    """أوّل خانة غير مُغطّاة في القائمة، أو None عند اكتمال جمع التاريخ."""
    checklist = build_checklist(symptoms, context_text, primary)

    # ما دام المريض لم يذكر أي عرَض، نُصرّ على الشكوى الرئيسية حتى لو سبق
    # السؤال عنها — وإلا خلت القائمة فتنتهي المقابلة بلا أي أعراض إطلاقاً.
    # (حدّ MAX_QUESTIONS في محرك المحادثة يمنع الدوران اللانهائي.)
    if not _positive_symptoms(symptoms):
        return checklist[0]

    covered = set(asked)
    for target, question in checklist:
        if target not in covered:
            return target, question
    return None


def missing_targets(
    symptoms: Sequence[SymptomView],
    context_text: str,
    asked: Sequence[str],
    limit: Optional[int] = None,
    primary: Optional[str] = None,
) -> List[str]:
    """قائمة الخانات المفقودة عالية القيمة (لإرشاد الـ LLM)."""
    checklist = build_checklist(symptoms, context_text, primary)

    # بلا أعراض: تبقى الشكوى الرئيسية هي المطلوب الوحيد (ولو سُئلت سابقاً).
    if not _positive_symptoms(symptoms):
        return [checklist[0][0]]

    covered = set(asked)
    missing = [t for t, _ in checklist if t not in covered]
    return missing[:limit] if limit else missing
