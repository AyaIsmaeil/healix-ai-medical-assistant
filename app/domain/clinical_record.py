"""
Healix - Clinical Record Domain
السجل الطبي المنظَّم المتراكم عبر أدوار المقابلة (طبقة المجال الخالصة).

لا يعتمد على FastAPI ولا على أي مزوّد LLM ولا على أي بنية تحتية — بنية
معطيات + قواعد دمج فقط، فيبقى قابلاً للاختبار بمعزل تام.

لماذا يوجد هذا الكيان
---------------------
بعد الانتقال إلى معمارية "المقابلة السريرية بالـLLM"، صار الـLLM هو المصدر
الوحيد للاستخراج المنظَّم (أعراض + شدّة + مدّة + موضع + أدوية + حساسية +
أمراض مزمنة + تاريخ عائلي). لكنّ الـLLM يرى رسالة الدور الحالي بالأساس، فقد
يُعيد حقلاً فارغاً لمعلومة ذكرها المريض في دور سابق. لذلك لا تُستبدل الحالة
أبداً بمخرجات دور واحد، بل تُدمج تراكمياً عبر ``merge`` أدناه.

قواعد الدمج (مقصودة وموثّقة)
-----------------------------
* الحقول المفردة (شكوى/شدّة/مدّة/موضع): آخر قيمة غير فارغة تفوز. المريض قد
  يُصحّح أو يُحدّث نفسه ("صار أشدّ")، والأحدث أصدق طبياً.
* حقول القوائم (أدوية/حساسية/مزمنة/عائلي): اتحاد يحفظ ترتيب أول ظهور ولا
  يفقد شيئاً — إزالة أي عنصر تحتاج قراراً طبياً صريحاً لا استنتاجاً من صمت
  الـLLM في دور لاحق.
* ``missing_fields``: **تُستبدل** ولا تُدمج — فهي لقطة لحظية لما ينقص الآن،
  ودمجها تراكمياً كان سيُبقي حقولاً أُجيب عنها فعلاً مُدرَجةً كناقصة.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, Optional


class FactSource(str, Enum):
    """من أين جاءت هذه المعلومة الطبية؟

    التمييز جوهري للسلامة: حساسية دوائية **ذكرها المريض** تختلف تماماً عن
    حساسية **استنتجها النموذج**، وبلا هذا الحقل يستحيل التفريق بينهما بعد
    دخولهما السجل. الطبقات اللاحقة (الترتيب/الفرز) يحقّ لها أن تزن كلاً
    منهما بشكل مختلف، وهذا لا يكون ممكناً إلا إذا سُجّل المصدر.
    """

    PATIENT_EXPLICIT = "patient_explicit"   # وُجد شاهد نصّي في كلام المريض
    PATIENT_IMPLIED = "patient_implied"     # مفهوم ضمناً بوضوح
    LLM_INFERRED = "llm_inferred"           # لم يُعثر على شاهد — استنتاج
    SYSTEM_DERIVED = "system_derived"       # اشتقّها النظام بقاعدة حتمية
    UNKNOWN = "unknown"                     # لم يُقيَّم المصدر بعد


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FactProvenance:
    """أثر منشأ حقيقة طبية واحدة."""

    source: FactSource = FactSource.UNKNOWN
    # نصّ المريض الحرفي الذي تستند إليه — لا إعادة صياغة، لا تلخيص.
    evidence: Optional[str] = None
    turn_number: int = 0
    confidence: Optional[float] = None
    recorded_at: str = field(default_factory=_utc_now)

    @property
    def is_patient_stated(self) -> bool:
        """هل صدرت عن المريض فعلاً (لا استنتاج نموذج)؟"""
        return self.source in (
            FactSource.PATIENT_EXPLICIT, FactSource.PATIENT_IMPLIED
        )


@dataclass
class FactRevision:
    """تغيّر قيمة حقل مفرد — يُلحَق ولا يُمحى.

    هذا هو الإصلاح المباشر لعيب "الاستبدال الصامت": القيمة الجديدة تُطبَّق
    (فالمريض قد يصحّح نفسه فعلاً: «صار أشدّ»)، لكن القديمة تبقى مسجّلة
    ويُرفع ``is_contradiction`` لتنبيه الطبقات اللاحقة والمراجع البشري.
    """

    field_name: str
    old_value: Optional[str]
    new_value: Optional[str]
    turn_number: int
    evidence: Optional[str] = None
    recorded_at: str = field(default_factory=_utc_now)

    @property
    def is_contradiction(self) -> bool:
        """تغيّر قيمة موجودة سلفاً إلى قيمة مختلفة = تناقض يستحق المراجعة."""
        return bool(self.old_value) and self.old_value != self.new_value


def _merge_list(current: List[str], incoming: Iterable[str]) -> List[str]:
    """اتحاد يحفظ الترتيب ويتجاهل الفراغات والتكرار (مع تطبيع المسافات)."""
    merged = list(current)
    seen = {item.strip() for item in merged}
    for raw in incoming or ():
        item = str(raw).strip()
        if item and item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _pick(current: Optional[str], incoming: Optional[str]) -> Optional[str]:
    """آخر قيمة غير فارغة تفوز؛ الفراغ لا يمحو معلومة سابقة."""
    if incoming is None:
        return current
    candidate = str(incoming).strip()
    return candidate if candidate else current


@dataclass
class ClinicalRecord:
    """المعلومات الطبية المنظَّمة المتراكمة لمريض واحد خلال المقابلة.

    لا يحتوي إطلاقاً على تشخيص أو احتمالات أمراض أو تخصّص أو درجة خطورة —
    تلك مسؤولية وحدات لاحقة (التنبؤ/الفرز/التوصية) خلف منافذها الخاصة.
    """

    chief_complaint: Optional[str] = None
    severity: Optional[str] = None
    # شدّة رقمية 0-10 — تُملأ فقط من خانة severity@ عبر SlotAnswerValidator
    severity_numeric: Optional[int] = None
    duration: Optional[str] = None
    body_location: Optional[str] = None
    # بيانات ديموغرافية مُتحقَّقة حتمياً (لا تخمين LLM)
    age: Optional[int] = None
    gender: Optional[str] = None
    pregnancy_possible: Optional[bool] = None
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    chronic_conditions: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    # لقطة لحظية لما ينقص الآن (تُستبدل كل دور، لا تُدمج — انظر توثيق الوحدة).
    missing_fields: List[str] = field(default_factory=list)

    # --- إثبات المصدر (إضافة غير كاسرة) ---
    # الحقول أعلاه تبقى قيماً بسيطة كما هي، فكل قارئ حالي يعمل بلا تعديل؛
    # أثر المنشأ يُخزَّن بالتوازي هنا بدل تغليف القيم في كائنات.
    provenance: Dict[str, FactProvenance] = field(default_factory=dict)
    # سجلّ التغييرات — يُلحَق ولا يُمحى (يمنع الاستبدال الصامت).
    revisions: List[FactRevision] = field(default_factory=list)

    _SCALAR_FIELDS = (
        "chief_complaint", "severity", "duration", "body_location",
    )
    _INT_FIELDS = ("age", "severity_numeric")
    _BOOL_FIELDS = ("pregnancy_possible",)
    _STR_FIELDS = ("gender",)
    _LIST_FIELDS = (
        "medications", "allergies", "chronic_conditions", "family_history",
    )

    def merge(
        self,
        incoming: "ClinicalRecord",
        turn_number: int = 0,
        provenance: Optional[Dict[str, FactProvenance]] = None,
    ) -> None:
        """دمج مخرجات دور واحد في السجل المتراكم (تعديل في المكان).

        القيمة الجديدة تُطبَّق (المريض قد يصحّح نفسه)، لكن **كل تغيير يُسجَّل**
        في ``revisions`` مع القيمة القديمة — فلم يعد أي استبدال صامتاً.
        """
        incoming_provenance = provenance or incoming.provenance or {}

        for name in self._SCALAR_FIELDS:
            current = getattr(self, name)
            new_value = _pick(current, getattr(incoming, name))
            if new_value != current:
                fact = incoming_provenance.get(name)
                self.revisions.append(FactRevision(
                    field_name=name,
                    old_value=current,
                    new_value=new_value,
                    turn_number=turn_number,
                    evidence=fact.evidence if fact else None,
                ))
                setattr(self, name, new_value)
                if fact is not None:
                    self.provenance[name] = fact

        for name in self._INT_FIELDS:
            incoming_val = getattr(incoming, name)
            if incoming_val is None:
                continue
            current = getattr(self, name)
            if incoming_val != current:
                self.revisions.append(FactRevision(
                    field_name=name,
                    old_value=str(current) if current is not None else None,
                    new_value=str(incoming_val),
                    turn_number=turn_number,
                ))
                setattr(self, name, incoming_val)

        for name in self._STR_FIELDS:
            incoming_val = getattr(incoming, name)
            if not incoming_val:
                continue
            current = getattr(self, name)
            if incoming_val != current:
                self.revisions.append(FactRevision(
                    field_name=name,
                    old_value=current,
                    new_value=incoming_val,
                    turn_number=turn_number,
                ))
                setattr(self, name, incoming_val)

        for name in self._BOOL_FIELDS:
            incoming_val = getattr(incoming, name)
            if incoming_val is None:
                continue
            current = getattr(self, name)
            if incoming_val != current:
                self.revisions.append(FactRevision(
                    field_name=name,
                    old_value=str(current) if current is not None else None,
                    new_value=str(incoming_val),
                    turn_number=turn_number,
                ))
                setattr(self, name, incoming_val)

        for name in self._LIST_FIELDS:
            setattr(self, name, _merge_list(
                getattr(self, name), getattr(incoming, name)
            ))
            fact = incoming_provenance.get(name)
            if fact is not None and name not in self.provenance:
                self.provenance[name] = fact

        # استبدال لا دمج: قائمة النواقص حالة لحظية لا تاريخ تراكمي.
        self.missing_fields = [
            str(item).strip()
            for item in (incoming.missing_fields or ())
            if str(item).strip()
        ]

    # ------------------------------------------------------------------
    # استعلامات المراجعة
    # ------------------------------------------------------------------
    @property
    def contradictions(self) -> List[FactRevision]:
        """التغييرات التي بدّلت قيمة موجودة سلفاً — تستحق مراجعة بشرية."""
        return [rev for rev in self.revisions if rev.is_contradiction]

    def unverified_fields(self) -> List[str]:
        """حقول لم يُعثر لها على شاهد في كلام المريض (استنتاج نموذج).

        الحقول التي لا أثر منشأ لها إطلاقاً تُعدّ غير مُتحقَّقة أيضاً —
        غياب الأثر ليس دليل صحّة.
        """
        out: List[str] = []
        for name in (*self._SCALAR_FIELDS, *self._LIST_FIELDS):
            value = getattr(self, name)
            if not value:
                continue
            fact = self.provenance.get(name)
            if fact is None or not fact.is_patient_stated:
                out.append(name)
        return out
