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
from typing import Iterable, List, Optional


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
    duration: Optional[str] = None
    body_location: Optional[str] = None
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    chronic_conditions: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    # لقطة لحظية لما ينقص الآن (تُستبدل كل دور، لا تُدمج — انظر توثيق الوحدة).
    missing_fields: List[str] = field(default_factory=list)

    def merge(self, incoming: "ClinicalRecord") -> None:
        """دمج مخرجات دور واحد في السجل المتراكم (تعديل في المكان)."""
        self.chief_complaint = _pick(self.chief_complaint, incoming.chief_complaint)
        self.severity = _pick(self.severity, incoming.severity)
        self.duration = _pick(self.duration, incoming.duration)
        self.body_location = _pick(self.body_location, incoming.body_location)

        self.medications = _merge_list(self.medications, incoming.medications)
        self.allergies = _merge_list(self.allergies, incoming.allergies)
        self.chronic_conditions = _merge_list(
            self.chronic_conditions, incoming.chronic_conditions
        )
        self.family_history = _merge_list(self.family_history, incoming.family_history)

        # استبدال لا دمج: قائمة النواقص حالة لحظية لا تاريخ تراكمي.
        self.missing_fields = [
            str(item).strip()
            for item in (incoming.missing_fields or ())
            if str(item).strip()
        ]
