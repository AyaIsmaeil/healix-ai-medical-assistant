"""
Healix - Symptom Evidence Encoder (ربط مفاهيم الأدلة المُستخلَصة برموز DDXPlus)

يسدّ فجوة موثَّقة بـ``domain.ml_disease_predictor``: مخطّط الترميز النشط
(v1.json) لا يُصدِّر أي عمود ``E_*``، فالنموذج المدرَّب على 225 عمود DDXPlus
كان يستقبل دائماً متجهاً فارغاً من الأدلة مهما قالت المقابلة.

قرار تصميم (بُني على بحث، لا اختراع): الأنظمة الصناعية (Infermedica، Ada)
تتجنّب استخراج نص حرّ ثم مطابقته بأثر رجعي — تعتمد بدلاً منه استنباطاً من
مجموعة مرشَّحة مغلقة، وDDXPlus نفسه مصمَّم لأسئلة مغلقة الإجابة (223 دليل
Yes/No أو متعدّد الخيارات)، لا سرداً حرّاً. لذلك اختيار **أيّ مفاهيم الأدلة
تنطبق** مسؤولية ``services.evidence_concept_extractor.EvidenceConceptExtractor``
(نموذج LLM مُقيَّد بمخطّط JSON صارم = enum مغلق، لا توليد حرّ) — وهذا الصنف
هنا مسؤوليته محصورة بخطوة **حتمية بحتة تالية**: تحويل مفاهيم مُختارة سلفاً
(نصوص مفاهيم معروفة، لا نص مريض خام) إلى مفاتيح ``E_*`` ثنائية. لا مطابقة
نصّية هنا إطلاقاً — تلك مسؤولية الـLLM المُقيَّد، لا هذا الصنف.

منطق مجال خالص — بلا I/O، بلا شبكة. القاموس (مفهوم → رموز أدلة) يصله محقوناً
بالمُنشئ من ``infrastructure.dictionary_loader`` (نفس مبدأ حقن ``RedFlagEngine``).

⚠️ تغطية جزئية معلنة (انظر توثيق ``app/dictionaries/symptom_evidence_map.yaml``):
أعراض شائعة فقط، والألم (صداع/صدر/بطن) يُنتج نفس رمزَي الدليل حالياً — قيد
بمخطّط DDXPlus نفسه (الموقع قيمة فرعية غير مُنمذَجة بهذه النسخة)، لا خطأ ربط.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class SymptomEvidenceEncoder:
    """يحوّل مفاهيم أدلة مُختارة سلفاً (بمعرفة الـLLM) إلى مفاتيح ``E_*`` ثنائية."""

    def __init__(self, mappings: Sequence[Dict[str, Any]], version: str = "unknown") -> None:
        self._concept_to_codes: Dict[str, List[str]] = {
            entry["concept"]: list(entry["evidence_codes"]) for entry in mappings
        }
        self.version = version

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymptomEvidenceEncoder":
        """بناء المُرمِّز من بنية ``symptom_evidence_map.yaml`` المُحمَّلة."""
        return cls(mappings=data["mappings"], version=str(data.get("version", "unknown")))

    @property
    def concepts(self) -> List[str]:
        """قائمة أسماء المفاهيم المعروفة — تُستخدَم لبناء enum مخطّط JSON
        الخاص بـ``EvidenceConceptExtractor`` (مصدر حقيقة واحد، لا تكرار)."""
        return list(self._concept_to_codes.keys())

    def encode(self, concepts: Sequence[str]) -> Dict[str, int]:
        """يُعيد ``{E_NN: 1}`` لكل رمز دليل يقابل مفهوماً بالقائمة المُمرَّرة.

        مفهوم غير معروف (خارج القاموس المحقون) يُتجاهَل بصمت — الـLLM مُقيَّد
        أصلاً بـenum لا يسمح بغيرها، فهذا خطّ دفاع أخير فقط.
        """
        active_codes: Dict[str, int] = {}
        for concept in concepts:
            for code in self._concept_to_codes.get(concept, ()):
                active_codes[code] = 1
        return active_codes
