"""
Healix - Mock LLM Provider
مزوّد LLM حتمي بديل، يعمل دون نموذج حقيقي (للتطوير والاختبار).

يقرأ الأعراض الموسومة (EXTRACTED/NEGATED_SYMPTOMS) من تعليمات الدور، ثمّ يستدعي مُخطِّط المقابلة في
``domain.clinical`` لاختيار الخانة التالية الأعلى قيمة غير المُغطّاة — فيُظهر
التساؤل الديناميكي حسب الأعراض (حرارة → درجة الحرارة/القشعريرة...، ألم بطن →
المكان/الانتقال/التقيّؤ...). عند اكتمال جمع التاريخ يُعيد finished=true.

هذا ليس استخراجاً بقواعد للأعراض؛ بل بديل حتمي لمنطق اختيار السؤال في الـ LLM
إلى أن يُوصَل نموذج حقيقي (مثل Qwen3)، ويعيد استخدام نفس المعرفة السريرية بلا تكرار.
"""

from __future__ import annotations

import json
import re
from collections import namedtuple

from app.domain import clinical
from app.domain.ports import Completion
from app.prompts.interview_builder import (
    LABEL_ASKED,
    LABEL_EXTRACTED,
    LABEL_NEGATED,
)

_SymptomView = namedtuple("_SymptomView", ["text", "negated"])


class MockLLMProvider:
    """مزوّد وهمي حتمي يختار السؤال التالي عبر المُخطِّط السريري."""

    name = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        extracted = self._grab(user_prompt, LABEL_EXTRACTED) or []
        negated = self._grab(user_prompt, LABEL_NEGATED) or []
        asked = self._grab(user_prompt, LABEL_ASKED) or []

        # الأعراض تأتي من MARBERT عبر الحقلين الموسومين (لا من الرسالة الخام).
        symptoms = [_SymptomView(text=str(t), negated=False) for t in extracted]
        symptoms += [_SymptomView(text=str(t), negated=True) for t in negated]

        # التغطية للتكرار = الخانات المطروحة فقط (لا خريطة خانة→قيمة).
        item = clinical.next_missing(symptoms, {}, asked)
        if item is None:
            return Completion(json.dumps({"finished": True}), model="mock")

        target, question = item
        payload = {"finished": False, "next_slot": target, "question": question}
        return Completion(json.dumps(payload, ensure_ascii=False), model="mock")

    def health(self) -> dict:
        """فحص جاهزية منظَّم (المزوّد الوهمي جاهز دائماً — بلا شبكة)."""
        return {
            "provider": self.name,
            "model": "mock",
            "api_key_configured": True,
            "reachable": True,
            "model_accessible": True,
            "ok": True,
            "error": None,
        }

    @staticmethod
    def _grab(user_prompt: str, label: str):
        """قراءة قيمة JSON لسطر موسوم ``LABEL: <json>`` من تعليمات الدور."""
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", user_prompt, re.MULTILINE)
        if not match:
            return None
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
