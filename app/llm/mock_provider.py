"""
Healix - Mock LLM Provider
مزوّد LLM حتمي بديل، يعمل دون نموذج حقيقي (للتطوير والاختبار).

بعد الانتقال لمعمارية LLM-first لم يعد هناك مستخرج أعراض خارجي يُغذّي التعليمات،
فصار على المزوّد الوهمي أن يُنتج **العقد الموحّد كاملاً**: استخراج + قرار.

كيف يفعل ذلك حتمياً:
- الأعراض: مطابقة كلمات مفتاحية عربية بسيطة على PATIENT_MESSAGES، مع كشف نفي
  محلي ("لا/ما في/بدون") قبل الكلمة. هذا **بديل تطويري خشن** لا يُقارن بنموذج
  حقيقي، وغرضه الوحيد إبقاء المسار قابلاً للتشغيل والاختبار بلا شبكة.
- السؤال التالي: نفس مُخطِّط المقابلة في ``domain.clinical`` المستخدَم كحارس
  مجال — فلا تكرار لمعرفة سريرية.
- بقية حقول السجل: تُعاد فارغة (null/[]) ولا تُخمَّن إطلاقاً — المزوّد الوهمي
  لا يخترع معلومات طبية.

قيد صريح: المعجم أدناه محدود عمداً بأعراض شائعة تكفي للتطوير والاختبار، وليس
مرجعاً سريرياً ولا بديلاً عن نموذج حقيقي (Qwen3) في أي بيئة إنتاج.
"""

from __future__ import annotations

import json
import re
from collections import namedtuple
from typing import Dict, List, Tuple

from app.domain import clinical
from app.domain.ports import Completion
from app.prompts.interview_builder import (
    LABEL_ASKED,
    LABEL_KNOWN_NEGATED,
    LABEL_KNOWN_SYMPTOMS,
    LABEL_PATIENT_MESSAGES,
)

_SymptomView = namedtuple("_SymptomView", ["text", "negated"])

# معجم تطويري محدود: (الاسم المعياري، الكلمات المفتاحية).
_SYMPTOM_LEXICON: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("حرارة", ("حرار", "حمى", "حمّى", "سخون")),
    ("صداع", ("صداع", "ألم رأس", "وجع رأس")),
    ("سعال", ("سعال", "كحة")),
    ("ألم بطن", ("بطن", "معدة", "مغص")),
    ("ألم صدر", ("ألم صدر", "وجع صدر", "ذبحة")),
    ("ضيق تنفس", ("ضيق تنفس", "ضيق نفس", "صعوبة تنفس")),
    ("غثيان", ("غثيان", "استفراغ", "تقيؤ", "تقيّؤ")),
    ("إسهال", ("إسهال", "اسهال")),
    ("دوخة", ("دوخة", "دوار")),
    ("ألم حلق", ("حلق", "بلعوم")),
)

# صيغ النفي التي تسبق العرَض مباشرةً (نافذة قصيرة لتفادي النفي البعيد الخاطئ).
_NEGATION_HINTS = ("لا ", "ما في", "مافي", "بدون", "ليس", "بلا", "من غير")
_NEGATION_WINDOW = 12


class MockLLMProvider:
    """مزوّد وهمي حتمي: استخراج بالكلمات المفتاحية + سؤال من المُخطِّط السريري."""

    name = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        patient_messages = self._grab(user_prompt, LABEL_PATIENT_MESSAGES) or []
        known = self._grab(user_prompt, LABEL_KNOWN_SYMPTOMS) or []
        known_negated = self._grab(user_prompt, LABEL_KNOWN_NEGATED) or []
        asked = self._grab(user_prompt, LABEL_ASKED) or []

        context_text = " ".join(str(m) for m in patient_messages)

        # الأعراض = ما تراكم سلفاً + ما يُطابَق الآن من نص المريض (بلا تكرار).
        detected: Dict[Tuple[str, bool], float] = {
            (str(t), False): 1.0 for t in known
        }
        detected.update({(str(t), True): 1.0 for t in known_negated})
        for name, negated in self._detect(context_text):
            detected.setdefault((name, negated), 0.75)

        symptoms = [
            {"text": name, "negated": negated, "confidence": conf}
            for (name, negated), conf in detected.items()
        ]

        # السؤال التالي عبر المُخطِّط السريري (نفس معرفة حارس المجال).
        views = [
            _SymptomView(text=s["text"], negated=s["negated"]) for s in symptoms
        ]
        item = clinical.next_missing(views, context_text, asked)

        payload = {
            "chief_complaint": next(
                (s["text"] for s in symptoms if not s["negated"]), None
            ),
            "symptoms": symptoms,
            # المزوّد الوهمي لا يخترع قيماً طبية — تبقى فارغة دائماً.
            "severity": None,
            "duration": None,
            "body_location": None,
            "medications": [],
            "allergies": [],
            "chronic_conditions": [],
            "family_history": [],
            "missing_fields": [] if item is None else [item[0]],
            "finished": item is None,
            "next_slot": None if item is None else item[0],
            "question": None if item is None else item[1],
        }
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

    # ------------------------------------------------------------------
    # استخراج تطويري خشن
    # ------------------------------------------------------------------
    @staticmethod
    def _detect(text: str) -> List[Tuple[str, bool]]:
        """مطابقة معجمية بسيطة مع كشف نفي محلي. ليست استخراجاً سريرياً."""
        found: List[Tuple[str, bool]] = []
        for name, keywords in _SYMPTOM_LEXICON:
            for keyword in keywords:
                index = text.find(keyword)
                if index == -1:
                    continue
                window = text[max(0, index - _NEGATION_WINDOW) : index]
                negated = any(hint in window for hint in _NEGATION_HINTS)
                found.append((name, negated))
                break
        return found

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
