"""
Healix - Mock LLM Provider
مزوّد LLM حتمي بديل، يعمل دون نموذج حقيقي (للتطوير والاختبار).

يُستدعى هذا المزوّد من ثلاثة مسارات مستقلّة، ويميّز بينها من الوسم الفريد
الموجود بتعليمات كل مسار:
- استخراج الأعراض (SymptomExtractor): وسم CONCEPT_CATALOG.
- اختيار مفاهيم الأدلة (EvidenceConceptExtractor): وسم KNOWN_CONCEPTS.
- دور المقابلة (سجل + قرار السؤال التالي، بلا أعراض): الافتراضي.

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
from app.prompts.evidence_concept_extraction_builder import LABEL_KNOWN_CONCEPTS
from app.prompts.symptom_extraction_builder import LABEL_CONCEPT_CATALOG

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

# ربط أسماء الأعراض العربية (معجم المقابلة) → مفاهيم symptom_evidence_map.
_SYMPTOM_TO_CONCEPT: Dict[str, str] = {
    "حرارة": "fever",
    "صداع": "headache",
    "سعال": "cough",
    "ألم بطن": "abdominal_pain",
    "ألم صدر": "chest_pain",
    "ضيق تنفس": "dyspnea",
    "غثيان": "nausea",
    "إسهال": "diarrhea",
    "دوخة": "dizziness",
    "ألم حلق": "sore_throat",
}

# صيغ النفي التي تسبق العرَض مباشرةً (نافذة قصيرة لتفادي النفي البعيد الخاطئ).
_NEGATION_HINTS = ("لا ", "ما في", "مافي", "بدون", "ليس", "بلا", "من غير")
_NEGATION_WINDOW = 12


class MockLLMProvider:
    """مزوّد وهمي حتمي: استخراج بالكلمات المفتاحية + سؤال من المُخطِّط السريري."""

    name = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        if LABEL_KNOWN_CONCEPTS in user_prompt:
            return self._evidence_concept_completion(user_prompt)
        if LABEL_CONCEPT_CATALOG in user_prompt:
            return self._symptom_extraction_completion(user_prompt)

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

    def _symptom_extraction_completion(self, user_prompt: str) -> Completion:
        """عقد استخراج الأعراض المستقلّ (SymptomExtractionPromptBuilder)."""
        patient_messages = self._grab(user_prompt, LABEL_PATIENT_MESSAGES) or []
        context_text = " ".join(str(message) for message in patient_messages)

        symptoms = []
        for name, negated in self._detect(context_text):
            concept = _SYMPTOM_TO_CONCEPT.get(name)
            if not concept:
                continue
            symptoms.append({
                "text": name,
                "concept": concept,
                "negated": negated,
                "confidence": 0.75,
                "evidence": None,
            })

        payload = {"symptoms": symptoms}
        return Completion(json.dumps(payload, ensure_ascii=False), model="mock")

    def _evidence_concept_completion(self, user_prompt: str) -> Completion:
        """عقد اختيار مفاهيم الأدلة — enum مغلق من symptom_evidence_map."""
        known_concepts = self._grab(user_prompt, LABEL_KNOWN_CONCEPTS) or []
        patient_messages = self._grab(user_prompt, LABEL_PATIENT_MESSAGES) or []
        context_text = " ".join(str(message) for message in patient_messages)

        selected: set[str] = set()
        for name, negated in self._detect(context_text):
            if negated:
                continue
            concept = _SYMPTOM_TO_CONCEPT.get(name)
            if concept and concept in known_concepts:
                selected.add(concept)

        payload = {"concepts": sorted(selected)}
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
