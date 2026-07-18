"""
Healix - Interview Output Parser
تحويل مخرجات الـ LLM (نص JSON) إلى ``InterviewDecision`` مع تحقّق صارم.

يقبل JSON فقط بالشكل:
    {"finished": false, "next_slot": "...", "question": "..."}
    {"finished": true}
ويتسامح مع أسوار الشيفرة (```json) أو نصّ محيط بسيط، لكنه يرفض أي شكل غير صالح.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from app.domain.conversation import InterviewDecision
from app.exceptions import InterviewParsingError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """استخراج أول كائن JSON من النص (يزيل الأسوار والنص المحيط)."""
    if not text or not text.strip():
        raise InterviewParsingError("مخرجات الـ LLM فارغة.")

    candidate = text.strip()

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    # الرجوع إلى أول '{' وآخر '}' في حال وجود نصّ محيط.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise InterviewParsingError("لم يُعثر على كائن JSON في مخرجات الـ LLM.")

    snippet = candidate[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise InterviewParsingError(f"JSON غير صالح: {exc}") from exc

    if not isinstance(data, dict):
        raise InterviewParsingError("المتوقَّع كائن JSON وليس نوعاً آخر.")
    return data


def parse_interview_decision(text: str) -> InterviewDecision:
    """تحليل مخرجات الـ LLM إلى قرار مقابلة مُتحقَّق منه."""
    data = _extract_json_object(text)

    if "finished" not in data or not isinstance(data["finished"], bool):
        raise InterviewParsingError("الحقل 'finished' (boolean) مطلوب.")

    if data["finished"]:
        return InterviewDecision(finished=True)

    next_slot = data.get("next_slot")
    question = data.get("question")

    if not isinstance(next_slot, str) or not next_slot.strip():
        raise InterviewParsingError("الحقل 'next_slot' مطلوب عند finished=false.")
    if not isinstance(question, str) or not question.strip():
        raise InterviewParsingError("الحقل 'question' مطلوب عند finished=false.")

    return InterviewDecision(
        finished=False,
        next_slot=next_slot.strip(),
        question=question.strip(),
    )
