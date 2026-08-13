"""
Healix — Interview Triage Short-Circuit
إنهاء مبكر للمقابلة عند اكتمال البيانات الحرجة — مبنيّ على مسارات triage
موثّقة، لا على عدّ الأسئلة وحده.

مراجع:
- NICE CG95 (chest pain assessment) — ألم صدر + أعراض مصاحبة → تقييم عاجل
- ESC Guidelines for ACS — ألم صدر مع ضيق تنفس/راحة = مسار عاجل
- Healix red_flags.yaml — pathway ``acs_pattern`` (chest_pain + dyspnea/...)
"""

from __future__ import annotations

from typing import List, Sequence

from app.domain.conversation import ConversationState
from app.domain.text_preprocessing import normalize_arabic

_MIN_QUESTIONS_BEFORE_SHORT_CIRCUIT = 6

_CHEST_HINTS = ("صدر", "ضغط", "ذبحة", "chest")
_DYSPNEA_HINTS = ("ضيق نفس", "ضيق تنفس", "صعوبة تنفس", "dyspnea")
_SEVERE_HINTS = ("شديد", "8", "9", "10", "7")


def _positive_texts(state: ConversationState) -> List[str]:
    return [s.text for s in state.symptoms if not s.negated]


def _context_text(state: ConversationState) -> str:
    return normalize_arabic(" ".join(state.raw_messages))


def _has_chest_symptom(texts: Sequence[str]) -> bool:
    for text in texts:
        normalized = normalize_arabic(text)
        if any(h in normalized for h in _CHEST_HINTS):
            return True
    return False


def _has_dyspnea(texts: Sequence[str], context: str) -> bool:
    combined = " ".join(normalize_arabic(t) for t in texts) + " " + context
    return any(h in combined for h in _DYSPNEA_HINTS)


def _has_high_severity(state: ConversationState) -> bool:
    if state.record.severity_numeric is not None:
        return state.record.severity_numeric >= 7
    severity = state.record.severity or ""
    normalized = normalize_arabic(severity)
    return any(h in normalized for h in _SEVERE_HINTS)


def _core_chest_slots_covered(asked_slots: Sequence[str]) -> bool:
    """هل غُطّت خانات OLDCARTS + صدر نوعية أساسية؟"""
    asked = set(asked_slots)
    has_severity = any(s.startswith("severity@") for s in asked)
    has_dyspnea_slot = any(s.startswith("dyspnea@") for s in asked)
    has_exertion_or_relief = any(
        s.startswith(prefix) for s in asked
        for prefix in ("exertion@", "relieving@", "aggravating@")
    )
    has_demographics = "context:age" in asked and "context:gender" in asked
    return has_severity and has_dyspnea_slot and has_exertion_or_relief and has_demographics


def should_short_circuit_interview(state: ConversationState) -> bool:
    """هل اكتمل جمع البيانات الحرجة لمسار ألم صدر عاجل؟

    يُفعَّل فقط عند:
    - ألم/ضغط صدر + ضيق تنفس (نمط ACS/NICE)
    - شدّة ≥ 7
    - ≥ 6 أسئلة + خانات أساسية مُغطّاة
    - العمر والجنس مُسجَّلان في السجل (لا تخمين)
    """
    if state.risk.is_emergency:
        return True

    if len(state.asked_questions) < _MIN_QUESTIONS_BEFORE_SHORT_CIRCUIT:
        return False

    positives = _positive_texts(state)
    context = _context_text(state)

    if not _has_chest_symptom(positives):
        return False
    if not _has_dyspnea(positives, context):
        return False
    if not _has_high_severity(state):
        return False
    if state.record.age is None or state.record.gender is None:
        return False
    if not _core_chest_slots_covered(state.asked_slots):
        return False

    return True
