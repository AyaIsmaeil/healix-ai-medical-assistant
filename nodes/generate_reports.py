"""generate_reports: terminal node for the differential-diagnosis
branch. No LLM call — every fact is already in state (match_score,
differential, certainty, specialty); this node only formats it into two
Arabic registers, never generates new clinical content.

Reasoning trail: state["messages"] is already an alternating
[user, assistant, user, ...] transcript (accumulating reducer,
ask_followup's own one-question-per-turn contract) — _reasoning_trail()
just walks and pairs it, no separate tracking needed.

Patient register (Syrian colloquial): every candidate listed, not just
the top one — a differential, not a single diagnosis. Never match_score
or a percentage-reading certainty word, only the pre-computed qualitative
band. Disease names use name_ar only (name is English/Latin, doctor-only).
Always ends with the "see a doctor" statement + specialty. This is also
what gets appended as the turn's assistant reply.

Doctor register (standard medical Arabic): full differential as-is, plus
accumulated symptoms/negated/unmatched_mentions, the reasoning trail,
information_limited when true, and red_flags if non-empty (structurally
expected empty on this path — check_red_flags only lets a turn continue
here when it found nothing — but rendered defensively rather than
assumed impossible).

ml_corroboration renders doctor-side only, as a fixed label + disclaimer,
never a number — the one place that signal is ever turned into text.

Each differential entry's `source` (the KB entry's own cited guideline,
e.g. "GINA — Global Strategy for Asthma Management") renders doctor-side
only, verbatim from rag/knowledge_base/*.json via rag_retrieve.py/
diagnose.py — never shown to the patient, never generated or paraphrased
by the LLM. See research/RAG_AUDIT.md section 4: previously captured in
the KB data and then silently dropped before reaching any output.
"""

from __future__ import annotations

from typing import Any

from nodes.route_specialty import GENERAL_PRACTICE
from state import HealixState, Symptom

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"

_CERTAINTY_AR = {
    "high": "احتمال قوي نسبيًا",
    "medium": "احتمال متوسط",
    "low": "احتمال أضعف، بس وارد",
}

_UNCERTAINTY_NOTE = (
    "بس مهم توضح لي: هاي احتمالات أولية بس، مش تشخيص نهائي — الترتيب "
    "والقناعة فيهن بتختلف، وفي شي غير مؤكد بطبيعة الحال."
)

_INFORMATION_LIMITED_NOTE_PATIENT = (
    "بس خليني كون صريح معك: المعلومات يلي توفرت كانت محدودة شوي، فالتقييم "
    "هاد أقل دقة من العادة — ممكن يكون في تفاصيل ناقصة ما وصلتنا."
)

_INSUFFICIENT_PATIENT_MESSAGE = (
    "بعد كل يلي حكيناه، ما قدرنا نكوّن صورة واضحة عن احتمال محدد لهلق. هاد "
    "مش معناه إنه الموضوع بسيط أو مو مهم — الأفضل تراجع دكتور مباشرة "
    "يقيّمك وجهًا لوجه."
)

_INSUFFICIENT_DOCTOR_NOTE = (
    "لم يتم التوصل إلى تشخيص تفريقي كافٍ استناداً إلى المعطيات المتوفرة حالياً."
)

_ML_CORROBORATION_LABEL = "إشارة دعم إحصائي إضافية من نموذج تعلم آلي مساعد (XGBoost)"
_ML_CORROBORATION_DISCLAIMER = (
    "هذه الإشارة ناتجة عن نموذج تعلم آلي مساعد ومدرّب على مجموعة بيانات محدودة "
    "وتعليمية، ولا تمثل تشخيصاً مستقلاً أو حكماً سريرياً أو نسبة انتشار حقيقية، "
    "ولا تُستخدم كبديل لتقييم الطبيب."
)


def _format_symptom(symptom: Symptom) -> str:
    name = symptom.get("name", "")
    details = [
        f"{field}={symptom[field]}"
        for field in ("raw_mention", "duration", "severity", "onset")
        if symptom.get(field)
    ]
    return f"- {name} ({', '.join(details)})" if details else f"- {name}"


def _format_symptoms(symptoms: list[Symptom]) -> str:
    if not symptoms:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(_format_symptom(symptom) for symptom in symptoms)


def _format_mentions(mentions: list[str]) -> str:
    if not mentions:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(f"- {mention}" for mention in mentions)


def _reasoning_trail(messages: list[dict[str, str]]) -> str:
    if not messages:
        return _NO_ENTRIES_PLACEHOLDER

    lines = [f"- الشكوى الأولية: {messages[0].get('content', '')}"]
    index = 1
    while index < len(messages):
        entry = messages[index]
        if entry.get("role") != "assistant":
            index += 1
            continue
        question = entry.get("content", "")
        answer = _NO_ENTRIES_PLACEHOLDER
        if index + 1 < len(messages) and messages[index + 1].get("role") == "user":
            answer = messages[index + 1].get("content", "") or _NO_ENTRIES_PLACEHOLDER
            index += 2
        else:
            index += 1
        lines.append(f"- سؤال: {question}\n  جواب: {answer}")
    return "\n".join(lines)


def _patient_differential_lines(differential: list[dict[str, Any]]) -> str:
    lines = []
    for entry in differential:
        certainty_ar = _CERTAINTY_AR.get(entry.get("certainty"), _CERTAINTY_AR["low"])
        lines.append(f"- {entry['name_ar']} ({certainty_ar})")  # name_ar only, never name
    return "\n".join(lines)


def _build_patient_report(
    diagnosis: dict[str, Any], specialty: str, information_limited: bool
) -> str:
    differential = diagnosis.get("differential") or []
    if diagnosis.get("status") != "differential" or not differential:
        body = _INSUFFICIENT_PATIENT_MESSAGE
    else:
        body = (
            "طيب، خلصت من مراجعة كل يلي حكيتلي ياه. في كم احتمال أولي "
            "حابب شاركك ياهم:\n"
            + _patient_differential_lines(differential)
            + f"\n\n{_UNCERTAINTY_NOTE}"
        )

    referral = (
        f"بحسب هاد، بنصحك تراجع اختصاص {specialty}، لأنه هو الأقدر يأكد "
        "الصورة ويحدد الخطوة الجاية. أي تقييم مني ما بيعوّض عن الفحص عند "
        "دكتور مختص."
    )
    parts = [body, referral]
    if information_limited:
        parts.append(_INFORMATION_LIMITED_NOTE_PATIENT)
    return "\n\n".join(parts)


def _format_candidate_doctor(entry: dict[str, Any]) -> str:
    matched = "، ".join(entry.get("matched_symptoms") or []) or _NO_ENTRIES_PLACEHOLDER
    missing = "، ".join(entry.get("missing_symptoms") or []) or _NO_ENTRIES_PLACEHOLDER
    negated = "، ".join(entry.get("negated_symptoms") or []) or _NO_ENTRIES_PLACEHOLDER
    specialties = "، ".join(entry.get("specialties") or []) or _NO_ENTRIES_PLACEHOLDER
    lines = [
        f"- {entry['name']} ({entry['name_ar']}) "
        f"(match_score={entry['match_score']}, certainty={entry['certainty']})",
        f"  matched: {matched}",
        f"  missing: {missing}",
        f"  negated: {negated}",
        f"  specialties: {specialties}",
        f"  المصدر: {entry.get('source') or _NO_ENTRIES_PLACEHOLDER}",
    ]
    if entry.get("ml_corroboration"):
        lines.append(f"  {_ML_CORROBORATION_LABEL}")
        lines.append(f"  {_ML_CORROBORATION_DISCLAIMER}")
    return "\n".join(lines)


def _build_doctor_report(
    *,
    diagnosis: dict[str, Any],
    specialty: str,
    symptoms: list[Symptom],
    negated_symptoms: list[Symptom],
    unmatched_mentions: list[str],
    red_flags: list[dict[str, str]],
    information_limited: bool,
    messages: list[dict[str, str]],
) -> str:
    differential = diagnosis.get("differential") or []
    lines = [f"الحالة: {diagnosis.get('status')}"]

    if diagnosis.get("status") == "differential" and differential:
        lines.append("التشخيص التفريقي (مرتب حسب match_score):")
        lines.extend(_format_candidate_doctor(entry) for entry in differential)
    else:
        lines.append(_INSUFFICIENT_DOCTOR_NOTE)

    if diagnosis.get("reasoning"):
        lines.append(f"تعليل النموذج: {diagnosis['reasoning']}")

    lines.append(f"الاختصاص الموصى به: {specialty}")

    lines.append("الأعراض المؤكدة:")
    lines.append(_format_symptoms(symptoms))

    lines.append("الأعراض المنفية:")
    lines.append(_format_symptoms(negated_symptoms))

    lines.append("عبارات غير مطابقة للمفردات المعتمدة (unmatched_mentions):")
    lines.append(_format_mentions(unmatched_mentions))

    if red_flags:
        lines.append("علامات خطر (غير متوقعة على هاد المسار، معروضة احتياطياً):")
        lines.extend(f"- {flag.get('id')}: {flag.get('reason')}" for flag in red_flags)

    if information_limited:
        lines.append(
            "تنبيه: تم بلوغ الحد الأقصى لعدد الأسئلة (information_limited=True) — "
            "الصورة السريرية قد تكون غير مكتملة."
        )

    lines.append("مسار الأسئلة والأجوبة:")
    lines.append(_reasoning_trail(messages))

    return "\n\n".join(lines)


def generate_reports(state: HealixState) -> dict[str, Any]:
    diagnosis = state.get("diagnosis") or {
        "status": "insufficient_information",
        "differential": [],
        "reasoning": None,
    }
    specialty = state.get("specialty") or GENERAL_PRACTICE  # defensive default
    information_limited = bool(state.get("information_limited"))

    patient_report = _build_patient_report(diagnosis, specialty, information_limited)
    doctor_report = _build_doctor_report(
        diagnosis=diagnosis,
        specialty=specialty,
        symptoms=state.get("symptoms", []),
        negated_symptoms=state.get("negated_symptoms", []),
        unmatched_mentions=state.get("unmatched_mentions", []),
        red_flags=state.get("red_flags", []),
        information_limited=information_limited,
        messages=state.get("messages", []),
    )

    return {
        "messages": [{"role": "assistant", "content": patient_report}],
        "reports": {"patient": patient_report, "doctor": doctor_report},
        "stage": "diagnosis",
    }
