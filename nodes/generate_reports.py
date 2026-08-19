"""generate_reports: the terminal node for the differential-diagnosis
branch (CLAUDE.md > Graph flow: runs after route_specialty — wired in
graph.py as route_specialty -> generate_reports -> END).

No LLM call. Every fact in both reports is already sitting in state by
the time this node runs — rag_retrieve computed match_score, diagnose
computed the ranked differential and certainty bands, route_specialty
computed the specialty — so this node's only job is FORMATTING existing
state into two Arabic registers, never generating new clinical content
or re-deriving anything already computed upstream. This is the same
"structural enforcement over LLM cooperation" pattern diagnose() and
route_specialty() already follow, taken one step further: there isn't
even a natural-language-phrasing LLM call for the patient register,
though CLAUDE.md's task description explicitly allowed one. Chosen
against, not just defaulted: an LLM call here would (a) re-read the
patient's raw conversation through a "quality"-tier call for no decision
that actually needs judgment, since every fact is already computed, (b)
risk paraphrasing a match_score into a sentence that reads as a numeric
confidence despite passing a description that says not to (safety rule
7's concern applies to phrasing, not just the schema field), and (c)
break the "patient report never contains raw jargon" test guarantee
without a code-level check. A future revision can layer LLM-polished
phrasing on top of these same code-determined facts if the tests below
prove too rigid stylistically — nothing here forecloses that.

--- Reasoning trail: state ALREADY carries enough to reconstruct "what
was asked, what was answered" — no new tracking field was needed. Per
CLAUDE.md's task description, this was checked before writing anything:
state["messages"] (state.py) uses `Annotated[list[dict], operator.add]`,
an accumulating reducer, not last-value-wins — every turn's messages
persist, not just the current turn's. Given the graph's own turn
mechanics (CLAUDE.md > Graph flow: ask_followup ends a turn with exactly
one assistant message; the next invoke() supplies exactly one new user
message, per api.contracts.ChatRequest.message being "the patient's new
message this turn only"), state["messages"] by the time this node runs
is already an alternating [user, assistant, user, assistant, ..., user]
transcript: index 0 is the opening complaint (no preceding question),
and every subsequent (assistant, user) pair is one follow-up question and
its answer. _reasoning_trail() below just walks that list and pairs them
up — nothing new to track, nothing that could drift out of sync with the
real conversation the way a separately-maintained log could.

--- Patient register (Syrian colloquial, CLAUDE.md > Language, safety
rule 1: "ranked possibilities... explicit uncertainty", never a
definitive diagnosis):
  - Every candidate in the differential is listed (not just the top
    one) — safety rule 1 says "possibilities", plural. route_specialty's
    top-candidate-only choice is about picking ONE specialty to route
    booking to; presenting the differential to the patient is a
    different question, and narrowing it to one disease name would be
    the "definitive diagnosis" safety rule 1 forbids.
  - Never match_score, never a certainty word that reads as a
    percentage — only the pre-existing qualitative certainty band
    (high/medium/low, already computed by diagnose(), never re-derived
    here) rendered as a plain-language phrase. Guarded by a test that
    checks the literal match_score value never appears as a substring of
    the patient report.
  - Always ends with an explicit "this does not replace seeing a doctor"
    statement (safety rule 1) plus the recommended specialty.
  - information_limited (CLAUDE.md > State) gets a plain add-on note,
    not a silently-omitted caveat.
  - Disease names: `entry["name_ar"]` only, never `entry["name"]`.
    RESOLVED GAP: an earlier version of this node used the raw `name`
    field here, which rag/schema.py's own description marks explicit
    ("Disease name, as given by the source — not translated") — meaning
    an English/Latin term (e.g. "Migraine") would have appeared inside an
    otherwise Syrian-Arabic sentence, the first place in the project this
    would have happened. Fixed by adding rag.schema.KnowledgeBaseEntry.
    name_ar — a separately AUTHORED (not translated/derived) field,
    carried through candidate_diseases (nodes.rag_retrieve) into
    diagnosis["differential"] (nodes.diagnose) the same way specialties
    already was. Authored rather than translated on purpose: a name_ar
    is not always a literal translation of `name` — e.g. conjunctivitis's
    clinically-accurate Arabic term is not what a patient actually says
    (the same finding that drove measles.json's symptom-wording fix
    earlier this project), so name_ar had to go through the same
    per-entry human review as symptom wording, not a mechanical mapping.
  - This is also what gets appended to state["messages"] as this turn's
    assistant reply (see generate_reports() below) — api.contracts.
    ChatResponse.reply is documented as "taken from the last
    assistant-role entry appended to state['messages']", and every other
    terminal node (crisis_node, emergency_node, ask_followup) already
    appends one; a diagnosis-stage turn that appended nothing would leave
    Laravel with no reply to show the patient at all. reports["patient"]
    and the appended message are deliberately the same string, not two
    separately-authored texts — CLAUDE.md's task description didn't ask
    for a distinct short "reply" vs. long "report", and inventing that
    split here would be scope beyond what was requested.

--- Doctor register (Arabic, standard medical terminology, CLAUDE.md >
Language): full differential surfaced from diagnosis["differential"]
AS-IS (match_score, matched/missing/negated symptoms, specialties, per
candidate — this is exactly what rag_retrieve/diagnose already computed,
never regenerated here), plus the full accumulated state["symptoms"] /
state["negated_symptoms"] / state["unmatched_mentions"] (CLAUDE.md >
State: unmatched_mentions "must be surfaced in the doctor report" — this
is that surfacing), the reasoning trail described above,
information_limited flagged explicitly when true, and
state["red_flags"] surfaced (doctor-facing only, same as
check_red_flags's own "reason" field is documented doctor-facing, never
patient-facing) IF non-empty. CLAUDE.md's own task description flags
this as "unlikely on this path but check" — check_red_flags only routes
past itself to assess_sufficiency (and everything after it, including
this node) when it found nothing this turn (CLAUDE.md > Graph flow:
"routes to emergency_node when it set red_flags"), and red_flags has no
reducer (state.py: recomputed fresh each call, not merged with a stale
prior turn's value), so in practice this is always empty by the time
generate_reports runs. Read and rendered defensively anyway rather than
assumed impossible — CLAUDE.md's own reasoning for red_flags being
recomputed fresh, not accumulated, is that a red flag from a PRIOR turn
that no longer applies should not silently persist; the flip side is
this node must not crash if some future graph change ever did leave one
sitting in state at this point.

--- insufficient_information: both registers say so honestly rather than
fabricating content (CLAUDE.md > Non-negotiable safety rule 6:
insufficient_information is a valid, expected output, not an error to
paper over).

--- ml_corroboration (CLAUDE.md's XGBoost corroboration-signal section):
doctor register only, never the patient register — same reasoning as
red_flags' "reason" field above. Rendered as a fixed label plus a fixed
disclaimer (_ML_CORROBORATION_LABEL / _ML_CORROBORATION_DISCLAIMER), only
when nodes.diagnose carried a differential entry's ml_corroboration
through from nodes.ml_corroborate. No number, no percentage, never the
word "تشخيص" — this is the one and only place that field is ever turned
into text anywhere in this project.
"""

from __future__ import annotations

from typing import Any

from nodes.route_specialty import GENERAL_PRACTICE
from state import HealixState, Symptom

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"

# Certainty bands are computed in code by diagnose() (safety rule 7) —
# this is only a plain-language rendering of an already-computed word,
# never a number and never re-derived from match_score itself.
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

# Doctor-only, per the explicit design in CLAUDE.md's XGBoost
# corroboration-signal section: rendered ONLY when a differential entry
# carries ml_corroboration (nodes.diagnose carries this through from
# nodes.ml_corroborate, itself gated on a feature-density floor and
# rank-agreement with the model's own top pick — never a probability).
# Never "تشخيص" (diagnosis) — always "إشارة"/"دعم إحصائي إضافي"
# (additional signal / additional statistical support), and never a
# number: this line and its fixed disclaimer are the ONLY place this
# signal is ever rendered into text anywhere in this project.
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
    """Reconstruct the Q&A sequence from state["messages"] — see module
    docstring for why this needs no new tracking field."""
    if not messages:
        return _NO_ENTRIES_PLACEHOLDER

    lines = [f"- الشكوى الأولية: {messages[0].get('content', '')}"]
    index = 1
    while index < len(messages):
        entry = messages[index]
        if entry.get("role") != "assistant":
            # Not the alternating shape this node expects (see module
            # docstring) — skip rather than mispair, defensive only.
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
        # name_ar only — never entry["name"] (English/Latin, doctor
        # register only). See module docstring's "Disease names" note.
        lines.append(f"- {entry['name_ar']} ({certainty_ar})")
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
        # Doctor-facing only, same as check_red_flags's own "reason"
        # field (CLAUDE.md > State) — see module docstring for why this
        # is expected empty on this path but handled anyway.
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
    """Format the already-computed diagnosis/specialty/symptom state into
    a patient report and a doctor report. No LLM call — see module
    docstring."""
    diagnosis = state.get("diagnosis") or {
        "status": "insufficient_information",
        "differential": [],
        "reasoning": None,
    }
    # Defensive default only (see module docstring's red_flags note for
    # the same reasoning): route_specialty always sets this once wired
    # into graph.py, so this fallback is for isolated node-level calls,
    # not an expected real-graph path.
    specialty = state.get("specialty") or GENERAL_PRACTICE
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
