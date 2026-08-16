"""extract_symptoms: reads the patient's latest message for confirmed and
negated symptoms (CLAUDE.md > Graph flow: runs after crisis_check, then
merges into accumulated state across turns).

Also returns unmatched_mentions: symptom-like phrases the model
recognized but couldn't map to vocabulary/symptoms.py, which otherwise
have nowhere to go — schemas.symptoms.ExtractedSymptom/NegatedSymptom
both require a valid canonical `name`, so a real symptom outside that
list would silently vanish without this field. Observed directly: given
"طنين بالأذن" (tinnitus, not yet in the vocabulary), the model correctly
declined to force a near-match name onto it, and the mention was lost
before this field existed.

Calls the LLM on the "quality" tier — CLAUDE.md > Non-negotiable safety
rule 12: this node reads and interprets the patient's raw Arabic message,
which is exactly what that rule scopes to quality, not just crisis/red-flag
detection.

--- Conversational context, not just the isolated latest message.
CONFIRMED BUG, fixed here: a patient answering a direct follow-up
question very often does not restate the symptom it was about ("منذ متى
بدأ هذا الصداع؟" -> "من الصبح ومستمر" — never says "صداع" again). Given
ONLY that reply with no context, extract_symptoms previously returned
completely empty output — not even unmatched_mentions — because the
model, in isolation, had no way to know what "since this morning,
ongoing" was even about. Verified directly against the real model: the
identical reply attaches correctly to the existing symptom's
duration/onset fields once it can see (a) what was just asked and (b)
what's already confirmed this conversation; with neither, it silently
dropped the information every time, which is what produced the observed
failure — the same follow-up question repeating turn after turn, because
state["symptoms"] never actually gained the detail that would have
satisfied it, until the turn_count ceiling forced a stop.

Two things now reach the prompt beyond the latest message:
  - previous_question (nodes._shared.previous_assistant_message): the
    single immediately-preceding assistant message, if any — not the
    full conversation history. One turn of context is enough to resolve
    "what is this reply answering," and is far cheaper/simpler than
    reconstructing the whole thread every call.
  - known_symptoms (state["symptoms"] itself, formatted): so the model
    can attach the reply to the RIGHT existing entry by name, not just
    recognize that SOME prior symptom is being elaborated on. This is
    option (a) of two considered — passing the full existing state
    rather than having assess_sufficiency separately tag which symptom
    each next_question targets (option (b)). (a) needs no new state
    field, and lets the model use full context the way a clinician
    reading the whole chart would, rather than trusting a single tag
    computed one node upstream. Verified reliable in testing (see
    scripts/try_full_chain_manually.py and
    tests/unit/test_nodes_extract_symptoms.py) — (b) was not needed.

Merging (state["symptoms"] / state["negated_symptoms"] persisting and
combining across turns, deduplicated by name) is entirely
state.merge_symptoms's job, wired in via HealixState's own Annotated
reducer — this node does not reimplement any of that logic. It only
returns this turn's newly extracted lists; the graph folds them in.

Malformed or out-of-vocabulary model output is not logged here either.
schemas.symptoms.SymptomExtraction's own validator already audits
within-response duplicates (log_malformed_output), and llm_client.call_llm
already audits a validation failure against the schema — both fire
automatically the moment schema=SymptomExtraction is passed below. Adding
a second logging call in this node would just duplicate that trail.

--- Negation: rule-based first, LLM second, combined with OR — the same
pattern crisis_check and check_red_flags already use, and previously
missing here despite rules/negation.py's own module docstring claiming
it (a documentation bug this wiring fixes, not just a feature gap).
rules.negation.detect_negated_symptoms runs against the RAW message
(never against the LLM's own output — the two layers must stay
independent, same reasoning as the other two dual-layer checks), and
rules.negation.merge_negations unions its result with the LLM's own
negated_symptoms names. The rule layer only ever ADDS names the LLM
missed — see _merge_negated_symptoms below for how a rule-only name is
represented (bare {"name": ...}, no raw_mention: the deterministic
layer detects that a canonical term's wording appears negated, it does
not capture free text the way the LLM does for its own entries).

Both layers' verdicts are audit-logged together via
audit.logger.log_negation_detection, same rationale as
crisis_check/check_red_flags's own dual-layer logging: measuring how
often the rule layer actually recovers something the LLM missed needs
both verdicts recorded side by side, not just the merged result.

--- Duration: vocabulary.duration.parse_duration_days was built and
tested earlier this session but never actually called from anywhere —
the same "built then wired" gap rules/negation.py had before this
session's negation fix. Wired here: _with_duration_days runs it over
each confirmed symptom's own `duration` free text and adds the result
as a NEW `duration_days` field, alongside `duration`, never replacing
it — parse_duration_days's own docstring is explicit that its output is
"for downstream code that needs a number — never for overwriting what
the patient actually said". `duration_days` is None whenever `duration`
is None/unparseable (e.g. "بيجي وبيروح", vague-timing phrasing with no
extractable count) — matching parse_duration_days's own "None means the
patient did not say, never assume zero" contract. NegatedSymptom has no
`duration` field at all (schemas/symptoms.py), so this only touches
`symptoms`, never `negated_symptoms`.
"""

from __future__ import annotations

from typing import Any

from audit.logger import log_negation_detection
from llm_client import call_llm
from nodes._shared import latest_user_message, previous_assistant_message
from prompts.base import build_prompt
from rules.crisis import normalize
from rules.negation import detect_negated_symptoms, merge_negations
from schemas.symptoms import ExtractedSymptom, NegatedSymptom, SymptomExtraction
from state import HealixState, Symptom
from vocabulary.duration import parse_duration_days

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def _format_known_symptoms(symptoms: list[Symptom]) -> str:
    if not symptoms:
        return _NO_ENTRIES_PLACEHOLDER
    lines = []
    for symptom in symptoms:
        name = symptom.get("name", "")
        details = [
            f"{field}={symptom[field]}"
            for field in ("raw_mention", "duration", "severity", "onset")
            if symptom.get(field)
        ]
        lines.append(f"- {name} ({', '.join(details)})" if details else f"- {name}")
    return "\n".join(lines)


def _with_duration_days(symptom: ExtractedSymptom) -> dict[str, Any]:
    """symptom.model_dump() plus duration_days — see module docstring's
    "Duration" section for why this is an added field, not a replacement
    of the patient's own `duration` wording.
    """
    data = symptom.model_dump()
    data["duration_days"] = parse_duration_days(symptom.duration) if symptom.duration else None
    return data


def _merge_negated_symptoms(
    llm_negated: list[NegatedSymptom], message: str, thread_id: str | None
) -> list[Symptom]:
    """OR-combine the LLM's negated_symptoms with rules.negation's own
    deterministic detection over the raw message.

    LLM entries are kept exactly as returned (name + raw_mention, the
    patient's own wording for the denial). A name the rule layer catches
    that the LLM did NOT report is appended as a bare {"name": ...} —
    there is no free-text capture on that side, only "this canonical
    term's wording appears negated in the raw message" (rules/negation.py's
    own scope). Never the other direction: an LLM-only name is never
    dropped just because the rule layer's bounded-window patterns didn't
    happen to catch it too — that is exactly the under-matching
    rules/negation.py's own docstring documents as by design.

    Audit-logs both layers' own verdicts plus the merged result — see
    module docstring's "Both layers' verdicts are audit-logged" note.
    """
    llm_entries = [symptom.model_dump() for symptom in llm_negated]
    llm_names = {normalize(entry["name"]) for entry in llm_entries}

    rule_names = detect_negated_symptoms(message)
    combined_names = merge_negations(llm_names, rule_names)

    log_negation_detection(
        thread_id=thread_id,
        rule_negated=sorted(rule_names),
        llm_negated=sorted(llm_names),
        combined=sorted(combined_names),
    )

    for name in combined_names:
        if name not in llm_names:
            llm_entries.append({"name": name})

    return llm_entries


def extract_symptoms(state: HealixState) -> dict[str, Any]:
    """Extract this turn's confirmed and negated symptoms from the patient's message."""
    message = latest_user_message(state)
    previous_question = previous_assistant_message(state)
    known_symptoms = state.get("symptoms", [])

    prompt = build_prompt(
        "extract_symptoms",
        message=message,
        previous_question=previous_question or _NO_ENTRIES_PLACEHOLDER,
        known_symptoms=_format_known_symptoms(known_symptoms),
    )
    result = call_llm(
        prompt,
        schema=SymptomExtraction,
        tier="quality",
        prompt_name="extract_symptoms",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(result, SymptomExtraction)

    return {
        "symptoms": [_with_duration_days(symptom) for symptom in result.symptoms],
        "negated_symptoms": _merge_negated_symptoms(
            result.negated_symptoms, message, state.get("thread_id")
        ),
        "unmatched_mentions": result.unmatched_mentions,
    }
