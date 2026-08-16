"""check_red_flags: combines the deterministic rule engine with an LLM
second opinion over the patient's confirmed symptoms (CLAUDE.md > Graph
flow: runs after extract_symptoms).

Per CLAUDE.md > Non-negotiable safety rule 3: red-flag detection is
rule-based first, LLM second, combined with OR. The deterministic layer
(rules.red_flags.check_red_flags) is never removed, weakened, or made
conditional on the LLM result — this node adds an LLM layer alongside it,
it does not replace it.

Calls the LLM on the "quality" tier — CLAUDE.md > Non-negotiable safety
rule 12: this reads the patient's confirmed symptoms (derived from their
raw Arabic message) to decide something safety-critical.

The two layers deliberately see different inputs. The rule engine can
only ever match against vocabulary/symptoms.py's canonical names, so it
gets exactly what rules.red_flags.check_red_flags accepts:
state["symptoms"] and state["medical_record_summary"] (the
chronic-condition context some rules lower their threshold for). The LLM
additionally gets state["unmatched_mentions"]: a symptom-like phrase that
didn't map to a canonical name
(schemas.symptoms.SymptomExtraction.unmatched_mentions) can still be
clinically significant, and the rule engine has structurally no way to
see it.

Either layer firing populates state["red_flags"]: a list of
state.RedFlag ({"id": ..., "reason": ...}) entries, one per rule match
plus (if it fired) one more for the LLM layer, id="llm". Built as a
single list comprehension per source, each entry paired at construction
— never as two separately-built, same-length-assumed lists a caller
would have to zip. That matters here specifically because the two
sources are NOT symmetric: the LLM's entry has no rule_id, so a
parallel "reasons" list would have been one element shorter than
red_flags whenever the LLM fires, a silent off-by-one waiting to bite
whoever indexed the two together.

This is a fresh union computed from the CURRENT accumulated
state["symptoms"] each turn, not merged with a previous turn's list —
state.py has no reducer for this field, on purpose: symptoms only grow
across turns (never removed), so recomputing the full rule set against
the full accumulated symptom set each time is already at least as
complete as any earlier turn's result, and a plain replace keeps that
recomputation authoritative rather than carrying forward a stale flag
alongside it.

Both layers' verdicts are audit-logged together via
audit.logger.log_red_flag_detection, separately from the LLM call's own
audit record (which llm_client.call_llm already writes on every call) —
same rationale as crisis_check: measuring how often each layer catches
something the other missed needs both verdicts recorded side by side,
not just the combined result.
"""

from __future__ import annotations

from typing import Any

from audit.logger import log_red_flag_detection
from llm_client import call_llm
from prompts.base import build_prompt
from rules.red_flags import check_red_flags as run_red_flag_rules
from schemas.red_flags import RedFlagAssessment
from state import HealixState, RedFlag, Symptom

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def _format_symptom_names(symptoms: list[Symptom]) -> str:
    names = [name for symptom in symptoms if (name := symptom.get("name"))]
    if not names:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(f"- {name}" for name in names)


def _format_mentions(mentions: list[str]) -> str:
    if not mentions:
        return _NO_ENTRIES_PLACEHOLDER
    return "\n".join(f"- {mention}" for mention in mentions)


def check_red_flags(state: HealixState) -> dict[str, Any]:
    """Run both red-flag detectors over this turn's accumulated symptoms and OR them."""
    symptoms = state.get("symptoms", [])
    medical_record_summary = state.get("medical_record_summary", "")
    unmatched_mentions = state.get("unmatched_mentions", [])

    rule_matches = run_red_flag_rules(symptoms, medical_record_summary)

    prompt = build_prompt(
        "check_red_flags",
        symptoms=_format_symptom_names(symptoms),
        unmatched_mentions=_format_mentions(unmatched_mentions),
    )
    llm_result = call_llm(
        prompt,
        schema=RedFlagAssessment,
        tier="quality",
        prompt_name="check_red_flags",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(llm_result, RedFlagAssessment)

    rule_ids = [match.rule_id for match in rule_matches]
    red_flags: list[RedFlag] = [
        {"id": match.rule_id, "reason": match.reason_ar} for match in rule_matches
    ]
    if llm_result.has_red_flag:
        red_flags.append({"id": "llm", "reason": llm_result.reasoning or "unspecified"})

    log_red_flag_detection(
        thread_id=state.get("thread_id"),
        rule_matched=bool(rule_matches),
        rule_ids=rule_ids,
        llm_matched=llm_result.has_red_flag,
        llm_reasoning=llm_result.reasoning,
        combined=bool(red_flags),
    )

    return {"red_flags": red_flags}
