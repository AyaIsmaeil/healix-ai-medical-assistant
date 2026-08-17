"""crisis_check: the first node in the graph (CLAUDE.md > Graph flow).

Combines two independent detectors of acute psychological distress or
self-harm signal with OR, per CLAUDE.md > Non-negotiable safety rule 3:

  - rules.crisis.detect_crisis — the deterministic layer, matched against
    the raw patient message ALONE, exactly as extract_symptoms/rules
    receive it. Never removed, weakened, or made conditional on the LLM
    result, and never given the extra context below either — widening
    what this layer matches against is a separate, deliberate decision
    this change does not make.
  - an LLM call against prompts/templates/crisis_check.txt, constrained to
    schemas.crisis.CrisisCheckResult, on the "quality" tier — crisis
    detection is the least acceptable place for a weaker model.

Either one firing sets is_crisis. This node only decides the flag itself;
routing a crisis straight to the terminal crisis_node, bypassing
extract_symptoms and RAG entirely (safety rule 4), is graph.py's job, not
this one's — that is why this file has no branching logic in it.

--- One turn of conversational context, LLM layer only. Same fix as
nodes/extract_symptoms.py, applied here after that one was verified: a
terse or ambiguous reply ("أيوة", "مش مهم") can mean something different
depending on what was just asked, and the LLM layer previously judged it
from the bare text alone. previous_assistant_message (nodes/_shared.py)
supplies the single immediately-preceding assistant message, or the
"لا يوجد" placeholder on a first turn — one turn of context, not the full
history, same scope as extract_symptoms's fix. This ONLY reaches the LLM
call above; rule_result is computed from `message` alone, before this
context is even built, so the deterministic layer and the OR combination
below are both completely unaffected — this is strictly additional
context for the LLM's own judgment, not a change to what "matched" means
for either layer.

Both detectors' verdicts are audit-logged together via
audit.logger.log_crisis_detection, separately from the LLM call's own
audit record (which llm_client.call_llm already writes on every call) — so
it is later possible to measure how often each layer catches something the
other missed, not just how often the combined flag ends up True.
"""

from __future__ import annotations

from typing import Any

from audit.logger import log_crisis_detection
from llm_client import call_llm
from nodes._shared import latest_user_message, previous_assistant_message
from prompts.base import build_prompt
from rules.crisis import detect_crisis
from schemas.crisis import CrisisCheckResult
from state import HealixState

_NO_ENTRIES_PLACEHOLDER = "لا يوجد"


def crisis_check(state: HealixState) -> dict[str, Any]:
    """Run both crisis detectors on this turn's patient message and OR them."""
    message = latest_user_message(state)

    # Deterministic layer: raw message only, computed before the LLM
    # prompt (and its extra context) is even built — see module docstring.
    rule_result = detect_crisis(message)

    previous_question = previous_assistant_message(state)
    prompt = build_prompt(
        "crisis_check",
        message=message,
        previous_question=previous_question or _NO_ENTRIES_PLACEHOLDER,
    )
    llm_result = call_llm(
        prompt,
        schema=CrisisCheckResult,
        tier="quality",
        prompt_name="crisis_check",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(llm_result, CrisisCheckResult)

    is_crisis = rule_result.matched or llm_result.is_crisis

    log_crisis_detection(
        thread_id=state.get("thread_id"),
        rule_matched=rule_result.matched,
        rule_categories=list(rule_result.categories),
        llm_matched=llm_result.is_crisis,
        llm_reasoning=llm_result.reasoning,
        combined=is_crisis,
    )

    return {"is_crisis": is_crisis}
