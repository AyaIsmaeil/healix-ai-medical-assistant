"""verify_red_flag: asks ONE targeted clarification question for a
cited-rule candidate whose core symptom is present but whose
discriminators aren't yet confirmed or denied. Runs only when
check_red_flags set safety_decision="NEEDS_CLARIFICATION".

state["red_flag_candidates"] supplies the fixed, cited list of missing
discriminators; the LLM only phrases the question, never adds to that
list (prompts/templates/verify_red_flag.txt).

Resolution happens on a LATER turn, not here: check_red_flags re-runs
fresh next turn — a confirmed discriminator becomes a real hard match
(-> emergency_node), a denied one marks the candidate rejected. This
node never classifies the answer itself.

Shares nodes.assess_sufficiency.MAX_FOLLOW_UP_QUESTIONS rather than a
new ceiling. Past the budget, it stops asking and sets
information_limited=True instead of escalating on an unresolved
candidate.
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS
from prompts.base import build_prompt
from schemas.red_flags import VerificationQuestion
from state import HealixState


def _pick_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    # One question per turn — any other unresolved candidates are
    # re-evaluated fresh next turn, not lost.
    return candidates[0]


def _format_required_information(missing_any_of: list[str]) -> str:
    return "\n".join(f"- {item}" for item in missing_any_of)


def verify_red_flag(state: HealixState) -> dict[str, Any]:
    candidates = state.get("red_flag_candidates") or []
    if not candidates:
        return {"next_question": None}

    turn_count = state.get("turn_count", 0)
    if turn_count >= MAX_FOLLOW_UP_QUESTIONS:
        return {"next_question": None, "information_limited": True}

    candidate = _pick_candidate(candidates)
    prompt = build_prompt(
        "verify_red_flag",
        required_information=_format_required_information(candidate["missing_any_of"]),
        candidate_id=candidate["rule_id"],
    )
    result = call_llm(
        prompt,
        schema=VerificationQuestion,
        tier="quality",
        prompt_name="verify_red_flag",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(result, VerificationQuestion)

    return {"next_question": result.question, "turn_count": turn_count + 1}
