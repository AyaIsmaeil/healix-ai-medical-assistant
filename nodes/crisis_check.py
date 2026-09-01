
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
    message = latest_user_message(state)

    # Raw message only — computed before the LLM's extra context is built.
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
