"""crisis_node: terminal node for the crisis path.

The LLM only phrases the reply (prompts/templates/crisis_node.txt keeps
it from analyzing, diagnosing, or asking follow-ups). Support-line
numbers are never LLM-written — support_line_text() appends real,
verified numbers in code after the call returns.

severity is deliberately not set here — crisis is a different axis from
clinical severity (contrast emergency_node, which does set it).
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from nodes._shared import latest_user_message, support_line_text
from prompts.base import build_prompt
from schemas.crisis import CrisisResponse
from state import HealixState


def crisis_node(state: HealixState) -> dict[str, Any]:
    message = latest_user_message(state)

    prompt = build_prompt("crisis_node", message=message)
    llm_result = call_llm(
        prompt,
        schema=CrisisResponse,
        tier="quality",
        prompt_name="crisis_node",
        thread_id=state.get("thread_id"),
    )
    assert isinstance(llm_result, CrisisResponse)

    reply = f"{llm_result.message}\n\n{support_line_text()}"

    return {
        "messages": [{"role": "assistant", "content": reply}],
        "stage": "crisis",
        "thread_outcome": "crisis",
    }
