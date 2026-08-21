"""ask_followup: terminal node for the "insufficient information" branch.
No LLM call — the question text was already generated and validated
upstream (assess_sufficiency or rag_retrieve). Purely mechanical: send
it, end the turn.
"""

from __future__ import annotations

from typing import Any

from state import HealixState


def ask_followup(state: HealixState) -> dict[str, Any]:
    return {
        "messages": [{"role": "assistant", "content": state["next_question"]}],
        "stage": "followup",
    }
