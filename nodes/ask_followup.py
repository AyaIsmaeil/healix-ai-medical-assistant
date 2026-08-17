"""ask_followup: the terminal node for the "insufficient information"
branch of the assess_sufficiency loop (CLAUDE.md > Graph flow:
assess_sufficiency -> insufficient -> ask_followup -> END, awaiting the
next user message).

Deliberately has no LLM call, unlike crisis_node: the question text was
already generated upstream by assess_sufficiency
(schemas.sufficiency.SufficiencyAssessment.next_question) and validated
there (schemas/sufficiency.py's model_validator guarantees a non-empty
Arabic question whenever is_sufficient is False, falling back to a fixed
generic question rather than leaving it blank). This node's only job is
mechanical: turn that already-decided text into a conversation turn and
end the graph invocation, so the next real invoke() on the same
thread_id picks up the patient's answer via the checkpointer
(CLAUDE.md > State: persistence is a LangGraph checkpointer keyed by
thread_id) — there is nothing left to decide here.

state["next_question"] is read, not cleared: it is plain (no reducer,
CLAUDE.md > State), and assess_sufficiency recomputes it fresh on every
call it makes, so the next turn's assess_sufficiency run will overwrite
it (to a new question, or None once sufficient) before this node could
ever see a stale value.
"""

from __future__ import annotations

from typing import Any

from state import HealixState


def ask_followup(state: HealixState) -> dict[str, Any]:
    """Send state["next_question"] to the patient and end this turn."""
    return {
        "messages": [{"role": "assistant", "content": state["next_question"]}],
        "stage": "followup",
    }
