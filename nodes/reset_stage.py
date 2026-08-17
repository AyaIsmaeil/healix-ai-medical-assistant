"""reset_stage: clears state["stage"] to None at the very start of every
turn.

CLAUDE.md > State: state["stage"] has no reducer, so without this it is
last-write-wins across the WHOLE checkpointed thread, not reset per turn
— a turn that reaches a branch with no stage-setting node of its own
(today: assess_sufficiency's sufficient branch, the rag_retrieve
placeholder) would silently carry over a PREVIOUS turn's stage value
from earlier in the same thread, misreporting what actually happened
this turn.

Wired as the very first node in the graph (START -> reset_stage ->
crisis_check -> ...) — before crisis_check, extract_symptoms, or any
other node that runs later in the same turn and could go on to set a
real stage (crisis_node, emergency_node, ask_followup today; more once
rag_retrieve/diagnose/generate_reports exist). Runs unconditionally on
every turn regardless of which branch that turn ultimately takes, so it
is correct today with only crisis_check/emergency_node/ask_followup
live — it does not depend on generate_reports (or any other node) being
built yet, unlike the placeholder-END branches elsewhere in this graph.

No LLM call, no clinical judgment — pure per-turn state bookkeeping, not
business logic.

Deliberately does NOT touch state["thread_outcome"] (CLAUDE.md >
Non-negotiable safety rule 13). That field is sticky ON PURPOSE — the
opposite of stage — so a thread that already reached crisis or emergency
stays flagged across every later turn, not just the one that set it. See
state.py's own field comment and nodes/reiterate_terminal_outcome.py's
module docstring for why the two fields need opposite per-turn
lifetimes.
"""

from __future__ import annotations

from typing import Any

from state import HealixState


def reset_stage(state: HealixState) -> dict[str, Any]:
    """Clear this turn's stage before any node in it can set one."""
    return {"stage": None}
