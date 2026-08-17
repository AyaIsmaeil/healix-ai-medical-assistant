"""crisis_node: the terminal node for the crisis path (CLAUDE.md > Graph
flow, CLAUDE.md > Non-negotiable safety rule 9: "stop analyzing symptoms
entirely and output only the crisis marker").

Produces the one message a patient in acute distress receives. This node
never touches symptoms, red flags, or diagnosis — its return value is
only ever a new assistant message, stage="crisis", and thread_outcome=
"crisis" (below). Its prompt (prompts/templates/crisis_node.txt) is
written to keep the LLM from analyzing, diagnosing, reassuring, asking a
follow-up question, or suggesting a coping technique either; that
prompt-level enforcement is the best available tool here, the same way
the rest of the safety preamble is prompt-level (CLAUDE.md tags this
[preamble] for what reaches every call and [node: x.txt] for what's
specific to one).

Support-line numbers are never the LLM's to write (CLAUDE.md >
Non-negotiable safety rules: a model-generated helpline number is
unacceptable). support_lines.py is the single source of truth for them;
nodes._shared.support_line_text() calls verified_support_lines() and
returns whatever it finds — real numbers if any have been verified, or a
generic doctor/hospital referral if not — entirely in code, after the LLM
call returns. schemas.crisis.CrisisResponse has no field for a number at
all, so there is no path from the LLM's output to one appearing in the
reply. support_line_text() lives in nodes/_shared.py, not here, because
nodes/reiterate_terminal_outcome.py (CLAUDE.md > Non-negotiable safety
rule 13) needs the identical lookup for a later turn's fixed reminder on
an already-crisis thread — factored out once a second node needed it,
same reasoning as nodes/_shared.py's other helpers.

state["thread_outcome"] = "crisis": set unconditionally here, the moment
this node fires. Sticky — unlike state["stage"], nodes/reset_stage.py
never clears it — so a LATER turn that escalates nothing new (neither a
fresh crisis signal nor a red flag) is routed by graph.py to
reiterate_terminal_outcome instead of silently continuing into normal
symptom triage. See state.py's own field comment and
nodes/reiterate_terminal_outcome.py's module docstring for the full
reasoning, including why this is a genuinely different concept from
state["severity"] below.

state["severity"] is deliberately NOT set here. Safety rule 9 requires
stopping symptom analysis entirely on this path — there is no clinical-
severity judgment to record, because none is made. (Contrast
nodes/emergency_node.py, which DOES set severity="emergency" — a red
flag firing is itself an emergency-level severity judgment; crisis is a
different axis, not a point on that scale at all.)
"""

from __future__ import annotations

from typing import Any

from llm_client import call_llm
from nodes._shared import latest_user_message, support_line_text
from prompts.base import build_prompt
from schemas.crisis import CrisisResponse
from state import HealixState


def crisis_node(state: HealixState) -> dict[str, Any]:
    """Produce the crisis-path response and mark this turn's stage."""
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
