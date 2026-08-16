"""Small helpers shared by more than one node.

Not a node itself — nothing here is registered in graph.py. Factored out
once a second node (nodes/crisis_node.py) needed the exact logic
nodes/crisis_check.py already had, rather than let two copies of the same
"which message is this turn about" rule quietly drift apart. Same
reasoning applies to support_line_text() below, factored out here once
nodes/reiterate_terminal_outcome.py needed the identical verified-numbers-
or-generic-fallback logic nodes/crisis_node.py already had.
"""

from __future__ import annotations

from state import HealixState
from support_lines import verified_support_lines


def latest_user_message(state: HealixState) -> str:
    """The message this turn is about — the most recent "user"-role entry.

    Each entry conforms to api.contracts.Message (CLAUDE.md > Laravel <->
    service contract).

    state["messages"] can hold prior turns too (CLAUDE.md > State), so this
    is not simply the last entry in the list: an assistant follow-up
    question (or, on the crisis path, the crisis response itself) appended
    after the patient's own message would otherwise be read instead of
    what the patient actually said.
    """
    for message in reversed(state["messages"]):
        if message.get("role") == "user":
            return message.get("content", "")
    raise ValueError("state['messages'] has no user-role message.")


def previous_assistant_message(state: HealixState) -> str | None:
    """The assistant message immediately before this turn's patient
    message, if there is one — the follow-up question the patient may be
    replying to (CLAUDE.md > Graph flow: ask_followup ends a turn with
    exactly one assistant message, awaiting the next user message).

    None on a thread's first turn (no assistant message exists yet), or
    if the entry immediately before the latest user message isn't an
    assistant message at all — never guesses by falling back to some
    earlier assistant message further back in state["messages"], which
    would not actually be what this specific reply is answering.

    Exists because a patient's reply to a direct question is very often
    NOT self-contained ("من الصبح ومستمر" only means anything in light of
    "منذ متى بدأ هذا الصداع؟") — see nodes/extract_symptoms.py, the node
    this was built for.
    """
    messages = state["messages"]
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            if index == 0:
                return None
            previous = messages[index - 1]
            return previous.get("content") if previous.get("role") == "assistant" else None
    return None


# Used only when support_lines.verified_support_lines() has nothing yet
# (CLAUDE.md > Non-negotiable safety rule 11 — never invents a number) —
# a generic, always-true referral that names no specific service.
GENERIC_REFERRAL = "الأفضل توجه هلق لأقرب طبيب أو مستشفى، وشارك معهم شو عم تحس فيه."


def support_line_text() -> str:
    """Real verified support-line numbers if support_lines.verified_support_lines()
    has any, GENERIC_REFERRAL otherwise. Deterministic and entirely
    code-driven, never the LLM's (safety rule 11) — never empty.

    Shared by crisis_node (its first, personalized crisis message) and
    reiterate_terminal_outcome (a later turn's fixed reminder on an
    already-crisis thread) — both need the identical lookup, not just
    similar-looking text, since a real number appearing in one path and
    not the other would be exactly the kind of drift safety rule 11
    exists to prevent.
    """
    lines = verified_support_lines()
    if not lines:
        return GENERIC_REFERRAL
    return "\n".join(f"{line.name}: {line.phone}" for line in lines)
