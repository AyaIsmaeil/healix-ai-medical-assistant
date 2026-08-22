"""Small helpers shared by more than one node. Not a node itself —
nothing here is registered in graph.py."""

from __future__ import annotations

from state import HealixState
from support_lines import verified_support_lines


def latest_user_message(state: HealixState) -> str:
    """The most recent user-role message — not just the last entry in the
    list, since an assistant reply may already have been appended."""
    for message in reversed(state["messages"]):
        if message.get("role") == "user":
            return message.get("content", "")
    raise ValueError("state['messages'] has no user-role message.")


def previous_assistant_message(state: HealixState) -> str | None:
    """The assistant message immediately before this turn's patient
    message, if any — never falls back to an earlier one further back."""
    messages = state["messages"]
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            if index == 0:
                return None
            previous = messages[index - 1]
            return previous.get("content") if previous.get("role") == "assistant" else None
    return None


# Fallback when no verified support line exists yet — never a made-up number.
GENERIC_REFERRAL = "الأفضل توجه هلق لأقرب طبيب أو مستشفى، وشارك معهم شو عم تحس فيه."


def support_line_text() -> str:
    """Real verified support-line numbers if any exist, GENERIC_REFERRAL
    otherwise. Deterministic, never the LLM's — never empty."""
    lines = verified_support_lines()
    if not lines:
        return GENERIC_REFERRAL
    return "\n".join(f"{line.name}: {line.phone}" for line in lines)
