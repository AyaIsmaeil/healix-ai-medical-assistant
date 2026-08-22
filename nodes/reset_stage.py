"""reset_stage: clears state["stage"] to None at the start of every turn.

stage has no reducer (last-write-wins across the whole thread), so
without this a turn that doesn't reach a stage-setting node would carry
over a previous turn's value. Runs first, unconditionally.

Deliberately does NOT touch thread_outcome — that field is sticky on
purpose (see state.py).
"""

from __future__ import annotations

from typing import Any

from state import HealixState


def reset_stage(state: HealixState) -> dict[str, Any]:
    return {"stage": None}
