"""emergency_node: terminal node for a confirmed red flag.

Fixed message, no LLM call — an emergency reply must not drift into
reassurance or clinical explanation. The red flag's own reason stays in
state["red_flags"] for the doctor report; not shown to the patient.
"""

from __future__ import annotations

from typing import Any

from state import HealixState

_EMERGENCY_MESSAGE = (
    "الأعراض يلي ذكرتها بتستدعي تدخل طبي إسعافي فورًا. توجه حالًا لأقرب "
    "قسم طوارئ، أو اتصل بالإسعاف إذا ما قدرت توصل بسرعة. لا تأجل هاد "
    "الشي ولا تنتظر لبكرا."
)


def emergency_node(state: HealixState) -> dict[str, Any]:
    return {
        "messages": [{"role": "assistant", "content": _EMERGENCY_MESSAGE}],
        "stage": "emergency",
        "thread_outcome": "emergency",  # sticky — see state.py
        "severity": "emergency",
    }
