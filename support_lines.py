"""Crisis support-line contact info — the single source of truth.

*** PLACEHOLDER DATA — DO NOT DEPLOY AS-IS ***

The entries in SUPPORT_LINES below are inert placeholders, not real phone
numbers. This mirrors rules/crisis.py's CRISIS_PATTERNS: the module fails
safe (loudly absent, obviously fake) rather than giving false confidence
from something that merely *looks* real.

nodes/crisis_node.py never writes a phone number itself — the LLM is never
trusted with this (CLAUDE.md > Non-negotiable safety rules: a
model-generated helpline number is unacceptable), and neither is a
hand-typed one in a prompt template. This file is the only place a real
number may be entered, and verified_support_lines() is the only way
crisis_node may read one.

--- How to add a real number ---

Replace a placeholder SupportLine's `name` and `phone` with the verified
real values, and set `is_placeholder=False`. Do not just flip the flag —
_validate_no_placeholder_markers_on_real_entries() runs at import time and
refuses to start if an entry claims to be real (is_placeholder=False)
while its text still contains "PLACEHOLDER": a real number silently
withheld (a mistaken True) just means a more generic referral; a fake one
presented as real (a mistaken False) means a patient in crisis dials a
number that was never checked. The two mistakes are not equally bad, and
only the second one is blocked by construction here — the same "bias
toward the safe failure" reasoning as rules/crisis.py.

--- If SUPPORT_LINES has no non-placeholder entries ---

verified_support_lines() returns an empty tuple. crisis_node.py treats
that as "no numbers configured yet" — not an error — and falls back to a
generic doctor/hospital referral instead. It never invents a number.
"""

from __future__ import annotations

from dataclasses import dataclass

_PLACEHOLDER_MARKER = "PLACEHOLDER"


@dataclass(frozen=True)
class SupportLine:
    name: str  # organization/service name, shown to the patient as-is
    phone: str  # exactly as the patient should dial it
    is_placeholder: bool = True


SUPPORT_LINES: tuple[SupportLine, ...] = (
    SupportLine(
        name="PLACEHOLDER_ORGANIZATION_NAME_1",
        phone="PLACEHOLDER_PHONE_NUMBER_1",
        is_placeholder=True,
    ),
    SupportLine(
        name="PLACEHOLDER_ORGANIZATION_NAME_2",
        phone="PLACEHOLDER_PHONE_NUMBER_2",
        is_placeholder=True,
    ),
)


def _validate_no_placeholder_markers_on_real_entries(
    lines: tuple[SupportLine, ...],
) -> None:
    """Refuse to start if an entry claims to be real but still reads as one.

    See module docstring: this is the dangerous direction to get wrong —
    a fake number reaching a patient in crisis because someone flipped
    is_placeholder without actually replacing the text.
    """
    problems = [
        line
        for line in lines
        if not line.is_placeholder
        and (_PLACEHOLDER_MARKER in line.name or _PLACEHOLDER_MARKER in line.phone)
    ]
    if problems:
        names = ", ".join(f"{line.name!r} ({line.phone!r})" for line in problems)
        raise ValueError(
            "support_lines.SUPPORT_LINES has entr(y/ies) marked is_placeholder=False "
            f"whose name or phone still contains {_PLACEHOLDER_MARKER!r}: {names}. "
            "Replace the placeholder text itself, not just the flag."
        )


_validate_no_placeholder_markers_on_real_entries(SUPPORT_LINES)


def verified_support_lines() -> tuple[SupportLine, ...]:
    """Real, human-confirmed entries only — never a placeholder.

    Empty until someone replaces the placeholder entries in SUPPORT_LINES
    above with verified local numbers. crisis_node.py is the only caller
    and treats an empty result as expected, not an error.
    """
    return tuple(line for line in SUPPORT_LINES if not line.is_placeholder)
