"""Guards vocabulary/symptoms.py's normalization-at-storage contract.

CLAUDE.md > Symptom vocabulary > Compare against the normalized form,
never the authored spelling: CANONICAL_SYMPTOMS and
schemas.symptoms.SYMPTOM_NAMES hold NORMALIZED strings, not the
human-readable spelling _SYMPTOM_NAMES is authored with (e.g. أ vs ا).
Comparing a symptom name against either collection without normalizing
first silently fails to match — the live value is the normalized one,
and the authored spelling is never live past import time. This file
makes that an explicit, tested contract rather than something only
discoverable by reading vocabulary/symptoms.py's source and reasoning
about import order.
"""

from __future__ import annotations

import pytest

from rules.crisis import normalize
from schemas.symptoms import SYMPTOM_NAMES
from vocabulary.symptoms import _SYMPTOM_NAMES, CANONICAL_SYMPTOMS, is_canonical


def _an_entry_that_actually_changes_under_normalization() -> str:
    """An authored _SYMPTOM_NAMES entry where normalize(entry) != entry.

    Not just any entry: one that genuinely changes is what exercises the
    contract this file guards. Picking one that happens to be unaffected
    by normalize() (no hamza, no taa marbuta, ...) would let both
    assertions below pass vacuously without proving anything.
    """
    for entry in _SYMPTOM_NAMES:
        if normalize(entry) != entry:
            return entry
    pytest.fail(
        "No entry in _SYMPTOM_NAMES differs from its normalized form — "
        "this test needs one to actually exercise the normalization "
        "contract it guards."
    )


AUTHORED = _an_entry_that_actually_changes_under_normalization()


def test_authored_spelling_is_not_the_live_canonical_value():
    # The exact regression this guards: new code compares a symptom name
    # against CANONICAL_SYMPTOMS or SYMPTOM_NAMES using the spelling as
    # written in vocabulary/symptoms.py's source instead of normalizing
    # first, and the comparison silently fails.
    assert AUTHORED not in CANONICAL_SYMPTOMS
    assert AUTHORED not in SYMPTOM_NAMES


def test_normalized_spelling_is_the_live_canonical_value():
    normalized = normalize(AUTHORED)

    assert normalized in CANONICAL_SYMPTOMS
    assert normalized in SYMPTOM_NAMES


def test_is_canonical_accepts_the_authored_spelling_by_normalizing_internally():
    # is_canonical() is the sanctioned way to check membership precisely
    # because it normalizes internally — a caller using it never has to
    # remember this rule for themselves.
    assert is_canonical(AUTHORED) is True
