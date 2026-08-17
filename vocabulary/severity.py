"""Symptom severity: the closed level set, and the Arabic words that map to it.

Adapted from the previous project's `feature_extraction_rules.py`, which
graded severity on a 0-10 numeric scale. This service uses three qualitative
levels instead, so both the word list and the numeric scale are collapsed
onto them here.

Why three and not 0-10: a number invites false precision from something that
is a patient's subjective report, and CLAUDE.md > Non-negotiable safety rules
keeps computed numbers to the RAG match score. "شديد" is a real thing a
patient said; "8/10" implies a measurement nobody took.

`SymptomSeverity` lives here rather than in `schemas/` for the same reason
symptom names do: `vocabulary/` is where closed vocabularies are defined, and
`schemas/` consumes them. Defining it in `schemas/` and importing it here
would invert that.

--- The 0-10 collapse -------------------------------------------------------

Band boundaries follow the conventional trichotomy used with numeric rating
scales for symptom intensity:

    0-3   -> mild
    4-6   -> moderate
    7-10  -> severe

Every word inherited from the old project lands consistently under it:

    word          old score   ->  level
    خفيف              2       ->  mild
    بسيط              2       ->  mild
    متوسط             5       ->  moderate
    شديد              8       ->  severe
    شديد جدا          9       ->  severe
    لا يطاق          10       ->  severe

The mapping below is stated directly as words -> levels rather than
words -> numbers -> levels: routing a qualitative word through a number that
nothing else uses would just be a lossy detour.
"""

from __future__ import annotations

import re
from typing import Literal

from rules.crisis import normalize
from rules.red_flags import _compile_keyword_matcher

SymptomSeverity = Literal["mild", "moderate", "severe"]

# Band upper bounds for the 0-10 collapse documented above.
_MILD_MAX = 3
_MODERATE_MAX = 6
_SCALE_MAX = 10

_RAW_SEVERITY_WORDS: dict[str, SymptomSeverity] = {
    "خفيف": "mild",
    "بسيط": "mild",
    "متوسط": "moderate",
    "شديد": "severe",
    "شديد جدا": "severe",
    "لا يطاق": "severe",
}

SEVERITY_WORDS: dict[str, SymptomSeverity] = {
    normalize(word): level for word, level in _RAW_SEVERITY_WORDS.items()
}

# Word-boundary-aware matcher per word — same rules.red_flags.
# _compile_keyword_matcher discipline used for its own keyword lists
# (reused, not reinvented: that module's own docstring documents exactly
# this collision class, "سكر" inside "سكرتير"). Plain substring
# containment is wrong here too: "خفيف" (mild) is a bare substring of
# "تخفيف" (relief/reduction, e.g. "تخفيف الألم" — pain relief), a
# completely unrelated word a patient asking for pain relief would
# naturally use, with no severity grading intended at all. Built once
# per word at import, mirroring SEVERITY_WORDS's own "normalize once"
# construction above.
_SEVERITY_WORD_MATCHERS: dict[str, re.Pattern[str]] = {
    word: matcher
    for word in SEVERITY_WORDS
    if (matcher := _compile_keyword_matcher(frozenset({word}))) is not None
}

# "8 من 10" / "8/10". normalize() has already folded Arabic-Indic digits.
_SCORE_OUT_OF_TEN = re.compile(r"(\d{1,2})\s*(?:من|/)\s*10")


def severity_from_score(score: int) -> SymptomSeverity:
    """Collapse a 0-10 rating onto the three levels. See module docstring."""
    if not 0 <= score <= _SCALE_MAX:
        raise ValueError(f"severity score {score} is outside 0-{_SCALE_MAX}")
    if score <= _MILD_MAX:
        return "mild"
    if score <= _MODERATE_MAX:
        return "moderate"
    return "severe"


def severity_from_text(text: str) -> SymptomSeverity | None:
    """Best-effort severity from patient wording, or None if not stated.

    None means "the patient did not grade it", never a default level —
    guessing a severity would put an invented clinical fact into the record.

    An explicit "8 من 10" wins over a severity word: a patient who gives both
    has been more specific with the number.
    """
    normalized = normalize(text)

    match = _SCORE_OUT_OF_TEN.search(normalized)
    if match:
        score = int(match.group(1))
        if 0 <= score <= _SCALE_MAX:
            return severity_from_score(score)

    # Longest first: "شديد جدا" contains "شديد", and a shorter match would
    # shadow it. Both are "severe" today, but relying on that would break
    # silently the moment the level set changes.
    for word in sorted(SEVERITY_WORDS, key=len, reverse=True):
        if _SEVERITY_WORD_MATCHERS[word].search(normalized):
            return SEVERITY_WORDS[word]

    return None
