"""Arabic duration lexicon for the `duration` field of extract_symptoms.

Adapted from the previous project's `feature_extraction_rules.py`. Its value
is the Arabic **dual** forms — يومين / أسبوعين / شهرين ("two days", "two
weeks", "two months") are single words with no digit in them, so a general
"number + unit" regex misses them entirely, and they are among the most
common ways a patient states a duration.

Everything here is keyed on normalized text (rules.crisis.normalize), so
callers do not need to spell each word two ways: the old project listed both
"أسبوع" and "اسبوع" because it matched raw text, and normalization collapses
that pair to one entry.

This module parses; it does not decide. `extract_symptoms` stores the
patient's own wording in `ExtractedSymptom.duration` (free text, per
CLAUDE.md). `parse_duration_days` is for downstream code that needs a number
— never for overwriting what the patient actually said.
"""

from __future__ import annotations

import re

from rules.crisis import normalize
from rules.red_flags import _compile_keyword_matcher

# Fixed dual forms: word -> days. No digit appears in these, which is exactly
# why a numeric regex cannot find them.
_RAW_FIXED_WORDS: dict[str, int] = {
    "يومين": 2,
    "أسبوعين": 14,
    "شهرين": 60,
}

# Unit word -> days per single unit. Singular and plural both map to the same
# per-unit value; the count comes from the digits beside them.
_RAW_UNIT_DAYS: dict[str, int] = {
    "يوم": 1,
    "أيام": 1,
    "أسبوع": 7,
    "أسابيع": 7,
    "شهر": 30,
    "أشهر": 30,
    "سنة": 365,
    "سنوات": 365,
}

DURATION_FIXED_WORDS: dict[str, int] = {
    normalize(word): days for word, days in _RAW_FIXED_WORDS.items()
}
DURATION_UNIT_DAYS: dict[str, int] = {
    normalize(word): days for word, days in _RAW_UNIT_DAYS.items()
}

# Word-boundary-aware matcher per fixed word — same rules.red_flags.
# _compile_keyword_matcher discipline used for its own keyword lists
# (reused, not reinvented: that module's own docstring documents exactly
# this collision class, "سكر" inside "سكرتير"). Plain substring
# containment is wrong here too: "يوم" (day) is a bare substring of
# "يومي"/"يومياً" (daily, an adjective/adverb about FREQUENCY, not
# duration — e.g. "بصير معي 3 مرات يومياً" describes how often an
# attack recurs, not how long ago it started). Built once per word at
# import, mirroring DURATION_FIXED_WORDS's own "normalize once"
# construction above.
_DURATION_FIXED_WORD_MATCHERS: dict[str, re.Pattern[str]] = {
    word: matcher
    for word in DURATION_FIXED_WORDS
    if (matcher := _compile_keyword_matcher(frozenset({word}))) is not None
}

# "منذ ٣ أيام" / "من 3 ايام" / bare "3 ايام". normalize() has already folded
# Arabic-Indic digits to ASCII by the time this runs, so \d suffices.
# Trailing (?!\w): without it, "3 اسبوعية" ("3 weekly [something]", an
# adjective — nothing to do with a 3-week duration) would match "اسبوع" as
# a bare PREFIX of "اسبوعية" and silently compute 21 days. No leading
# boundary needed on the digit side — \d never blends with a preceding
# Arabic letter the way \w-vs-\w does on the trailing side.
_COUNTED_DURATION = re.compile(
    r"(?:منذ|من)?\s*(\d{1,3})\s*("
    + "|".join(sorted(DURATION_UNIT_DAYS, key=len, reverse=True))
    + r")(?!\w)"
)


def parse_duration_days(text: str) -> int | None:
    """Best-effort duration in days, or None when nothing is stated.

    None means "the patient did not say", never "assume zero" — an absent
    duration is information, and inventing one corrupts the record.

    Fixed dual forms are checked first: "يومين" contains "يوم", so a
    unit-based match would read it as an unqualified single day.
    """
    normalized = normalize(text)

    for word, days in DURATION_FIXED_WORDS.items():
        if _DURATION_FIXED_WORD_MATCHERS[word].search(normalized):
            return days

    match = _COUNTED_DURATION.search(normalized)
    if not match:
        return None

    count = int(match.group(1))
    return count * DURATION_UNIT_DAYS[match.group(2)]
