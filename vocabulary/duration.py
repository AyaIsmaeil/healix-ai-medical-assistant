"""Duration in days, or None when nothing is stated.

None means "the patient did not say", never "assume zero" — an absent
duration is information, and inventing one corrupts the record.
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

_DURATION_FIXED_WORD_MATCHERS: dict[str, re.Pattern[str]] = {
    word: matcher
    for word in DURATION_FIXED_WORDS
    if (matcher := _compile_keyword_matcher(frozenset({word}))) is not None
}

_COUNTED_DURATION = re.compile(
    r"(?:منذ|من)?\s*(\d{1,3})\s*("
    + "|".join(sorted(DURATION_UNIT_DAYS, key=len, reverse=True))
    + r")(?!\w)"
)


def parse_duration_days(text: str) -> int | None:
    """Return the duration in days, or None if no duration is stated.
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
