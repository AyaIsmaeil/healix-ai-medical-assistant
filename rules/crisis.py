"""Deterministic crisis detection — no LLM calls in this file.

Pattern-matches the raw patient message for signs of acute psychological
distress or self-harm intent, before symptom extraction ever runs
(CLAUDE.md > Graph flow: crisis_check precedes extract_symptoms). This is
the rule-based half of crisis detection; per CLAUDE.md > Non-negotiable
safety rules it is never removed, weakened, or made conditional on LLM
output — it is meant to be combined with an LLM check via OR by the
crisis_check node, not replaced by one.

Bias toward false positives: a missed crisis is unacceptable, a false
positive costs one gentle redirect message. When in doubt, match.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --- normalization ---------------------------------------------------------
#
# Syrian colloquial Arabic chat text varies a lot in spelling. Patterns are
# written once against normalized text; everything below folds those
# variants down so a pattern doesn't have to enumerate them.

# Arabic combining diacritics (tashkeel). Written as \uXXXX escapes rather
# than literal combining marks: those attach to whatever character precedes
# them in the source and render as invisible or ambiguous in most editors,
# making the class impossible to read or verify by eye.
_DIACRITICS = re.compile(
    "["
    "\u0610-\u061a"  # honorific signs (salla-allahu alayhi wasallam, etc.)
    "\u064b-\u065f"  # tanwin, harakat, shadda, sukun, hamza marks
    "\u0670"  # superscript alef
    "\u06d6-\u06ed"  # Quranic annotation marks
    "]"
)
_TATWEEL = "ـ"  # ـ (kashida), used to stretch words for emphasis/justification

_ALEF_VARIANTS = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
})
_TAA_MARBUTA_TO_HAA = str.maketrans({"ة": "ه"})

# Arabic-Indic digits -> ASCII. An Arabic keyboard emits these by default, so
# a patient typing "٣٩ درجة" would otherwise carry a number nothing downstream
# can read. NFKC does not do this (verified — they have no compatibility
# decomposition to ASCII), so it has to be explicit. Both ranges are folded:
# U+0660-0669 (Arabic-Indic) and U+06F0-06F9 (Extended, Persian/Urdu
# keyboards) — the latter is visually near-identical and just as invisible.
_ARABIC_INDIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩" "۰۱۲۳۴۵۶۷۸۹",
    "0123456789" "0123456789",
)

# Same letter repeated 3+ times in a row -> collapse to one. Common in chat
# Arabic for emphasis (e.g. "موووت" for "موت") and would otherwise dodge
# every literal pattern.
#
# Digits are excluded (\D): collapsing them would turn "999" into "9" and
# "١١١" into "1", silently corrupting exactly the numbers digit folding above
# exists to make readable.
_ELONGATION = re.compile(r"(\D)\1{2,}")

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize Syrian colloquial Arabic chat text for pattern matching.

    Order matters: diacritics/tatweel are stripped before elongation is
    collapsed, so a stretched, diacritized letter doesn't leave stray
    marks behind; digits are folded before elongation so the digit-safe
    branch of that rule sees ASCII; alef/taa-marbuta unification happens
    last so patterns only need to spell each word one way.

    Known gap, not handled here: letters spaced out one-by-one for
    emphasis or censorship evasion (e.g. "ب د ي"). Collapsing that
    reliably without merging unrelated short words needs real chat data
    to tune against, not a guess.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = _ELONGATION.sub(r"\1", text)
    text = text.translate(_ALEF_VARIANTS)
    text = text.translate(_TAA_MARBUTA_TO_HAA)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# --- pattern data ------------------------------------------------------
#
# *** PLACEHOLDER DATA — DO NOT DEPLOY AS-IS ***
#
# The entries below only demonstrate the shape of a CrisisPattern. They are
# deliberately inert (they match a literal marker string, not real speech)
# so this module fails safe — silent, obvious non-coverage — rather than
# giving false confidence from an incomplete list that "looks real."
#
# This list MUST be reviewed and completed by someone with both clinical
# mental-health expertise and fluency in Syrian colloquial Arabic before
# rules/crisis.py is trusted for real patient messages.
#
# --- How to write a pattern (read this before editing) -----------------
#
# A pattern is checked against a CLEANED-UP version of what the patient
# typed, not against the raw message. That cleanup (done automatically,
# above) does the following before any pattern is checked:
#   - the different spellings of alef with hamza (أ إ آ ٱ) all become
#     plain ا
#   - ة becomes ه
#   - diacritic marks (تشكيل) are removed entirely
#   - a letter stretched out for emphasis ("موووت") is shrunk back down
#     to one letter ("موت")
#   - extra spaces are collapsed to one
#
# So: write each pattern the way it looks AFTER that cleanup, not the way
# you would naturally type it. For example:
#   - use ا, never أ / إ / آ / ٱ  →  write "انا", not "أنا"
#   - use ه, never ة             →  write "المدرسه", not "المدرسة"
#   - no diacritics — plain letters only
#   - repeated letters and extra spacing don't need special handling,
#     they're already taken care of
#
# You do not have to get this right by memory: if a pattern is written
# the wrong way, the program will refuse to start and print an error
# naming the exact pattern, showing both what you wrote and what it
# should be instead. So a mistake here is caught immediately — it will
# not sit silently broken.
#
# When filling this in more broadly:
#   - cover direct statements, indirect/euphemistic phrasing, and the
#     misspellings/shorthand actually seen in chat messages
#   - err toward overly broad patterns — see module docstring: false
#     positives are cheap here, false negatives are not
#   - keep categories clinically meaningful (e.g. distinguish active
#     intent from passive hopelessness) so downstream routing and the
#     audit log can tell them apart, not just "matched: true/false"


@dataclass(frozen=True)
class CrisisPattern:
    category: str
    pattern: re.Pattern[str]


CRISIS_PATTERNS: tuple[CrisisPattern, ...] = (
    # TODO(mental-health-review): replace with real, reviewed patterns.
    CrisisPattern("suicidal_ideation", re.compile(r"PLACEHOLDER_SUICIDAL_IDEATION")),
    CrisisPattern("self_harm_intent", re.compile(r"PLACEHOLDER_SELF_HARM_INTENT")),
    CrisisPattern("hopelessness_severe", re.compile(r"PLACEHOLDER_HOPELESSNESS_SEVERE")),
)


def _validate_patterns_are_normalized(patterns: tuple[CrisisPattern, ...]) -> None:
    """Fail fast if a pattern isn't written the way normalize() would write it.

    Every pattern is matched against normalize(message), never against the
    raw message. A pattern that still has e.g. أ where normalize() would
    produce ا will therefore never match anything — silently, with every
    test still green, because there's nothing in the pattern's own text
    to say it's wrong. This runs at import time so that mistake surfaces
    immediately instead of showing up as a missed crisis later.
    """
    problems = []
    for crisis_pattern in patterns:
        written = crisis_pattern.pattern.pattern
        should_be = normalize(written)
        if written != should_be:
            problems.append(f"  [{crisis_pattern.category}] wrote {written!r}, should be {should_be!r}")

    if problems:
        raise ValueError(
            "CRISIS_PATTERNS has pattern(s) not written in normalized form "
            "(see 'How to write a pattern' above rules/crisis.py:CRISIS_PATTERNS):\n"
            + "\n".join(problems)
        )


_validate_patterns_are_normalized(CRISIS_PATTERNS)


# --- matching ------------------------------------------------------------


@dataclass(frozen=True)
class CrisisResult:
    # Every category that matched, in the order CRISIS_PATTERNS defines
    # them, deduplicated. Not just the first hit: a single message can
    # plausibly signal more than one category (e.g. hopelessness and
    # active intent together), and the audit trail should show all of
    # them rather than let pattern order silently pick one to keep.
    categories: tuple[str, ...] = ()

    @property
    def matched(self) -> bool:
        """The only thing crisis_check's branch decision needs: yes/no."""
        return bool(self.categories)


def detect_crisis(message: str) -> CrisisResult:
    """Check a raw patient message against CRISIS_PATTERNS.

    Matches on the raw message text, not on extracted symptoms — this
    must run before extract_symptoms (CLAUDE.md > Graph flow). Returns a
    structured result rather than a bool so the audit layer can record
    every category that fired, not just whether something did. Routing
    only needs CrisisResult.matched; the full category list is for the
    audit trail, not the crisis_check node's branch decision.
    """
    normalized = normalize(message)
    categories = tuple(
        dict.fromkeys(
            crisis_pattern.category
            for crisis_pattern in CRISIS_PATTERNS
            if crisis_pattern.pattern.search(normalized)
        )
    )
    return CrisisResult(categories=categories)
