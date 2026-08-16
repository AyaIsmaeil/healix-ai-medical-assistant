"""Deterministic negation detection over raw patient text — no LLM calls.

Adapted from the previous project's mock provider, which paired a list of
Arabic negation cues with a **bounded lookbehind window**. The window is the
part worth keeping: without it, "ما عندي حرارة بس عندي سعال" would read the
"ما" as negating the cough too, because the cue appears somewhere earlier in
the message. Bounding the lookbehind to a few words is what stops a distant
negation leaking across a clause boundary.

*** This layer supplements the LLM, it does not replace it. ***

Same OR pattern as rules/red_flags.py and rules/crisis.py: `extract_symptoms`
returns `negated_symptoms`, this returns its own set, and the node unions
them (see `merge_negations`). The LLM catches colloquial phrasing this cannot
— it matches canonical vocabulary wording against the raw message, so a
patient saying "ما في كحة" is invisible to it while "ما في سعال" is not.
Neither layer is trusted alone; a symptom either side calls negated is
negated.

Deliberately NOT ported: the old project's `_SYMPTOM_LEXICON`, a keyword
table mapping colloquial phrases to symptom names. That is the
dictionary/fuzzy-matching approach this service replaced with a schema-level
enum, and reintroducing it would put a second, hand-maintained symptom list
next to vocabulary/symptoms.py — exactly the drift the enum design removes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rules.crisis import normalize
from rules.red_flags import _compile_keyword_matcher
from vocabulary.symptoms import CANONICAL_SYMPTOMS

# Characters of lookbehind. Roughly two or three short Arabic words — long
# enough to span "ما في " plus a filler word, short enough that a negation in
# a previous clause does not reach the next symptom. Inherited from the old
# project; tuning it needs real chat transcripts, not a guess.
NEGATION_WINDOW = 12

# Cues written as patterns rather than plain substrings so word boundaries can
# be enforced where it matters. `\s*` absorbs the "ما في" / "مافي" split that
# the old project had to list twice.
_RAW_NEGATION_PATTERNS: tuple[str, ...] = (
    r"\bلا\b",      # لا  — bounded, or it matches inside لاصق, بلاش, ...
    r"ما\s*في",     # ما في / مافي
    r"ما\s*عندي",   # ما عندي
    r"بدون",
    r"\bبلا\b",
    r"ليس",
    r"من\s*غير",
)

_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in _RAW_NEGATION_PATTERNS
)


def _is_negated_at(normalized_text: str, index: int) -> bool:
    """True if a negation cue sits within the window before `index`."""
    window = normalized_text[max(0, index - NEGATION_WINDOW) : index]
    return any(pattern.search(window) for pattern in _NEGATION_PATTERNS)


def detect_negated_symptoms(
    message: str, candidates: Iterable[str] | None = None
) -> frozenset[str]:
    """Canonical symptom names the patient appears to have denied.

    Matches canonical wording against the raw message, both normalized, and
    checks for a negation cue in the bounded window before each occurrence.
    Returns normalized canonical names, consistent with rules/red_flags.py.

    Matching a symptom NAME here is the same word-boundary problem
    rules.red_flags._compile_keyword_matcher already solves for its own
    keyword lists (that module's own docstring: "سكر" appears inside
    "سكرتير"/"سكران"), just applied at a different call site — reused, not
    reinvented, same "confirmed scope, existing reviewed logic" reasoning
    nodes/rag_retrieve.py already used this session for _SUBSUMES. A short
    canonical name like a symptom root could otherwise be "found" as a bare
    substring inside an unrelated longer word, get treated as present, and
    then have its (equally spurious) negation status checked — a false
    detection either way. _PROCLITICS tolerance (attached ال/و/ف/ب/ك/ل)
    still applies, same as red_flags.py's own keyword matching.

    This under-matches by design — see the module docstring. A name absent
    from the result is not evidence the patient confirmed it.
    """
    normalized_text = normalize(message)
    names = CANONICAL_SYMPTOMS if candidates is None else candidates

    negated = set()
    for name in names:
        normalized_name = normalize(name)
        if not normalized_name:
            continue
        pattern = _compile_keyword_matcher(frozenset({normalized_name}))
        if pattern is None:
            continue
        for match in pattern.finditer(normalized_text):
            if _is_negated_at(normalized_text, match.start()):
                negated.add(normalized_name)
                break

    return frozenset(negated)


def merge_negations(
    llm_negated: Iterable[str] | None, rule_negated: Iterable[str] | None
) -> frozenset[str]:
    """Union the LLM's negations with this layer's — never the intersection.

    The OR is the safety property: each layer misses things the other
    catches, so requiring agreement would discard exactly the cases the
    second layer exists to recover. A negation carries diagnostic weight
    equal to a confirmed symptom (CLAUDE.md > State), so dropping one is
    losing clinical information, not being conservative.
    """
    return frozenset(normalize(n) for n in (llm_negated or []) if n) | frozenset(
        normalize(n) for n in (rule_negated or []) if n
    )
