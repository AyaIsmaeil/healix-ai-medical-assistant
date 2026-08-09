"""
Healix - Symptom Coder (runtime concept-coding layer).

⚠️ DORMANT / NOT WIRED (2026-08-08). This module — together with
``app/ml/`` (feature_encoder_v2, schema_v2_generator, evidence_parser,
dataset_builder), ``symptom_ontology.json`` and Feature Schema **v2** — is a
more ambitious alternative to the ACTIVE ML bridge
(``domain.symptom_evidence_encoder`` + ``dictionaries/symptom_evidence_map.yaml``).
It is deliberately left unconnected to ``main.py`` because it cannot run against
the current artifacts: it targets a 286-feature v2 vector, but the only trained
model that exists expects **225** features (v1 + E_* evidence), and its Arabic
lexicon is only ~32/98 concepts filled. Wiring it would require retraining the
model on v2 and completing the ontology under clinical review — future work,
not the demo path. Kept in-tree as that future direction, intentionally idle.


Bridges the gap the ML path was blocked on: the clinical interview emits
``Symptom.text`` as FREE Arabic text ("حمى", "سعال شديد"), but the disease model
consumes coded evidence (``HEALIX_SYMPTOM_####``). This layer maps the former to
the latter deterministically, so ``symptom_presence`` can be populated for the
Feature Schema v2 encoder.

Pure domain logic — no FastAPI, no network, no file I/O. The Arabic lexicon
(``HEALIX_SYMPTOM_#### -> [synonyms]``) is INJECTED at construction, exactly like
``FeatureEncoder(schema=...)`` and ``FeatureValidator(rules=...)``. Loading and
validating the dictionary is the infrastructure loader's job.

Open-world semantics are respected: this layer only asserts what it positively
matched (present or negated). Symptoms it could not code are surfaced in
``unmatched`` for observability — never silently dropped, and never zero-filled
(zero-filling would fabricate "confirmed absent" from "not mentioned", the exact
closed-vs-open-world hazard flagged in ``app/ml/evidence_parser.py``). Deciding
how the encoder treats un-mentioned symptoms is a separate, explicit step.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

# Arabic orthographic normalisation ------------------------------------------
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF_VARIANTS = str.maketrans({
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "آ": "ا",  # آ -> ا
    "ى": "ي",  # ى -> ي
    "ؤ": "و",  # ؤ -> و
    "ئ": "ي",  # ئ -> ي
    "ة": "ه",  # ة -> ه
})

# Only trust a fuzzy hit when it is this close (short Arabic strings: typos only).
_FUZZY_CUTOFF = 0.86


def normalize_ar(text: str) -> str:
    """Deterministic Arabic normalisation: strip diacritics/tatweel, unify alef/
    ya/ta-marbuta variants, collapse whitespace, lowercase any Latin."""
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF_VARIANTS)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


@dataclass(frozen=True)
class SymptomCoding:
    """One resolved mapping: an interview symptom -> a HEALIX code."""

    healix_id: str
    source_text: str      # the interview symptom text that matched
    matched_term: str     # the lexicon term it matched (normalised)
    negated: bool
    fuzzy: bool = False    # True when resolved by the fuzzy fallback, not exact


@dataclass
class CodingResult:
    """Outcome of coding one interview's symptom list."""

    codings: List[SymptomCoding] = field(default_factory=list)
    # HEALIX_SYMPTOM_#### -> True (present) / False (explicitly negated).
    # Only ids we actually matched appear here; the rest stay UNKNOWN by omission.
    presence: Dict[str, bool] = field(default_factory=dict)
    # Interview symptom texts we could not map to any code (observability).
    unmatched: List[str] = field(default_factory=list)


class SymptomCoder:
    """Maps free Arabic interview symptoms to HEALIX symptom codes."""

    def __init__(self, lexicon: Dict[str, List[str]]):
        """``lexicon``: HEALIX_SYMPTOM_#### -> list of Arabic synonym phrases."""
        # Reverse index: normalised phrase -> healix id. Longer phrases win, so a
        # specific term ("ضيق تنفس") beats a generic substring of it.
        self._phrase_to_id: Dict[str, str] = {}
        for healix_id, phrases in lexicon.items():
            for phrase in phrases:
                norm = normalize_ar(phrase)
                if norm:
                    self._phrase_to_id.setdefault(norm, healix_id)
        self._phrases_by_len: List[str] = sorted(
            self._phrase_to_id, key=len, reverse=True
        )

    # ------------------------------------------------------------------
    def code(self, symptoms) -> CodingResult:
        """Code an iterable of objects exposing ``.text`` and ``.negated``."""
        result = CodingResult()
        for symptom in symptoms:
            healix_id, matched_term, fuzzy = self._match(symptom.text)
            if healix_id is None:
                result.unmatched.append(symptom.text)
                continue
            result.codings.append(SymptomCoding(
                healix_id=healix_id,
                source_text=symptom.text,
                matched_term=matched_term,
                negated=bool(symptom.negated),
                fuzzy=fuzzy,
            ))
            # A positive mention always wins over a negated one for the same code.
            present = not bool(symptom.negated)
            if present or healix_id not in result.presence:
                result.presence[healix_id] = present
        return result

    # ------------------------------------------------------------------
    def _match(self, text: str):
        """Return (healix_id, matched_term, is_fuzzy) or (None, None, False)."""
        norm = normalize_ar(text)
        if not norm:
            return None, None, False

        # 1) a lexicon phrase occurs inside the interview text, longest first so
        #    the most specific present term wins ("ضيق تنفس" beats a bare "تنفس").
        #    Only this direction — NOT "text inside a longer phrase", which would
        #    let a bare "سعال" wrongly resolve to "نوبات سعال".
        for phrase in self._phrases_by_len:
            if phrase in norm:
                return self._phrase_to_id[phrase], phrase, False

        # 2) conservative fuzzy fallback for typos only.
        best_phrase, best_ratio = None, 0.0
        for phrase in self._phrase_to_id:
            ratio = SequenceMatcher(None, norm, phrase).ratio()
            if ratio > best_ratio:
                best_phrase, best_ratio = phrase, ratio
        if best_phrase is not None and best_ratio >= _FUZZY_CUTOFF:
            return self._phrase_to_id[best_phrase], best_phrase, True

        return None, None, False
