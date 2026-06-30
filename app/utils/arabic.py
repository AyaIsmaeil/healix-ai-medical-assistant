"""
Arabic text normalization for Healix symptom extraction.

This module is the SINGLE source of truth for text normalization. It is used
both at training time (training/preprocess.py) and at inference time
(app/ml + app/services). Training and serving must apply identical
normalization, otherwise the model sees a different text distribution in
production than it was trained on.

Design choice: *light* normalization. MARBERTv2 was pretrained on dialectal
Arabic, so we only remove orthographic noise (diacritics, tatweel, character
elongation) and unify a few interchangeable letters. We intentionally do NOT
strip hamza seats (ؤ/ئ) or map ة→ه, since that can change meaning.
"""

from __future__ import annotations

import re

import pyarabic.araby as araby

# Alef variants that should all collapse to a bare alef (ا).
_ALEF_VARIANTS = re.compile(r"[إأآٱ]")  # إ أ آ ٱ
# Runs of the same character repeated 3+ times (dialectal elongation),
# e.g. "حراااارة" -> "حرارة". Reduced to a single occurrence.
_ELONGATION = re.compile(r"(.)\1{2,}")
# Any run of whitespace.
_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Normalize a single Arabic string.

    Steps (in order):
        1. Coerce to string and strip surrounding whitespace.
        2. Remove tashkeel (diacritics) and tatweel (ـ).
        3. Unify alef variants (إ أ آ ٱ) -> ا and alef-maqsura (ى) -> ي.
        4. Collapse character elongations of length 3+ to a single character.
        5. Collapse internal whitespace to single spaces.

    Returns an empty string for null/empty input.
    """
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)

    text = _ALEF_VARIANTS.sub("ا", text)  # -> ا
    text = text.replace("ى", "ي")    # ى -> ي

    text = _ELONGATION.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()

    return text
