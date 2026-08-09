"""Tests for the runtime symptom-coding layer (app/domain/symptom_coder.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.domain.symptom_coder import SymptomCoder, normalize_ar

_ONTOLOGY = Path(__file__).resolve().parent.parent / "app" / "dictionaries" / "symptom_ontology.json"


@dataclass
class _Sym:
    """Minimal stand-in for domain.conversation.Symptom (text + negated)."""
    text: str
    negated: bool = False


# --- normalisation ----------------------------------------------------------
def test_normalize_strips_diacritics_and_unifies_alef():
    assert normalize_ar("حُمَّى") == normalize_ar("حمى")
    assert normalize_ar("إسهال") == normalize_ar("اسهال")
    assert normalize_ar("سُعال  ") == "سعال"


# --- core matching with a small inline lexicon ------------------------------
@pytest.fixture
def coder():
    return SymptomCoder({
        "HEALIX_SYMPTOM_0001": ["حمى", "حرارة", "سخونة"],
        "HEALIX_SYMPTOM_0034": ["سعال", "كحة"],
        "HEALIX_SYMPTOM_0024": ["ضيق تنفس", "صعوبة تنفس"],
    })


def test_exact_and_substring_match(coder):
    res = coder.code([_Sym("حمى"), _Sym("عندي سعال جاف")])
    assert res.presence["HEALIX_SYMPTOM_0001"] is True
    assert res.presence["HEALIX_SYMPTOM_0034"] is True
    assert res.unmatched == []


def test_negation_is_carried_through(coder):
    res = coder.code([_Sym("حمى", negated=True)])
    assert res.presence["HEALIX_SYMPTOM_0001"] is False
    assert res.codings[0].negated is True


def test_positive_mention_wins_over_negation(coder):
    res = coder.code([_Sym("حمى", negated=True), _Sym("حرارة", negated=False)])
    assert res.presence["HEALIX_SYMPTOM_0001"] is True


def test_longer_phrase_wins(coder):
    # "ضيق تنفس" must resolve to the dyspnea code, not accidentally something else.
    res = coder.code([_Sym("عندي ضيق تنفس شديد")])
    assert res.presence == {"HEALIX_SYMPTOM_0024": True}


def test_unmatched_is_surfaced_not_dropped(coder):
    res = coder.code([_Sym("طفح جلدي")])
    assert res.presence == {}
    assert res.unmatched == ["طفح جلدي"]


def test_fuzzy_handles_a_typo(coder):
    res = coder.code([_Sym("حمي")])  # missing dots on ya -> normalises close to حمى
    assert res.presence.get("HEALIX_SYMPTOM_0001") is True


# --- integration against the real seeded lexicon ----------------------------
@pytest.mark.skipif(not _ONTOLOGY.exists(), reason="ontology not generated")
def test_real_lexicon_codes_a_realistic_interview():
    data = json.loads(_ONTOLOGY.read_text(encoding="utf-8"))
    lexicon = {hid: info["ar"] for hid, info in data["symptoms"].items() if info["ar"]}
    coder = SymptomCoder(lexicon)

    symptoms = [
        _Sym("حمى"),
        _Sym("سعال"),
        _Sym("ألم بالصدر"),
        _Sym("إسهال", negated=True),
        _Sym("شي ما إلو ترميز بعد"),  # deliberately uncodable
    ]
    res = coder.code(symptoms)

    assert res.presence["HEALIX_SYMPTOM_0001"] is True      # حمى
    assert res.presence["HEALIX_SYMPTOM_0034"] is True      # سعال
    assert res.presence["HEALIX_SYMPTOM_0078"] is True      # ألم بالصدر
    assert res.presence["HEALIX_SYMPTOM_0009"] is False     # إسهال (negated)
    assert "شي ما إلو ترميز بعد" in res.unmatched
