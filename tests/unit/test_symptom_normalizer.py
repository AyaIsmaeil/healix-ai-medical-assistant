"""اختبارات SymptomNormalizer — تطبيع الأعراض وإسقاط الشظايا."""

from app.domain.conversation import Symptom
from app.domain.symptom_normalizer import (
    is_descriptor_fragment,
    normalize_symptom_text,
    normalize_symptoms_for_storage,
)


def test_descriptor_fragments_rejected():
    assert is_descriptor_fragment("يزداد")
    assert is_descriptor_fragment("اليمين")
    assert normalize_symptom_text("يزداد") is None


def test_dyspnea_canonical():
    assert normalize_symptom_text("ضيق نفس") == "ضيق تنفس"


def test_chest_pressure_canonical():
    assert normalize_symptom_text("ضغط على الصدر") == "ألم صدر"


def test_normalize_deduplicates_symptoms():
    raw = [
        Symptom(text="ضيق نفس", negated=False, confidence=0.9),
        Symptom(text="ضيق تنفس", negated=False, confidence=0.8),
        Symptom(text="يزداد", negated=False, confidence=0.7),
    ]
    out = normalize_symptoms_for_storage(raw)
    texts = [s.text for s in out if not s.negated]
    assert texts == ["ضيق تنفس"]
