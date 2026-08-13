"""اختبارات RuleBasedEvidenceConceptExtractor — استخراج حتمي لمفاهيم الأدلة."""

from app.infrastructure.dictionary_loader import DictionaryLoader
from app.services.rule_based_evidence_concept_extractor import RuleBasedEvidenceConceptExtractor


def _extractor() -> RuleBasedEvidenceConceptExtractor:
    return RuleBasedEvidenceConceptExtractor.from_dict(
        DictionaryLoader.load_symptom_evidence_map()
    )


def test_fever_from_arabic_message():
    concepts = _extractor().extract(["عندي حرارة وحمى من يومين"], [])
    assert "fever" in concepts


def test_negated_fever_not_selected():
    concepts = _extractor().extract(["ما عندي حرارة"], [])
    assert "fever" not in concepts


def test_symptom_names_used_when_messages_empty():
    concepts = _extractor().extract([], ["ضيق تنفس"])
    assert "dyspnea" in concepts


def test_chest_pain_concept():
    concepts = _extractor().extract(["ألم صدر مع ضيق نفس"], [])
    assert "chest_pain" in concepts
    assert "dyspnea" in concepts
