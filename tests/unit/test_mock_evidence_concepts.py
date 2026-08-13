"""اختبارات mock provider لعقد اختيار مفاهيم الأدلة."""

import json

from app.llm.mock_provider import MockLLMProvider
from app.prompts.evidence_concept_extraction_builder import EvidenceConceptExtractionPromptBuilder


def test_mock_returns_evidence_concepts_json():
    builder = EvidenceConceptExtractionPromptBuilder(
        concepts=["fever", "cough", "dyspnea"]
    )
    user_prompt = builder.extraction_prompt(["عندي حرارة وسعال"])
    completion = MockLLMProvider().generate(builder.system_prompt(), user_prompt)
    payload = json.loads(completion.text)
    assert "fever" in payload["concepts"]
    assert "cough" in payload["concepts"]
