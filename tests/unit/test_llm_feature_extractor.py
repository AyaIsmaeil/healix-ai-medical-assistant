"""اختبارات وحدة لـLLMFeatureExtractor — يشمل تخطّي الاستدعاء عند عدم الحاجة."""

import json

from app.domain.ports import Completion
from app.prompts.assessment_extraction_builder import AssessmentExtractionPromptBuilder
from app.services.llm_feature_extractor import LLMFeatureExtractor


class CountingProvider:
    """يُرجع رداً ثابتاً ويعدّ عدد الاستدعاءات."""

    name = "counting"

    def __init__(self, response: dict):
        self._response = response
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        self.calls += 1
        return Completion(json.dumps(self._response, ensure_ascii=False), model="counting")


class RaisingProvider:
    name = "raising"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        raise RuntimeError("تعذّر الاتصال")


def _extractor(provider):
    return LLMFeatureExtractor(provider=provider, prompt_builder=AssessmentExtractionPromptBuilder())


def test_skips_llm_call_when_nothing_unresolved():
    provider = CountingProvider({"age": 99})
    result = _extractor(provider).extract(["أي نص"], unresolved_fields=[])

    assert provider.calls == 0
    assert result.age is None  # نتيجة فارغة، لم يُستدعَ المزوّد إطلاقاً


def test_calls_llm_when_fields_unresolved():
    provider = CountingProvider({"age": 34, "gender": "male", "smoking": None,
                                 "temperature": None, "duration": None, "severity": None})
    result = _extractor(provider).extract(["عندي صداع"], unresolved_fields=["age", "gender"])

    assert provider.calls == 1
    assert result.age == 34
    assert result.gender == "male"


def test_provider_exception_degrades_gracefully_not_raises():
    result = _extractor(RaisingProvider()).extract(["نص"], unresolved_fields=["age"])
    assert result.age is None  # فشل ناعم، لا استثناء يصعد للمستدعي


def test_malformed_llm_output_degrades_gracefully():
    class BadProvider:
        name = "bad"

        def generate(self, system_prompt, user_prompt):
            return Completion("ليس JSON إطلاقاً", model="bad")

    result = _extractor(BadProvider()).extract(["نص"], unresolved_fields=["age"])
    assert result.age is None
