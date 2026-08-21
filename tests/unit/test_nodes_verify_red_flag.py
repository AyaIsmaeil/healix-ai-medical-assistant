import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS
from nodes.verify_red_flag import verify_red_flag
from schemas.red_flags import VerificationQuestion


class FakeProvider:
    name = "fake"

    def __init__(self, *, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _question_response(text="بتحس بضيق تنفس هلق؟") -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"question": text}, ensure_ascii=False))


def _candidate(rule_id="acs_chest_pain", missing_any_of=("ضيق تنفس", "تعرق غزير")):
    return {
        "rule_id": rule_id,
        "category": "cardiac",
        "reason_ar": "سبب",
        "source": "مصدر",
        "matched_symptoms": ["ألم في الصدر"],
        "missing_any_of": list(missing_any_of),
    }


def _state(*, candidates=(), turn_count=0, thread_id="thread-1"):
    return {
        "thread_id": thread_id,
        "red_flag_candidates": list(candidates),
        "turn_count": turn_count,
    }


def test_asks_a_question_sourced_from_the_candidates_missing_any_of():
    provider = FakeProvider(responses=[_question_response("بتحس بضيق تنفس أو تعرق غزير؟")])
    set_provider(provider)

    result = verify_red_flag(_state(candidates=[_candidate()]))

    assert result["next_question"] == "بتحس بضيق تنفس أو تعرق غزير؟"
    prompt = provider.calls[0]["prompt"]
    assert "ضيق تنفس" in prompt
    assert "تعرق غزير" in prompt


def test_calls_the_llm_with_the_verification_schema_on_the_quality_tier():
    provider = FakeProvider(responses=[_question_response()])
    set_provider(provider)

    verify_red_flag(_state(candidates=[_candidate()]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is VerificationQuestion


def test_increments_the_shared_turn_count_by_one():
    set_provider(FakeProvider(responses=[_question_response()]))

    result = verify_red_flag(_state(candidates=[_candidate()], turn_count=2))

    assert result["turn_count"] == 3


def test_picks_the_first_candidate_when_more_than_one_is_unresolved():
    provider = FakeProvider(responses=[_question_response()])
    set_provider(provider)

    verify_red_flag(
        _state(
            candidates=[
                _candidate(rule_id="acs_chest_pain", missing_any_of=["ضيق تنفس"]),
                _candidate(rule_id="sepsis", missing_any_of=["تسارع في التنفس"]),
            ]
        )
    )

    prompt = provider.calls[0]["prompt"]
    assert "ضيق تنفس" in prompt
    assert "تسارع في التنفس" not in prompt


def test_no_candidates_fails_safe_without_calling_the_llm():
    provider = FakeProvider(responses=[])
    set_provider(provider)

    result = verify_red_flag(_state(candidates=[]))

    assert result["next_question"] is None
    assert provider.calls == []


# --- shared follow-up budget (MAX_FOLLOW_UP_QUESTIONS) ------------------------


def test_does_not_ask_again_once_the_shared_budget_is_spent():
    provider = FakeProvider(responses=[])
    set_provider(provider)

    result = verify_red_flag(
        _state(candidates=[_candidate()], turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    assert result["next_question"] is None
    assert provider.calls == []


def test_marks_information_limited_once_the_shared_budget_is_spent():
    set_provider(FakeProvider(responses=[]))

    result = verify_red_flag(
        _state(candidates=[_candidate()], turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    assert result["information_limited"] is True


def test_does_not_increment_turn_count_past_the_ceiling():
    set_provider(FakeProvider(responses=[]))

    result = verify_red_flag(
        _state(candidates=[_candidate()], turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    assert "turn_count" not in result
