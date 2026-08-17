import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS, assess_sufficiency
from nodes.extract_symptoms import extract_symptoms
from schemas.sufficiency import SufficiencyAssessment


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("assess_sufficiency must not call the LLM past the ceiling")


def sufficiency_response(
    is_sufficient: bool, next_question: str | None = None, reasoning: str | None = "r"
) -> _ProviderResponse:
    payload = {
        "is_sufficient": is_sufficient,
        "next_question": next_question,
        "reasoning": reasoning,
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _symptom(name, **fields):
    return {"name": name, **fields}


def _state(
    *,
    symptoms=(),
    negated_symptoms=(),
    unmatched_mentions=(),
    medical_record_summary="",
    turn_count=0,
    thread_id="thread-1",
):
    return {
        "thread_id": thread_id,
        "symptoms": [dict(s) for s in symptoms],
        "negated_symptoms": [dict(s) for s in negated_symptoms],
        "unmatched_mentions": list(unmatched_mentions),
        "medical_record_summary": medical_record_summary,
        "turn_count": turn_count,
    }


# --- sufficient on first pass ------------------------------------------------


def test_sufficient_on_first_pass_with_a_clear_symptom_picture():
    provider = FakeProvider(responses=[sufficiency_response(True)])
    set_provider(provider)

    result = assess_sufficiency(
        _state(
            symptoms=[
                _symptom("حمى", duration="يومين", severity="moderate", onset="sudden"),
                _symptom("سعال", duration="يومين"),
            ],
            turn_count=0,
        )
    )

    assert result == {
        "is_sufficient": True,
        "next_question": None,
        "information_limited": False,
    }


def test_sufficient_verdict_does_not_touch_turn_count():
    provider = FakeProvider(responses=[sufficiency_response(True)])
    set_provider(provider)

    result = assess_sufficiency(_state(symptoms=[_symptom("حمى")], turn_count=2))

    assert "turn_count" not in result


# --- insufficient: a follow-up question is produced ---------------------------


def test_insufficient_triggers_a_follow_up_question():
    provider = FakeProvider(
        responses=[sufficiency_response(False, next_question="من متى بلش الوجع؟")]
    )
    set_provider(provider)

    result = assess_sufficiency(_state(symptoms=[_symptom("وجع راس")], turn_count=0))

    assert result == {
        "is_sufficient": False,
        "next_question": "من متى بلش الوجع؟",
        "information_limited": False,
        "turn_count": 1,
    }


def test_insufficient_increments_turn_count_from_its_current_value():
    provider = FakeProvider(
        responses=[sufficiency_response(False, next_question="سؤال متابعة")]
    )
    set_provider(provider)

    result = assess_sufficiency(_state(symptoms=[_symptom("وجع راس")], turn_count=3))

    assert result["turn_count"] == 4


# --- hard ceiling: force proceed-anyway, no LLM call ---------------------------


def test_ceiling_reached_forces_sufficient_and_flags_information_limited():
    set_provider(_ExplodingProvider())

    result = assess_sufficiency(
        _state(symptoms=[_symptom("وجع راس")], turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    assert result == {
        "is_sufficient": True,
        "next_question": None,
        "information_limited": True,
    }


def test_ceiling_reached_does_not_call_the_llm():
    provider = FakeProvider(responses=[])
    set_provider(provider)

    assess_sufficiency(
        _state(symptoms=[_symptom("وجع راس")], turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    assert provider.calls == []


def test_ceiling_reached_does_not_increment_turn_count_further():
    set_provider(_ExplodingProvider())

    result = assess_sufficiency(
        _state(symptoms=[_symptom("وجع راس")], turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    assert "turn_count" not in result


def test_a_turn_count_past_the_ceiling_also_forces_proceed_anyway():
    # Not expected in practice (this node is turn_count's sole writer and
    # increments by exactly one), but must not require exact equality to
    # behave safely.
    set_provider(_ExplodingProvider())

    result = assess_sufficiency(
        _state(symptoms=[_symptom("وجع راس")], turn_count=MAX_FOLLOW_UP_QUESTIONS + 5)
    )

    assert result["is_sufficient"] is True
    assert result["information_limited"] is True


# --- node contract: partial state dict, quality tier -----------------------------


def test_assess_sufficiency_calls_the_llm_on_the_quality_tier_with_its_schema():
    provider = FakeProvider(responses=[sufficiency_response(True)])
    set_provider(provider)

    assess_sufficiency(_state(symptoms=[_symptom("حمى")]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is SufficiencyAssessment


def test_negated_symptoms_and_unmatched_mentions_reach_the_prompt():
    provider = FakeProvider(responses=[sufficiency_response(True)])
    set_provider(provider)

    assess_sufficiency(
        _state(
            symptoms=[_symptom("حمى")],
            negated_symptoms=[_symptom("سعال")],
            unmatched_mentions=["طنين بالأذن"],
        )
    )

    prompt = provider.calls[0]["prompt"]
    assert "سعال" in prompt
    assert "طنين بالأذن" in prompt


# --- raw_mention reaches the prompt, not just the canonical name ----------------
#
# The regression this guards: _format_symptom used to forward only
# duration/severity/onset, silently dropping raw_mention — the field
# where a patient's own elaboration on an already-named symptom actually
# lands (schemas.symptoms.ExtractedSymptom.raw_mention). Without this,
# assess_sufficiency could re-ask for detail the patient already gave.


def test_raw_mention_reaches_the_prompt_alongside_structured_fields():
    provider = FakeProvider(responses=[sufficiency_response(True)])
    set_provider(provider)

    assess_sufficiency(
        _state(
            symptoms=[
                _symptom("صداع", raw_mention="وجع من جهة وحدة نابض", duration="منذ يومين")
            ]
        )
    )

    prompt = provider.calls[0]["prompt"]
    assert "وجع من جهة وحدة نابض" in prompt


# --- real end-to-end: extract_symptoms -> assess_sufficiency -------------------


def _extraction_response(symptoms) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _extraction_response_with_raw_mention(name: str, raw_mention: str) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name, "raw_mention": raw_mention}],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def test_real_end_to_end_an_ambiguous_one_symptom_message_is_ruled_insufficient():
    # "عندي صداع" (I have a headache) alone, no duration/severity/onset — a
    # single vague symptom is exactly the ambiguous case CLAUDE.md > Testing
    # says must produce a clarifying question, not a guess.
    set_provider(
        FakeProvider(
            responses=[
                _extraction_response(["صداع"]),
                sufficiency_response(
                    False, next_question="الصداع من متى بلش، وهل هو من جهة وحدة؟"
                ),
            ]
        )
    )

    state = {
        "thread_id": "t1",
        "messages": [{"role": "user", "content": "عندي صداع"}],
        "medical_record_summary": "",
        "turn_count": 0,
    }
    state.update(extract_symptoms(state))

    result = assess_sufficiency(state)

    assert result["is_sufficient"] is False
    assert result["next_question"] == "الصداع من متى بلش، وهل هو من جهة وحدة؟"
    assert result["turn_count"] == 1
    assert result["information_limited"] is False


def test_real_end_to_end_extracted_raw_mention_reaches_assess_sufficiencys_prompt():
    # extract_symptoms attaches the patient's own elaboration on صداع as
    # raw_mention, not as a new unmatched_mentions entry (it's detail
    # about an already-named symptom, not a separate one) — this proves
    # that detail is genuinely visible to assess_sufficiency's own LLM
    # call, not just present somewhere in state.
    provider = FakeProvider(
        responses=[
            _extraction_response_with_raw_mention("صداع", "وجع من جهة وحدة نابض"),
            sufficiency_response(True),
        ]
    )
    set_provider(provider)

    state = {
        "thread_id": "t1",
        "messages": [{"role": "user", "content": "عندي صداع، وجع من جهة وحدة نابض"}],
        "medical_record_summary": "",
        "turn_count": 0,
    }
    state.update(extract_symptoms(state))
    assert state["symptoms"][0]["raw_mention"] == "وجع من جهة وحدة نابض"

    assess_sufficiency(state)

    sufficiency_prompt = provider.calls[1]["prompt"]
    assert "وجع من جهة وحدة نابض" in sufficiency_prompt
