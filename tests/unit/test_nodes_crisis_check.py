import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.crisis_check import crisis_check
from rules.crisis import CrisisResult
from schemas.crisis import CrisisCheckResult


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def crisis_response(is_crisis: bool, reasoning: str | None = "r") -> _ProviderResponse:
    payload = {"is_crisis": is_crisis, "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _state(messages, thread_id="thread-1"):
    return {"thread_id": thread_id, "messages": messages}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


# Ordinary, non-crisis Arabic text — CRISIS_PATTERNS is placeholder data
# (see rules/crisis.py), so this is guaranteed never to match the rule
# layer. Using it lets a test exercise the real, unmocked rule layer while
# staying sure of what it will return.
ORDINARY_MESSAGE = "عندي صداع من يومين وتعبان شوي"


# latest_user_message extraction is now shared (nodes/_shared.py) and
# tested once, directly, in tests/unit/test_nodes_shared.py.


# --- OR logic: rule layer mocked --------------------------------------------


def test_crisis_check_fires_when_only_the_rule_layer_matches(monkeypatch):
    monkeypatch.setattr(
        "nodes.crisis_check.detect_crisis",
        lambda message: CrisisResult(categories=("suicidal_ideation",)),
    )
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    result = crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert result == {"is_crisis": True}


def test_crisis_check_fires_when_both_layers_match(monkeypatch):
    monkeypatch.setattr(
        "nodes.crisis_check.detect_crisis",
        lambda message: CrisisResult(categories=("self_harm_intent",)),
    )
    provider = FakeProvider(responses=[crisis_response(True)])
    set_provider(provider)

    result = crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert result == {"is_crisis": True}


def test_crisis_check_does_not_fire_when_neither_layer_matches(monkeypatch):
    monkeypatch.setattr(
        "nodes.crisis_check.detect_crisis", lambda message: CrisisResult(categories=())
    )
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    result = crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert result == {"is_crisis": False}


# --- the node still works when only the LLM layer fires ---------------------
# No mocking of detect_crisis here: real rule layer, real (placeholder,
# always-inert) patterns, exercising the actual OR against a genuine miss
# on the rule side rather than a stand-in for one.


def test_crisis_check_fires_when_only_the_llm_layer_matches():
    provider = FakeProvider(responses=[crisis_response(True, reasoning="إشارة يأس شديد")])
    set_provider(provider)

    result = crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert result == {"is_crisis": True}


def test_crisis_check_returns_false_via_the_real_rule_layer_when_llm_also_misses():
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    result = crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert result == {"is_crisis": False}


# --- node contract: partial state dict, quality tier -------------------------


def test_crisis_check_returns_only_the_is_crisis_key():
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    result = crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert set(result) == {"is_crisis"}


def test_crisis_check_calls_the_llm_on_the_quality_tier():
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is CrisisCheckResult


def test_crisis_check_prompt_bounds_context_to_one_prior_turn():
    # Same scope as nodes/extract_symptoms.py's identical fix: the latest
    # message plus the single immediately-preceding assistant message,
    # never the full history. An older message from two turns back must
    # not leak in just because it's in state["messages"].
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    crisis_check(_state([_user("رسالة قديمة"), _assistant("سؤال"), _user("رسالة جديدة")]))

    prompt = provider.calls[0]["prompt"]
    assert "رسالة جديدة" in prompt
    assert "سؤال" in prompt
    assert "رسالة قديمة" not in prompt


# --- conversational context: previous_question (LLM layer only) ----------------
#
# Same fix as nodes/extract_symptoms.py, applied here after that one was
# verified against real calls: a terse or ambiguous reply can only be
# correctly judged in light of what was just asked. previous_assistant_
# message's own logic is tested once, directly, in
# tests/unit/test_nodes_shared.py — these tests only confirm crisis_check
# actually wires it into the LLM prompt, and that the deterministic rule
# layer is completely unaffected either way.


def test_prompt_includes_the_previous_assistant_question_when_one_exists():
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    crisis_check(
        _state([_user("عندي صداع"), _assistant("منذ متى بدأ هذا الصداع؟"), _user("من الصبح")])
    )

    prompt = provider.calls[0]["prompt"]
    assert "منذ متى بدأ هذا الصداع؟" in prompt


def test_prompt_uses_the_placeholder_when_there_is_no_previous_question():
    # First turn of a thread — no assistant message exists yet. Must not
    # raise, and must not leave an unresolved $previous_question token.
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    crisis_check(_state([_user(ORDINARY_MESSAGE)]))

    prompt = provider.calls[0]["prompt"]
    assert "لا يوجد" in prompt


def test_added_context_does_not_reach_the_deterministic_rule_layer(monkeypatch):
    # The rule layer must see exactly the raw message, nothing appended
    # or altered — this is the "must not weaken the rule layer" guarantee,
    # checked directly rather than just asserted in a docstring.
    seen_by_rules = []
    monkeypatch.setattr(
        "nodes.crisis_check.detect_crisis",
        lambda message: seen_by_rules.append(message) or CrisisResult(categories=()),
    )
    provider = FakeProvider(responses=[crisis_response(False)])
    set_provider(provider)

    crisis_check(
        _state([_user("عندي صداع"), _assistant("منذ متى بدأ هذا الصداع؟"), _user("من الصبح")])
    )

    assert seen_by_rules == ["من الصبح"]


# --- audit: both verdicts logged separately ----------------------------------


def test_crisis_check_audit_logs_both_verdicts_separately(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "nodes.crisis_check.log_crisis_detection", lambda **kwargs: logged.append(kwargs)
    )
    monkeypatch.setattr(
        "nodes.crisis_check.detect_crisis",
        lambda message: CrisisResult(categories=("hopelessness_severe",)),
    )
    provider = FakeProvider(responses=[crisis_response(False, reasoning=None)])
    set_provider(provider)

    crisis_check(_state([_user(ORDINARY_MESSAGE)], thread_id="thread-42"))

    assert len(logged) == 1
    record = logged[0]
    assert record["thread_id"] == "thread-42"
    assert record["rule_matched"] is True
    assert record["rule_categories"] == ["hopelessness_severe"]
    assert record["llm_matched"] is False
    assert record["llm_reasoning"] is None
    assert record["combined"] is True
