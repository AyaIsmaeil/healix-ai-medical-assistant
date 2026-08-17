import pytest

import llm_client
from llm_client import set_provider
from nodes.reiterate_terminal_outcome import reiterate_terminal_outcome
from support_lines import SupportLine


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("reiterate_terminal_outcome must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _state(*, thread_outcome, thread_id="thread-1"):
    return {"thread_id": thread_id, "thread_outcome": thread_outcome}


# --- node contract: partial state dict, no LLM call ------------------------------


def test_reiterate_terminal_outcome_returns_only_messages_and_stage():
    result = reiterate_terminal_outcome(_state(thread_outcome="emergency"))

    assert set(result) == {"messages", "stage"}


def test_reiterate_terminal_outcome_appends_exactly_one_assistant_message():
    result = reiterate_terminal_outcome(_state(thread_outcome="emergency"))

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"


def test_reiterate_terminal_outcome_does_not_call_the_llm():
    set_provider(_ExplodingProvider())

    reiterate_terminal_outcome(_state(thread_outcome="crisis"))  # must not raise
    reiterate_terminal_outcome(_state(thread_outcome="emergency"))  # must not raise


# --- emergency case ---------------------------------------------------------------


def test_emergency_case_sets_stage_to_emergency():
    result = reiterate_terminal_outcome(_state(thread_outcome="emergency"))

    assert result["stage"] == "emergency"


def test_emergency_case_directs_to_the_er():
    reply = reiterate_terminal_outcome(_state(thread_outcome="emergency"))["messages"][0]["content"]

    assert "طوارئ" in reply or "الإسعاف" in reply


def test_emergency_case_makes_clear_this_is_a_reminder_not_a_fresh_assessment():
    reply = reiterate_terminal_outcome(_state(thread_outcome="emergency"))["messages"][0]["content"]

    assert "قبل شوي" in reply


# --- crisis case --------------------------------------------------------------------


def test_crisis_case_sets_stage_to_crisis():
    result = reiterate_terminal_outcome(_state(thread_outcome="crisis"))

    assert result["stage"] == "crisis"


def test_crisis_case_appends_verified_support_line_numbers_when_configured(monkeypatch):
    monkeypatch.setattr(
        "nodes._shared.verified_support_lines",
        lambda: (SupportLine(name="خط الدعم الوطني", phone="123-456", is_placeholder=False),),
    )

    reply = reiterate_terminal_outcome(_state(thread_outcome="crisis"))["messages"][0]["content"]

    assert "خط الدعم الوطني" in reply
    assert "123-456" in reply


def test_crisis_case_falls_back_to_generic_referral_when_no_numbers_configured(monkeypatch):
    monkeypatch.setattr("nodes._shared.verified_support_lines", lambda: ())

    reply = reiterate_terminal_outcome(_state(thread_outcome="crisis"))["messages"][0]["content"]

    assert "طبيب" in reply or "مستشفى" in reply


def test_crisis_case_never_invents_a_number_when_none_configured(monkeypatch):
    monkeypatch.setattr("nodes._shared.verified_support_lines", lambda: ())

    reply = reiterate_terminal_outcome(_state(thread_outcome="crisis"))["messages"][0]["content"]

    assert not any(char.isdigit() for char in reply)


# --- the two cases produce genuinely different content ----------------------------


def test_crisis_and_emergency_replies_differ():
    crisis_reply = reiterate_terminal_outcome(_state(thread_outcome="crisis"))["messages"][0][
        "content"
    ]
    emergency_reply = reiterate_terminal_outcome(_state(thread_outcome="emergency"))["messages"][0][
        "content"
    ]

    assert crisis_reply != emergency_reply


# --- defensive: must not crash on an unreached state -------------------------------


def test_missing_thread_outcome_still_produces_a_safe_response_rather_than_crashing():
    # Not expected in practice (graph.py's routing only reaches this node
    # when thread_outcome is already set), but must not raise or return
    # something misleading — the safer default is the emergency directive,
    # not a guess at which one applies.
    result = reiterate_terminal_outcome(_state(thread_outcome=None))

    assert result["stage"] == "emergency"
    assert result["messages"][0]["content"]
