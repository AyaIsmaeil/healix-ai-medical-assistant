import pytest

import llm_client
from llm_client import set_provider
from nodes.ask_followup import ask_followup


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("ask_followup must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _state(*, next_question, thread_id="thread-1"):
    return {"thread_id": thread_id, "next_question": next_question}


# --- node contract: partial state dict ----------------------------------------


def test_ask_followup_returns_only_messages_and_stage():
    result = ask_followup(_state(next_question="من متى بلش الوجع؟"))

    assert set(result) == {"messages", "stage"}


def test_ask_followup_sets_stage_to_followup():
    result = ask_followup(_state(next_question="من متى بلش الوجع؟"))

    assert result["stage"] == "followup"


def test_ask_followup_appends_one_assistant_message():
    result = ask_followup(_state(next_question="من متى بلش الوجع؟"))

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"


def test_ask_followup_does_not_call_the_llm():
    # The question was already generated upstream by assess_sufficiency —
    # nothing here needs to vary per-turn or call out to a provider.
    set_provider(_ExplodingProvider())

    ask_followup(_state(next_question="من متى بلش الوجع؟"))  # must not raise


# --- the question reaches messages verbatim -------------------------------------


def test_ask_followup_sends_next_question_verbatim():
    question = "وين مكان الصداع بالضبط، وهل هو نابض ولا ضاغط؟"

    result = ask_followup(_state(next_question=question))

    assert result["messages"][0]["content"] == question


def test_ask_followup_does_not_alter_or_translate_the_question():
    # No LLM call means no risk of paraphrasing/re-wording — this pins
    # down that guarantee at the assertion level too, not just via the
    # "no LLM call" test above.
    question = "سؤال متابعة محدد جدًا بصياغة معينة لازم تبقى كما هي"

    result = ask_followup(_state(next_question=question))

    assert result["messages"][0]["content"] is question
