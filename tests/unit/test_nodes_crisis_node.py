import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.crisis_node import crisis_node
from prompts.base import build_prompt
from schemas.crisis import CrisisResponse
from support_lines import SupportLine


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def crisis_message_response(message: str) -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"message": message}, ensure_ascii=False))


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


ACKNOWLEDGMENT = "فهمتك، وهاد الشي أكبر من إمكانياتي كمساعد."


# --- node contract: partial state dict, quality tier -------------------------


def test_crisis_node_returns_only_messages_stage_and_thread_outcome():
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    assert set(result) == {"messages", "stage", "thread_outcome"}


def test_crisis_node_sets_stage_to_crisis():
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    assert result["stage"] == "crisis"


def test_crisis_node_sets_thread_outcome_to_crisis():
    # CLAUDE.md > Non-negotiable safety rule 13 — sticky, unlike stage,
    # so a later turn on this thread that escalates nothing new is routed
    # to reiterate_terminal_outcome instead of normal symptom triage.
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    assert result["thread_outcome"] == "crisis"


def test_crisis_node_appends_one_assistant_message():
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"


def test_crisis_node_calls_the_llm_with_the_configured_schema_and_model():
    provider = FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)])
    set_provider(provider)

    crisis_node(_state([_user("بدي موت")]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is CrisisResponse


def test_crisis_node_prompts_the_llm_with_the_latest_patient_message_only():
    provider = FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)])
    set_provider(provider)

    crisis_node(
        _state(
            [_user("رسالة قديمة"), {"role": "assistant", "content": "سؤال"}, _user("بدي موت")]
        )
    )

    prompt = provider.calls[0]["prompt"]
    assert "بدي موت" in prompt
    assert "رسالة قديمة" not in prompt


# --- no diagnostic content ----------------------------------------------------


def test_crisis_node_never_touches_diagnostic_state_fields():
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    for field in (
        "symptoms",
        "negated_symptoms",
        "red_flags",
        "severity",
        "candidate_diseases",
        "diagnosis",
        "specialty",
        "reports",
    ):
        assert field not in result


def test_crisis_node_schema_has_no_field_for_a_phone_number():
    # Structural, not a prompt instruction: the model has nowhere to put a
    # number even if it tried. See schemas.crisis.CrisisResponse.
    assert set(CrisisResponse.model_fields) == {"message"}


# --- prompt content: the things the LLM is told never to do -------------------
#
# A FakeProvider can't prove a live model obeys these — this proves the
# instructions asking it to are actually present in what gets sent, the
# same way tests/unit/test_prompts_base.py checks known preamble content.


def test_crisis_node_prompt_forbids_reassurance():
    prompt = build_prompt("crisis_node", message="بدي موت")
    assert "لا تطمئنه" in prompt


def test_crisis_node_prompt_forbids_analysis_and_diagnosis():
    prompt = build_prompt("crisis_node", message="بدي موت")
    assert "لا تحلل ولا تفسر ولا تشخّص" in prompt


def test_crisis_node_prompt_forbids_followup_symptom_questions():
    prompt = build_prompt("crisis_node", message="بدي موت")
    assert "لا تسأله أي سؤال متابعة عن الأعراض" in prompt


def test_crisis_node_prompt_forbids_coping_techniques():
    prompt = build_prompt("crisis_node", message="بدي موت")
    assert "لا تقترح عليه أي أسلوب للتهدئة أو المواجهة" in prompt


def test_crisis_node_prompt_forbids_the_model_from_naming_a_phone_number():
    prompt = build_prompt("crisis_node", message="بدي موت")
    assert "لا تذكر أي رقم هاتف" in prompt


def test_crisis_node_prompt_directs_toward_a_professional_or_trusted_person():
    prompt = build_prompt("crisis_node", message="بدي موت")
    assert "مختص نفسي" in prompt
    assert "شخص يثق فيه" in prompt


# --- support-line numbers: config-driven, never the model's to write ----------


def test_crisis_node_appends_verified_numbers_when_configured(monkeypatch):
    # support_line_text() (and the verified_support_lines() it calls) now
    # lives in nodes/_shared.py, shared with reiterate_terminal_outcome —
    # monkeypatch it where the real import actually is.
    monkeypatch.setattr(
        "nodes._shared.verified_support_lines",
        lambda: (SupportLine(name="خط الدعم الوطني", phone="123-456", is_placeholder=False),),
    )
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    reply = result["messages"][0]["content"]
    assert ACKNOWLEDGMENT in reply
    assert "خط الدعم الوطني" in reply
    assert "123-456" in reply


def test_crisis_node_falls_back_to_a_generic_referral_when_no_numbers_are_configured(monkeypatch):
    # The real, shipped state of support_lines.py today — no monkeypatch
    # needed, but made explicit here so the test's intent doesn't depend
    # on remembering that fact from another file.
    monkeypatch.setattr("nodes._shared.verified_support_lines", lambda: ())
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    reply = result["messages"][0]["content"]
    assert ACKNOWLEDGMENT in reply
    assert "طبيب" in reply or "مستشفى" in reply


def test_crisis_node_never_invents_a_number_when_none_are_configured(monkeypatch):
    monkeypatch.setattr("nodes._shared.verified_support_lines", lambda: ())
    set_provider(FakeProvider(responses=[crisis_message_response(ACKNOWLEDGMENT)]))

    result = crisis_node(_state([_user("بدي موت")]))

    reply = result["messages"][0]["content"]
    assert not any(char.isdigit() for char in reply)
