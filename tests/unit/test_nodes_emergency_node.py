import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.check_red_flags import check_red_flags
from nodes.emergency_node import emergency_node
from nodes.extract_symptoms import extract_symptoms
from rules.crisis import normalize


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("emergency_node must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _red_flag(id_, reason="سبب"):
    return {"id": id_, "reason": reason}


def _state(*, red_flags=(), thread_id="thread-1"):
    return {"thread_id": thread_id, "red_flags": list(red_flags)}


# --- node contract: partial state dict ----------------------------------------


def test_emergency_node_returns_only_messages_stage_thread_outcome_and_severity():
    result = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))

    assert set(result) == {"messages", "stage", "thread_outcome", "severity"}


def test_emergency_node_sets_stage_to_emergency():
    result = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))

    assert result["stage"] == "emergency"


def test_emergency_node_sets_thread_outcome_to_emergency():
    # CLAUDE.md > Non-negotiable safety rule 13 — sticky, unlike stage,
    # so a later turn on this thread that escalates nothing new is routed
    # to reiterate_terminal_outcome instead of normal symptom triage.
    result = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))

    assert result["thread_outcome"] == "emergency"


def test_emergency_node_sets_severity_to_emergency():
    # A red flag firing IS an emergency-level severity judgment by
    # definition (CLAUDE.md > State) — the one place state["severity"]
    # (previously dormant) actually gets wired up.
    result = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))

    assert result["severity"] == "emergency"


def test_emergency_node_appends_one_assistant_message():
    result = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"


def test_emergency_node_does_not_call_the_llm():
    # No LLM call at all (unlike crisis_node) — nothing here needs to
    # vary per-turn. Set a provider that fails the test if it's ever
    # asked to generate anything.
    set_provider(_ExplodingProvider())

    emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))  # must not raise


# --- no diagnostic content ------------------------------------------------------


def test_emergency_node_never_touches_diagnostic_state_fields():
    # severity and thread_outcome are deliberately touched (see the two
    # tests above) — everything else stays untouched.
    result = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))

    for field in (
        "symptoms",
        "negated_symptoms",
        "red_flags",
        "candidate_diseases",
        "diagnosis",
        "specialty",
        "reports",
    ):
        assert field not in result


def test_emergency_node_message_excludes_the_clinical_reason_text():
    # A red flag's "reason" is doctor-facing (CLAUDE.md: it reads as a
    # clinical explanation bordering on diagnosis) — it must never leak
    # into the patient-facing message this node produces.
    clinical_reason = (
        "ألم الصدر المترافق مع أعراض إضافية قد يشير إلى متلازمة الشريان "
        "التاجي الحادة"
    )

    result = emergency_node(
        _state(red_flags=[_red_flag("acs_chest_pain", clinical_reason)])
    )

    reply = result["messages"][0]["content"]
    assert clinical_reason not in reply


def test_emergency_node_message_is_identical_regardless_of_which_rules_fired():
    # Deliberately not per-turn content: explaining *why* would drift
    # toward a clinical explanation. Same fixed message every time.
    reply_a = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))["messages"][0][
        "content"
    ]
    reply_b = emergency_node(
        _state(red_flags=[_red_flag("gi_bleed", "a"), _red_flag("sepsis", "b")])
    )["messages"][0]["content"]

    assert reply_a == reply_b


# --- states urgency clearly, not "see a doctor eventually" ----------------------


def test_emergency_node_message_directs_to_the_er():
    reply = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))["messages"][0][
        "content"
    ]
    assert "طوارئ" in reply


def test_emergency_node_message_offers_calling_emergency_services():
    reply = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))["messages"][0][
        "content"
    ]
    assert "الإسعاف" in reply


def test_emergency_node_message_says_immediately_not_eventually():
    reply = emergency_node(_state(red_flags=[_red_flag("acs_chest_pain")]))["messages"][0][
        "content"
    ]
    assert "حالًا" in reply or "فورًا" in reply
    assert "لبكرا" in reply  # explicitly: don't wait until tomorrow


# --- real end-to-end: extract_symptoms -> check_red_flags -> emergency_node -----


def _extraction_response(symptoms) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _red_flag_response(has_red_flag: bool) -> _ProviderResponse:
    return _ProviderResponse(
        text=json.dumps({"has_red_flag": has_red_flag, "reasoning": None}, ensure_ascii=False)
    )


class _FakeProvider:
    name = "fake"

    def __init__(self, *, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def test_real_end_to_end_chest_pain_and_shortness_of_breath_reaches_emergency_node():
    # The exact combination already verified (test_nodes_check_red_flags.py)
    # to fire two real rules — acs_chest_pain and pulmonary_embolism —
    # through the real rule engine. schemas.symptoms.SymptomExtraction's
    # name enum holds the NORMALIZED form (CLAUDE.md > Symptom vocabulary
    # > Compare against the normalized form...), so the fake extraction
    # response must too.
    set_provider(
        _FakeProvider(
            responses=[
                _extraction_response(
                    [normalize("ألم في الصدر"), normalize("ضيق تنفس")]
                ),
                _red_flag_response(False),
            ]
        )
    )

    state = {
        "thread_id": "t1",
        "messages": [{"role": "user", "content": "عندي ألم في الصدر وضيق تنفس من نص ساعة"}],
        "medical_record_summary": "",
        "unmatched_mentions": [],
    }
    state.update(extract_symptoms(state))
    state.update(check_red_flags(state))

    ids = {entry["id"] for entry in state["red_flags"]}
    assert ids == {"acs_chest_pain", "pulmonary_embolism"}
    assert all(entry["reason"] for entry in state["red_flags"])

    result = emergency_node(state)

    assert result["stage"] == "emergency"
    assert result["thread_outcome"] == "emergency"
    assert result["severity"] == "emergency"
    reply = result["messages"][0]["content"]
    assert "طوارئ" in reply
    # The doctor-facing reasons must not have leaked into this message.
    for entry in state["red_flags"]:
        assert entry["reason"] not in reply
