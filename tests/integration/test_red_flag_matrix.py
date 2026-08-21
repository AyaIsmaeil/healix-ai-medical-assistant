"""Red-flag regression matrix — the 10 categories from this phase's audit
(Phase 0/1: candidate/confirmed red-flag redesign, nodes/check_red_flags.py
+ nodes/verify_red_flag.py).

Real, unmocked graph + real, unmocked deterministic rule engine
(rules/red_flags.py) throughout — only the LLM boundary is faked, same
discipline as every other graph-level test in this project
(tests/unit/test_graph.py, tests/integration/test_graph_persistence.py).

--- Scope note on categories C-G (negation, family attribution,
hypothetical, past/resolved, anxiety without a current symptom) ---

These five categories are, structurally, extract_symptoms's own
responsibility: prompts/templates/extract_symptoms.txt and
prompts/templates/check_red_flags.txt (see the latter's explicit
exclusion list: "منفيًا صراحة / منسوبًا لشخص آخر / افتراضيًا أو خوفًا من
حدوثه مستقبلًا / عرضًا قديمًا زال / قلقًا دون ذكر عرض حالي") are what
decide whether the LLM extraction layer puts a given phrase into
state["symptoms"], state["negated_symptoms"], or neither. This file does
NOT re-verify that prompt-engineering judgment (that would require a real
LLM call, which this project's test suite deliberately never does — every
other graph-level test here fakes that boundary the same way). What it
verifies instead is the guarantee this PHASE actually controls: GIVEN a
correctly-extracted state (the symptom correctly absent from symptoms, or
correctly present in negated_symptoms), rules/red_flags.py and
nodes/check_red_flags.py do the safe thing, with no separate special-case
branch that could reintroduce the excluded symptom. That is what a
regression in THIS phase's code could actually break.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider

from graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from rules.crisis import normalize


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


def _crisis(is_crisis=False):
    return _ProviderResponse(
        text=json.dumps({"is_crisis": is_crisis, "reasoning": None}, ensure_ascii=False)
    )


def _extraction(symptoms=(), negated=(), unmatched=()):
    payload = {
        "symptoms": [{"name": n} for n in symptoms],
        "negated_symptoms": [{"name": n} for n in negated],
        "unmatched_mentions": list(unmatched),
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _red_flag(potential=False):
    return _ProviderResponse(
        text=json.dumps({"potential_red_flag": potential, "reasoning": None}, ensure_ascii=False)
    )


def _verification_question(question="بتحس بضيق تنفس أو تعرق غزير هلق؟"):
    return _ProviderResponse(text=json.dumps({"question": question}, ensure_ascii=False))


def _sufficient(is_sufficient=True, next_question=None):
    return _ProviderResponse(
        text=json.dumps(
            {"is_sufficient": is_sufficient, "next_question": next_question, "reasoning": None},
            ensure_ascii=False,
        )
    )


def _invoke(provider, *, thread_id, message, config=None):
    set_provider(provider)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        cfg = config or {"configurable": {"thread_id": thread_id}}
        return compiled.invoke(
            {"thread_id": thread_id, "messages": [{"role": "user", "content": message}]},
            config=cfg,
        )
    finally:
        conn.close()


CHEST_PAIN = normalize("ألم في الصدر")
DYSPNEA = normalize("ضيق تنفس")
SWEATING = normalize("تعرق غزير")
NAUSEA = normalize("غثيان")
VOMITING = normalize("تقيؤ")
ARM_JAW_PAIN = normalize("ألم منتشر للذراع أو الفك")

_ACS_ANY_OF = [DYSPNEA, SWEATING, ARM_JAW_PAIN, NAUSEA, VOMITING]


# --- A. Hard emergency (already-combined symptoms) ---------------------------


def test_a_chest_pain_plus_dyspnea_is_an_immediate_hard_emergency():
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([CHEST_PAIN, DYSPNEA]),
            _red_flag(False),
        ]
    )
    result = _invoke(provider, thread_id="matrix-a", message="عندي ألم في الصدر وضيق تنفس")

    assert result["stage"] == "emergency"
    assert result["safety_decision"] == "HARD_EMERGENCY"
    assert {f["id"] for f in result["red_flags"]} >= {"acs_chest_pain", "pulmonary_embolism"}


# --- B. Isolated chest pain -> candidate, not emergency -----------------------


def test_b_isolated_chest_pain_asks_for_clarification_not_emergency():
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([CHEST_PAIN]),
            _red_flag(False),
            _verification_question(),
        ]
    )
    result = _invoke(provider, thread_id="matrix-b", message="عندي ألم بالصدر")

    assert result["stage"] == "followup"
    assert result["safety_decision"] == "NEEDS_CLARIFICATION"
    assert result["red_flags"] == []
    assert result["messages"][-1]["role"] == "assistant"


# --- C. Explicit negation -----------------------------------------------------


def test_c_explicitly_negated_symptom_never_reaches_a_red_flag():
    # extract_symptoms correctly places the denial in negated_symptoms,
    # never symptoms — "ما عندي ضيق نفس" (see module docstring's scope note).
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([], negated=[DYSPNEA]),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    result = _invoke(provider, thread_id="matrix-c", message="ما عندي ضيق نفس")

    assert result["safety_decision"] == "NO_RED_FLAG"
    assert result["red_flags"] == []
    assert result["red_flag_candidates"] == []


# --- D. Family attribution ----------------------------------------------------


def test_d_a_family_members_symptom_is_not_attributed_to_the_patient():
    # "أمي عندها ضيق نفس" — extract_symptoms correctly returns nothing at
    # all (not the patient's own symptom, not a denial either).
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([]),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    result = _invoke(provider, thread_id="matrix-d", message="أمي عندها ضيق نفس")

    assert result["safety_decision"] == "NO_RED_FLAG"
    assert result["red_flags"] == []


# --- E. Hypothetical / feared future symptom ----------------------------------


def test_e_a_feared_future_symptom_is_not_a_current_one():
    # "خايف يصير معي ضيق نفس" — fear of a FUTURE symptom, not a report of
    # one happening now.
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([]),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    result = _invoke(provider, thread_id="matrix-e", message="خايف يصير معي ضيق تنفس")

    assert result["safety_decision"] == "NO_RED_FLAG"
    assert result["red_flags"] == []


# --- F. Past / resolved symptom -----------------------------------------------


def test_f_a_resolved_past_symptom_does_not_trigger_a_current_red_flag():
    # "كان عندي ألم صدر الأسبوع الماضي بس راح خلص" — extract_symptoms
    # correctly excludes a symptom the patient describes as already over.
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([]),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    result = _invoke(
        provider, thread_id="matrix-f", message="كان عندي ألم صدر الأسبوع الماضي بس راح خلص"
    )

    assert result["safety_decision"] == "NO_RED_FLAG"
    assert result["red_flags"] == []


# --- G. Anxiety without an actual reported symptom ----------------------------


def test_g_anxiety_about_a_symptom_with_no_symptom_actually_reported():
    # "خايف يكون في شي غلط بقلبي" — worry/anxiety, no concrete symptom.
    # unmatched_mentions may still carry the phrase for assess_sufficiency's
    # own follow-up judgment (module docstring) — that is a normal-flow
    # concern, not a red-flag one.
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([], unmatched=["قلق من مشكلة بالقلب"]),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    result = _invoke(provider, thread_id="matrix-g", message="خايف يكون في شي غلط بقلبي")

    assert result["safety_decision"] == "NO_RED_FLAG"
    assert result["red_flags"] == []


# --- H. Reassuring context (candidate explicitly rejected in the same turn) --


def test_h_chest_pain_with_every_discriminator_explicitly_denied_is_rejected():
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([CHEST_PAIN], negated=_ACS_ANY_OF),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    result = _invoke(
        provider,
        thread_id="matrix-h",
        message="عندي ألم بالصدر بس ما عندي ضيق تنفس ولا تعرق ولا غثيان ولا تقيؤ",
    )

    assert result["safety_decision"] == "NO_RED_FLAG"
    assert result["red_flags"] == []
    assert result["red_flag_candidates"] == []


# --- I. Dangerous information appearing on a LATER turn ----------------------


def test_i_a_discriminator_confirmed_on_a_later_turn_escalates_to_emergency():
    # Turn 1: isolated chest pain -> candidate, asks for clarification.
    # Turn 2: patient confirms shortness of breath -> the SAME cited rule
    # now fully matches through the ordinary deterministic engine (no
    # special-case code in verify_red_flag needed for this — see that
    # module's own docstring) -> HARD_EMERGENCY.
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([CHEST_PAIN]),
            _red_flag(False),
            _verification_question(),
            _crisis(False),
            _extraction([DYSPNEA]),
            _red_flag(False),
        ]
    )
    set_provider(provider)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        config = {"configurable": {"thread_id": "matrix-i"}}

        turn_1 = compiled.invoke(
            {"thread_id": "matrix-i", "messages": [{"role": "user", "content": "عندي ألم بالصدر"}]},
            config=config,
        )
        assert turn_1["safety_decision"] == "NEEDS_CLARIFICATION"
        assert turn_1["stage"] == "followup"

        turn_2 = compiled.invoke(
            {"messages": [{"role": "user", "content": "إي في ضيق تنفس كمان"}]},
            config=config,
        )
        assert turn_2["safety_decision"] == "HARD_EMERGENCY"
        assert turn_2["stage"] == "emergency"
        assert "acs_chest_pain" in {f["id"] for f in turn_2["red_flags"]}
    finally:
        conn.close()


# --- J. Reassuring information appearing on a LATER turn ---------------------


def test_j_a_discriminator_denied_on_a_later_turn_resolves_the_candidate():
    # Turn 1: isolated chest pain -> candidate, asks for clarification.
    # Turn 2: patient denies every remaining discriminator -> the same
    # candidate resolves to rejected -> normal flow continues, no
    # escalation.
    provider = FakeProvider(
        responses=[
            _crisis(False),
            _extraction([CHEST_PAIN]),
            _red_flag(False),
            _verification_question(),
            _crisis(False),
            _extraction([], negated=_ACS_ANY_OF),
            _red_flag(False),
            _sufficient(True),
        ]
    )
    set_provider(provider)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        config = {"configurable": {"thread_id": "matrix-j"}}

        turn_1 = compiled.invoke(
            {"thread_id": "matrix-j", "messages": [{"role": "user", "content": "عندي ألم بالصدر"}]},
            config=config,
        )
        assert turn_1["safety_decision"] == "NEEDS_CLARIFICATION"

        turn_2 = compiled.invoke(
            {
                "messages": [
                    {"role": "user", "content": "لأ، ما في ضيق تنفس ولا تعرق ولا غثيان ولا تقيؤ"}
                ]
            },
            config=config,
        )
        assert turn_2["safety_decision"] == "NO_RED_FLAG"
        assert turn_2["red_flags"] == []
        assert turn_2["red_flag_candidates"] == []
        assert turn_2["stage"] != "emergency"
    finally:
        conn.close()
