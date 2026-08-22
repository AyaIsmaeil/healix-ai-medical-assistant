"""Real, unmocked checkpointer persistence — no FakeCheckpointer here.

This is the mechanism the entire conversation-memory design rests on
(CLAUDE.md > Architecture: "this service owns... conversation state"), so
it is tested against a real SqliteSaver writing to a real file on disk via
graph.build_checkpointer()/build_graph(), the exact functions production
code calls. Only the LLM calls inside crisis_check/extract_symptoms/
check_red_flags/assess_sufficiency are faked (as in every other test in
this project) — the graph wiring and the checkpointer are completely
real. Every turn below takes the non-crisis, non-red-flag path (plain
"عندي صداع", nothing that fires a real rule), so it runs all four of
those nodes before ending — one turn is four scripted responses, not
three. assess_sufficiency is scripted sufficient=True, so the turn
continues rag_retrieve -> diagnose -> route_specialty -> generate_reports;
the single generic symptom never clears rag_retrieve.MIN_MATCHED_SYMPTOMS,
so diagnose short-circuits to insufficient_information with no LLM call
(same reasoning as tests/unit/test_graph.py's
test_invoking_the_graph_with_sufficient_information_reaches_diagnose),
meaning the script above still only needs four responses per turn.
generate_reports itself makes no LLM call either, but — unlike before it
was wired into graph.py — it DOES append one assistant message and set
stage="diagnosis" (ask_followup's own distinct behavior is covered
separately, in tests/integration/test_graph_followup_loop.py). These
tests assert message COUNT and that the user-authored entries are
preserved verbatim, not the generated report's exact wording — that
content isn't this file's concern, only checkpointer mechanics are.
"""

from __future__ import annotations

import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider

from graph import build_checkpointer, build_graph
from rules.crisis import normalize


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def _no_crisis_response() -> _ProviderResponse:
    payload = {"is_crisis": False, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _empty_extraction_response() -> _ProviderResponse:
    payload = {"symptoms": [], "negated_symptoms": [], "unmatched_mentions": []}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _no_red_flag_response() -> _ProviderResponse:
    payload = {"potential_red_flag": False, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _sufficient_response() -> _ProviderResponse:
    payload = {"is_sufficient": True, "next_question": None, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _one_turn_of_responses() -> list[_ProviderResponse]:
    """crisis_check, extract_symptoms, check_red_flags, assess_sufficiency
    — the four calls one non-crisis, non-red-flag turn makes before
    ending. Scripted sufficient=True so the turn ends at the placeholder
    END, not ask_followup — these tests assert exact message lists, and
    an extra assistant message would break that unrelated to what's
    actually under test here.
    """
    return [
        _no_crisis_response(),
        _empty_extraction_response(),
        _no_red_flag_response(),
        _sufficient_response(),
    ]


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _open_graph(sqlite_path, monkeypatch):
    """Build a fresh checkpointer + graph pointed at the same file.

    Called more than once against the same path within a test to stand in
    for a new process/connection picking the thread back up later — not
    just the same in-memory objects still holding the state.
    """
    monkeypatch.setenv("HEALIX_SQLITE_PATH", str(sqlite_path))
    monkeypatch.delenv("HEALIX_POSTGRES_DSN", raising=False)
    checkpointer = build_checkpointer()
    return checkpointer, build_graph(checkpointer)


def _extraction_response(symptoms=(), *, duration=None) -> _ProviderResponse:
    entries = []
    for name in symptoms:
        entry = {"name": name}
        if duration is not None:
            entry["duration"] = duration
        entries.append(entry)
    payload = {
        "symptoms": entries,
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _ordinary_turn_responses(*, symptoms=(), duration=None) -> list[_ProviderResponse]:
    return [
        _no_crisis_response(),
        _extraction_response(symptoms, duration=duration),
        _no_red_flag_response(),
        _sufficient_response(),
    ]


def _emergency_turn_responses() -> list[_ProviderResponse]:
    # Real deterministic rule match (chest pain + shortness of breath ->
    # acs_chest_pain + pulmonary_embolism) — since nodes/check_red_flags.py's
    # candidate/confirmed rewrite, the LLM's potential_red_flag screen
    # alone can no longer independently route to emergency_node (see that
    # module's docstring). Existing emergency path, used here only to
    # prove thread_outcome does not leak to a different conversation.
    return [
        _no_crisis_response(),
        _extraction_response([normalize("ألم في الصدر"), normalize("ضيق تنفس")]),
        _no_red_flag_response(),
    ]


def _symptom_names(state) -> list[str]:
    return [symptom.get("name") for symptom in state.get("symptoms") or []]


def _state_blob(state) -> str:
    return json.dumps(state, ensure_ascii=False, default=str)


def test_state_persists_across_invocations_on_the_same_thread_id(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "thread-persist"}}
    set_provider(FakeProvider(responses=[*_one_turn_of_responses(), *_one_turn_of_responses()]))

    checkpointer_1, graph_1 = _open_graph(sqlite_path, monkeypatch)
    graph_1.invoke(
        {"thread_id": "thread-persist", "messages": [{"role": "user", "content": "عندي صداع"}]},
        config=config,
    )
    checkpointer_1.conn.close()

    # A second, independent checkpointer/graph pair against the same file —
    # nothing Python-level is shared with the first invocation above.
    checkpointer_2, graph_2 = _open_graph(sqlite_path, monkeypatch)
    result = graph_2.invoke(
        {"messages": [{"role": "user", "content": "من امبارح كمان"}]}, config=config
    )
    checkpointer_2.conn.close()

    # Each turn now ends at generate_reports, which appends one assistant
    # message (see module docstring) — four entries total, not two.
    assert len(result["messages"]) == 4
    assert result["messages"][0] == {"role": "user", "content": "عندي صداع"}
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][2] == {"role": "user", "content": "من امبارح كمان"}
    assert result["messages"][3]["role"] == "assistant"


def test_different_thread_ids_do_not_see_each_others_state(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "checkpoints.sqlite"
    set_provider(FakeProvider(responses=[*_one_turn_of_responses(), *_one_turn_of_responses()]))

    checkpointer, graph = _open_graph(sqlite_path, monkeypatch)
    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    graph.invoke(
        {"thread_id": "thread-a", "messages": [{"role": "user", "content": "أعراض المريض أ"}]},
        config=config_a,
    )
    graph.invoke(
        {"thread_id": "thread-b", "messages": [{"role": "user", "content": "أعراض المريض ب"}]},
        config=config_b,
    )

    state_a = graph.get_state(config_a).values
    state_b = graph.get_state(config_b).values
    checkpointer.conn.close()

    # One turn each -> [user, assistant] now that generate_reports appends
    # a reply (see module docstring), not just the bare user message.
    assert state_a["messages"][0] == {"role": "user", "content": "أعراض المريض أ"}
    assert state_a["messages"][1]["role"] == "assistant"
    assert len(state_a["messages"]) == 2
    assert state_b["messages"][0] == {"role": "user", "content": "أعراض المريض ب"}
    assert state_b["messages"][1]["role"] == "assistant"
    assert len(state_b["messages"]) == 2
    # Not just "different" — neither thread's history contains a trace of
    # the other's content. A patient-privacy property, not just a
    # correctness one (CLAUDE.md > Non-negotiable safety rules; the two
    # threads could belong to two different patients).
    assert "أعراض المريض ب" not in json.dumps(state_a, ensure_ascii=False)
    assert "أعراض المريض أ" not in json.dumps(state_b, ensure_ascii=False)


HEADACHE = normalize("صداع")
# NOT "ألم بطن" (abdominal pain) — that term is RED_FLAG_RULES's
# ectopic_pregnancy rule's own all_of, so using it here would make these
# conversation-isolation tests accidentally exercise
# nodes/verify_red_flag.py's clarification path (a real, correct
# consequence of that rule's design, just not what this file is testing).
# "سعال" (cough), like HEADACHE above, is referenced by no rule at all —
# confirmed directly against RED_FLAG_RULES, same discipline
# tests/unit/test_nodes_check_red_flags.py's own ORDINARY_SYMPTOM uses.
ABDOMINAL_PAIN = normalize("سعال")


def test_a_new_conversation_does_not_inherit_prior_conversation_clinical_state(
    tmp_path, monkeypatch
):
    """Test A: conversation B must not contain A's headache, messages,
    candidates, reports, or thread outcome.
    """
    sqlite_path = tmp_path / "checkpoints.sqlite"
    set_provider(
        FakeProvider(
            responses=[
                *_ordinary_turn_responses(symptoms=[HEADACHE]),
                *_ordinary_turn_responses(symptoms=[ABDOMINAL_PAIN]),
            ]
        )
    )
    checkpointer, graph = _open_graph(sqlite_path, monkeypatch)
    config_a = {"configurable": {"thread_id": "conversation-a"}}
    config_b = {"configurable": {"thread_id": "conversation-b"}}

    graph.invoke(
        {
            "thread_id": "conversation-a",
            "messages": [{"role": "user", "content": "عندي صداع"}],
        },
        config=config_a,
    )
    graph.invoke(
        {
            "thread_id": "conversation-b",
            "messages": [{"role": "user", "content": "عندي ألم بالبطن"}],
        },
        config=config_b,
    )

    snapshot_a = graph.get_state(config_a)
    snapshot_b = graph.get_state(config_b)
    state_a = snapshot_a.values
    state_b = snapshot_b.values
    checkpointer.conn.close()

    assert snapshot_a.config["configurable"]["thread_id"] == "conversation-a"
    assert snapshot_b.config["configurable"]["thread_id"] == "conversation-b"

    assert HEADACHE in _symptom_names(state_a)
    assert ABDOMINAL_PAIN in _symptom_names(state_b)
    assert HEADACHE not in _symptom_names(state_b)
    assert ABDOMINAL_PAIN not in _symptom_names(state_a)

    blob_b = _state_blob(state_b)
    assert "عندي صداع" not in blob_b
    assert HEADACHE not in blob_b
    assert state_b.get("thread_outcome") in (None, )
    assert state_a.get("thread_outcome") in (None, )

    a_reports = state_a.get("reports") or {}
    b_reports = state_b.get("reports") or {}
    if a_reports:
        assert a_reports != b_reports or "صداع" not in json.dumps(b_reports, ensure_ascii=False)


def test_b_emergency_thread_outcome_does_not_leak_into_a_new_conversation(
    tmp_path, monkeypatch
):
    """Test B: conversation A reaches emergency; B starts clean."""
    sqlite_path = tmp_path / "checkpoints.sqlite"
    set_provider(
        FakeProvider(
            responses=[
                *_emergency_turn_responses(),
                *_ordinary_turn_responses(symptoms=[HEADACHE]),
            ]
        )
    )
    checkpointer, graph = _open_graph(sqlite_path, monkeypatch)
    config_a = {"configurable": {"thread_id": "conversation-emergency"}}
    config_b = {"configurable": {"thread_id": "conversation-new"}}

    turn_a = graph.invoke(
        {
            "thread_id": "conversation-emergency",
            "messages": [{"role": "user", "content": "عندي شي مو طبيعي وخايف"}],
        },
        config=config_a,
    )
    graph.invoke(
        {
            "thread_id": "conversation-new",
            "messages": [{"role": "user", "content": "عندي صداع"}],
        },
        config=config_b,
    )

    state_a = graph.get_state(config_a).values
    state_b = graph.get_state(config_b).values
    checkpointer.conn.close()

    assert turn_a["thread_outcome"] == "emergency"
    assert state_a["thread_outcome"] == "emergency"
    assert state_a["stage"] == "emergency"
    assert state_b.get("thread_outcome") is None
    assert state_b["stage"] != "emergency"
    assert "emergency" not in json.dumps(state_b.get("reports") or {}, ensure_ascii=False)


def test_c_same_conversation_preserves_state_across_turns(tmp_path, monkeypatch):
    """Test C: turn 2 on the same conversation keeps headache and adds duration."""
    sqlite_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "conversation-same"}}
    set_provider(
        FakeProvider(
            responses=[
                *_ordinary_turn_responses(symptoms=[HEADACHE]),
                *_ordinary_turn_responses(symptoms=[HEADACHE], duration="من مبارح"),
            ]
        )
    )
    checkpointer, graph = _open_graph(sqlite_path, monkeypatch)
    graph.invoke(
        {
            "thread_id": "conversation-same",
            "messages": [{"role": "user", "content": "عندي صداع"}],
        },
        config=config,
    )
    graph.invoke(
        {
            "thread_id": "conversation-same",
            "messages": [{"role": "user", "content": "من مبارح"}],
        },
        config=config,
    )

    state = graph.get_state(config).values
    checkpointer.conn.close()

    assert state["messages"][0] == {"role": "user", "content": "عندي صداع"}
    assert state["messages"][2] == {"role": "user", "content": "من مبارح"}
    names = _symptom_names(state)
    assert names == [HEADACHE]
    headache = state["symptoms"][0]
    assert headache["duration"] == "من مبارح"


def test_d_two_simultaneous_conversations_for_the_same_patient_stay_isolated(
    tmp_path, monkeypatch
):
    """Test D: same patient_sex and record summary, two conversation ids."""
    sqlite_path = tmp_path / "checkpoints.sqlite"
    patient = {
        "patient_sex": "female",
        "medical_record_summary": "Diabetes mellitus type 2",
    }
    set_provider(
        FakeProvider(
            responses=[
                *_ordinary_turn_responses(symptoms=[HEADACHE]),
                *_ordinary_turn_responses(symptoms=[ABDOMINAL_PAIN]),
            ]
        )
    )
    checkpointer, graph = _open_graph(sqlite_path, monkeypatch)
    config_a = {"configurable": {"thread_id": "patient-1-conv-a"}}
    config_b = {"configurable": {"thread_id": "patient-1-conv-b"}}

    graph.invoke(
        {
            "thread_id": "patient-1-conv-a",
            **patient,
            "messages": [{"role": "user", "content": "عندي صداع"}],
        },
        config=config_a,
    )
    graph.invoke(
        {
            "thread_id": "patient-1-conv-b",
            **patient,
            "messages": [{"role": "user", "content": "عندي ألم بالبطن"}],
        },
        config=config_b,
    )

    state_a = graph.get_state(config_a).values
    state_b = graph.get_state(config_b).values
    checkpointer.conn.close()

    assert state_a.get("patient_sex") == "female"
    assert state_b.get("patient_sex") == "female"
    assert HEADACHE in _symptom_names(state_a)
    assert HEADACHE not in _symptom_names(state_b)
    assert ABDOMINAL_PAIN in _symptom_names(state_b)
    assert "عندي صداع" not in _state_blob(state_b)
    assert "عندي ألم بالبطن" not in _state_blob(state_a)
    assert (state_a.get("turn_count") or 0) == (state_b.get("turn_count") or 0)


def test_e_langgraph_checkpoints_are_keyed_separately_per_thread_id(
    tmp_path, monkeypatch
):
    """Test E: graph.get_state() on each thread_id returns that thread only."""
    sqlite_path = tmp_path / "checkpoints.sqlite"
    set_provider(
        FakeProvider(
            responses=[
                *_ordinary_turn_responses(symptoms=[HEADACHE]),
                *_ordinary_turn_responses(symptoms=[ABDOMINAL_PAIN]),
            ]
        )
    )
    checkpointer, graph = _open_graph(sqlite_path, monkeypatch)
    config_a = {"configurable": {"thread_id": "ckpt-a"}}
    config_b = {"configurable": {"thread_id": "ckpt-b"}}

    graph.invoke(
        {"thread_id": "ckpt-a", "messages": [{"role": "user", "content": "عندي صداع"}]},
        config=config_a,
    )
    graph.invoke(
        {
            "thread_id": "ckpt-b",
            "messages": [{"role": "user", "content": "عندي ألم بالبطن"}],
        },
        config=config_b,
    )

    snapshot_a = graph.get_state(config_a)
    snapshot_b = graph.get_state(config_b)
    missing = graph.get_state({"configurable": {"thread_id": "ckpt-never-used"}})
    checkpointer.conn.close()

    assert snapshot_a.config["configurable"]["thread_id"] == "ckpt-a"
    assert snapshot_b.config["configurable"]["thread_id"] == "ckpt-b"
    assert snapshot_a.values["thread_id"] == "ckpt-a"
    assert snapshot_b.values["thread_id"] == "ckpt-b"
    assert HEADACHE in _symptom_names(snapshot_a.values)
    assert ABDOMINAL_PAIN in _symptom_names(snapshot_b.values)
    assert not missing.values.get("messages")
    assert not missing.values.get("symptoms")
