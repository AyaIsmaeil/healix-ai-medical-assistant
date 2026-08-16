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
    payload = {"has_red_flag": False, "reasoning": None}
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
