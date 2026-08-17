"""Unit tests for graph.py's own logic: checkpointer selection and
assembly shape.

The checkpointer's actual persistence/isolation behavior is a real,
unmocked integration test (tests/integration/test_graph.py) — this file
covers only the decision of which backend graph.py picks, and that
build_graph() wires the nodes it claims to.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.rag_retrieve import _SEX_CLARIFICATION_QUESTION
from rules.crisis import normalize
from schemas.crisis import CrisisResponse

import graph
from graph import (
    _DEFAULT_SQLITE_PATH,
    _route_after_assess_sufficiency,
    _route_after_check_red_flags,
    _route_after_crisis_check,
    _route_after_rag_retrieve,
    build_checkpointer,
    build_graph,
)
from state import HealixState


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def _crisis_check_response(is_crisis: bool) -> _ProviderResponse:
    payload = {"is_crisis": is_crisis, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _crisis_node_response(message: str) -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"message": message}, ensure_ascii=False))


def _extract_symptoms_response(symptoms=(), negated=(), unmatched=()) -> _ProviderResponse:
    payload = {
        "symptoms": [dict(s) for s in symptoms],
        "negated_symptoms": [dict(n) for n in negated],
        "unmatched_mentions": list(unmatched),
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _check_red_flags_response(has_red_flag: bool) -> _ProviderResponse:
    payload = {"has_red_flag": has_red_flag, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _sufficiency_response(is_sufficient: bool, next_question: str | None = None) -> _ProviderResponse:
    payload = {"is_sufficient": is_sufficient, "next_question": next_question, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _diagnose_response(status: str, differential=(), reasoning: str | None = None) -> _ProviderResponse:
    payload = {"status": status, "differential": list(differential), "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_checkpointer_env(monkeypatch):
    monkeypatch.delenv("HEALIX_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("HEALIX_SQLITE_PATH", raising=False)
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


# --- build_checkpointer: sqlite (default / local-dev) branch -----------------


def test_uses_sqlite_when_no_postgres_dsn_is_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALIX_SQLITE_PATH", str(tmp_path / "checkpoints.sqlite"))

    checkpointer = build_checkpointer()
    try:
        assert isinstance(checkpointer, SqliteSaver)
    finally:
        checkpointer.conn.close()


def test_sqlite_path_is_configurable(tmp_path, monkeypatch):
    target = tmp_path / "custom_name.sqlite"
    monkeypatch.setenv("HEALIX_SQLITE_PATH", str(target))

    checkpointer = build_checkpointer()
    try:
        db_file = checkpointer.conn.execute("PRAGMA database_list").fetchone()[2]
        assert db_file == str(target)
    finally:
        checkpointer.conn.close()


def test_sqlite_falls_back_to_the_documented_default_path_when_unset(tmp_path, monkeypatch):
    # HEALIX_SQLITE_PATH is deleted by the autouse fixture; chdir into a
    # scratch directory so the default (a relative path) can't write into
    # the real project root as a side effect of running this test.
    monkeypatch.chdir(tmp_path)

    checkpointer = build_checkpointer()
    try:
        db_file = checkpointer.conn.execute("PRAGMA database_list").fetchone()[2]
        assert db_file == str((tmp_path / _DEFAULT_SQLITE_PATH).resolve())
    finally:
        checkpointer.conn.close()


# --- build_checkpointer: postgres branch, connection mocked -------------------
#
# No real Postgres server is available to (or should be assumed by) this
# suite. What's under test here is graph.py's own selection logic — that
# HEALIX_POSTGRES_DSN being set routes to PostgresSaver and calls the
# setup() its docstring requires — not psycopg's or PostgresSaver's actual
# database behavior. Mocking the connection boundary is the same pattern
# llm_client's tests use for provider SDKs.


class _FakeConnection:
    """Stands in for whatever psycopg.Connection.connect would return."""


def test_uses_postgres_when_a_dsn_is_configured(monkeypatch):
    monkeypatch.setenv("HEALIX_POSTGRES_DSN", "postgresql://user:pass@host/db")

    connect_calls = []

    def fake_connect(dsn, **kwargs):
        connect_calls.append((dsn, kwargs))
        return _FakeConnection()

    setup_calls = []
    monkeypatch.setattr(graph.Connection, "connect", staticmethod(fake_connect))
    monkeypatch.setattr(graph.PostgresSaver, "setup", lambda self: setup_calls.append(self))

    checkpointer = build_checkpointer()

    assert isinstance(checkpointer, graph.PostgresSaver)
    assert connect_calls[0][0] == "postgresql://user:pass@host/db"
    assert len(setup_calls) == 1


def test_postgres_setup_is_called_so_tables_exist_on_first_use(monkeypatch):
    monkeypatch.setenv("HEALIX_POSTGRES_DSN", "postgresql://user:pass@host/db")
    monkeypatch.setattr(graph.Connection, "connect", staticmethod(lambda dsn, **kw: _FakeConnection()))

    setup_calls = []
    monkeypatch.setattr(graph.PostgresSaver, "setup", lambda self: setup_calls.append(self))

    build_checkpointer()

    assert len(setup_calls) == 1


# --- _route_after_crisis_check: pure routing decision -------------------------


def test_routes_to_crisis_node_when_is_crisis_is_true():
    assert _route_after_crisis_check({"is_crisis": True}) == "crisis_node"


def test_routes_to_extract_symptoms_when_is_crisis_is_false():
    assert _route_after_crisis_check({"is_crisis": False}) == "extract_symptoms"


def test_routes_to_extract_symptoms_when_is_crisis_is_absent():
    # Not expected in practice (crisis_check always sets it), but routing
    # must not crash on a state that hasn't been through crisis_check yet.
    assert _route_after_crisis_check({}) == "extract_symptoms"


# --- _route_after_check_red_flags: pure routing decision -----------------------


def test_routes_to_emergency_node_when_red_flags_is_non_empty():
    state = {"red_flags": [{"id": "acs_chest_pain", "reason": "سبب"}]}
    assert _route_after_check_red_flags(state) == "emergency_node"


def test_routes_to_assess_sufficiency_when_red_flags_is_empty():
    assert _route_after_check_red_flags({"red_flags": []}) == "assess_sufficiency"


def test_routes_to_assess_sufficiency_when_red_flags_is_absent():
    # Not expected in practice (check_red_flags always sets it), but
    # routing must not crash on a state that hasn't reached it yet.
    assert _route_after_check_red_flags({}) == "assess_sufficiency"


def test_routes_to_assess_sufficiency_when_thread_outcome_is_none():
    state = {"red_flags": [], "thread_outcome": None}
    assert _route_after_check_red_flags(state) == "assess_sufficiency"


# --- _route_after_check_red_flags: thread_outcome (CLAUDE.md > Non-negotiable
# safety rule 13) -----------------------------------------------------------------


def test_routes_to_reiterate_terminal_outcome_when_thread_already_reached_emergency():
    state = {"red_flags": [], "thread_outcome": "emergency"}
    assert _route_after_check_red_flags(state) == "reiterate_terminal_outcome"


def test_routes_to_reiterate_terminal_outcome_when_thread_already_reached_crisis():
    state = {"red_flags": [], "thread_outcome": "crisis"}
    assert _route_after_check_red_flags(state) == "reiterate_terminal_outcome"


def test_a_genuine_new_red_flag_still_reaches_emergency_node_even_if_thread_outcome_already_set():
    # Priority ordering: red_flags is checked BEFORE thread_outcome, so a
    # real new escalation is never swallowed by the "already terminal"
    # branch.
    state = {
        "red_flags": [{"id": "acs_chest_pain", "reason": "سبب"}],
        "thread_outcome": "crisis",
    }
    assert _route_after_check_red_flags(state) == "emergency_node"


# --- _route_after_assess_sufficiency: pure routing decision --------------------


def test_routes_to_ask_followup_when_is_sufficient_is_false():
    assert _route_after_assess_sufficiency({"is_sufficient": False}) == "ask_followup"


def test_routes_to_rag_retrieve_when_is_sufficient_is_true():
    assert _route_after_assess_sufficiency({"is_sufficient": True}) == "rag_retrieve"


def test_routes_to_end_when_is_sufficient_is_absent():
    # Not expected in practice (assess_sufficiency always sets it), but
    # routing must not crash on a state that hasn't reached it yet — in
    # particular must NOT default to "ask_followup" (reads
    # state["next_question"] with no fallback and would raise) nor to
    # "rag_retrieve" (would run rag_retrieve/diagnose, including a real
    # LLM call, against a state that was never actually judged
    # sufficient).
    assert _route_after_assess_sufficiency({}) == END


# --- _route_after_rag_retrieve: pure routing decision ---------------------------


def test_routes_to_ask_followup_when_rag_retrieve_set_a_next_question():
    # rag_retrieve only ever sets this itself for a sex-clarification need
    # (nodes/rag_retrieve.py) — assess_sufficiency already reset it to
    # None this same turn when it decided is_sufficient=True.
    state = {"next_question": "قبل ما نكمل، ممكن تحكيلي إذا كنت رجل أو امرأة؟"}
    assert _route_after_rag_retrieve(state) == "ask_followup"


def test_routes_to_ml_corroborate_when_rag_retrieve_set_no_next_question():
    # ml_corroborate, not diagnose directly — see CLAUDE.md's XGBoost
    # corroboration-signal section: it always runs immediately before
    # diagnose on this branch, never on the ask_followup branch above.
    assert _route_after_rag_retrieve({"next_question": None}) == "ml_corroborate"


def test_routes_to_ml_corroborate_when_next_question_is_absent():
    # Not expected in practice (assess_sufficiency always sets the key to
    # something, even if None), but routing must not crash on a state
    # that hasn't been through either node yet.
    assert _route_after_rag_retrieve({}) == "ml_corroborate"


# --- build_graph: assembly shape ----------------------------------------------


def test_build_graph_wires_the_full_currently_implemented_topology():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        representation = compiled.get_graph()

        assert set(representation.nodes) == {
            "__start__",
            "reset_stage",
            "crisis_check",
            "crisis_node",
            "extract_symptoms",
            "check_red_flags",
            "emergency_node",
            "assess_sufficiency",
            "ask_followup",
            "rag_retrieve",
            "ml_corroborate",
            "diagnose",
            "route_specialty",
            "generate_reports",
            "reiterate_terminal_outcome",
            "__end__",
        }
        edges = {(edge.source, edge.target) for edge in representation.edges}
        assert edges == {
            ("__start__", "reset_stage"),
            ("reset_stage", "crisis_check"),
            ("crisis_check", "crisis_node"),
            ("crisis_check", "extract_symptoms"),
            ("crisis_node", "__end__"),
            ("extract_symptoms", "check_red_flags"),
            ("check_red_flags", "emergency_node"),
            ("check_red_flags", "reiterate_terminal_outcome"),
            ("check_red_flags", "assess_sufficiency"),
            ("emergency_node", "__end__"),
            ("reiterate_terminal_outcome", "__end__"),
            ("assess_sufficiency", "ask_followup"),
            ("assess_sufficiency", "rag_retrieve"),
            ("assess_sufficiency", "__end__"),
            ("ask_followup", "__end__"),
            ("rag_retrieve", "ask_followup"),
            ("rag_retrieve", "ml_corroborate"),
            ("ml_corroborate", "diagnose"),
            ("diagnose", "route_specialty"),
            ("route_specialty", "generate_reports"),
            ("generate_reports", "__end__"),
        }
    finally:
        conn.close()


# --- build_graph: real end-to-end routing, LLM boundary faked ------------------
#
# The two tests above prove the edges exist and that the routing function
# picks correctly in isolation; this proves the two actually agree at
# runtime — e.g. a path_map key that doesn't match what
# _route_after_crisis_check returns would pass both of those and still
# blow up here.


def test_invoking_the_graph_with_a_crisis_reaches_crisis_node():
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(True),
                _crisis_node_response("فهمتك، وهاد الشي خارج قدرتي."),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {"thread_id": "t1", "messages": [{"role": "user", "content": "بدي موت"}]},
            config={"configurable": {"thread_id": "t1"}},
        )
        assert result["stage"] == "crisis"
        assert result["messages"][-1]["role"] == "assistant"
    finally:
        conn.close()


def test_invoking_the_graph_with_sufficient_information_reaches_diagnose():
    # "صداع" alone (bare headache, no detail) legitimately produces zero
    # rag_retrieve candidates against the real 37-disease knowledge base
    # (verified directly: nothing clears MIN_MATCHED_SYMPTOMS on a single
    # generic symptom) — so diagnose short-circuits to
    # insufficient_information without a 5th scripted LLM response.
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(symptoms=[{"name": "صداع"}]),
                _check_red_flags_response(False),
                _sufficiency_response(True),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {"thread_id": "t1", "messages": [{"role": "user", "content": "عندي صداع"}]},
            config={"configurable": {"thread_id": "t1"}},
        )
        # generate_reports is now the real terminal node on this branch —
        # it sets stage and appends this turn's reply, neither diagnose
        # nor route_specialty do (CLAUDE.md > State).
        assert result["stage"] == "diagnosis"
        assert len(result["messages"]) == 2
        assert result["messages"][-1]["role"] == "assistant"
        assert result["red_flags"] == []
        assert result["is_sufficient"] is True
        assert result["candidate_diseases"] == []
        assert result["diagnosis"] == {
            "status": "insufficient_information",
            "differential": [],
            "reasoning": None,
        }
        # route_specialty also ran (unconditional after diagnose, no LLM
        # call) and correctly fell back to general practice given
        # insufficient_information.
        assert result["specialty"] == "طب عام"
        # generate_reports made no LLM call either — both registers are
        # honest "couldn't determine" text, not fabricated content.
        assert result["reports"]["patient"]
        assert result["reports"]["doctor"]
        assert "insufficient_information" in result["reports"]["doctor"]
    finally:
        conn.close()


def test_invoking_the_graph_with_insufficient_information_reaches_ask_followup():
    question = "من متى بلش الصداع، وهل هو من جهة وحدة؟"
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(symptoms=[{"name": "صداع"}]),
                _check_red_flags_response(False),
                _sufficiency_response(False, next_question=question),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {"thread_id": "t1", "messages": [{"role": "user", "content": "عندي صداع"}]},
            config={"configurable": {"thread_id": "t1"}},
        )
        assert result["stage"] == "followup"
        assert result["messages"][-1] == {"role": "assistant", "content": question}
        assert result["turn_count"] == 1
    finally:
        conn.close()


# --- turn_count: single-increment-per-turn guarantee across its two writers ----
#
# nodes.assess_sufficiency and nodes.rag_retrieve both write turn_count
# (nodes/rag_retrieve.py's module docstring: a deliberate, coordinated
# exception to the original single-writer rule). This proves the two
# increment branches cannot both fire in the same turn — not by
# convention, but because graph.py's routing makes it structurally
# unreachable: rag_retrieve only ever runs when assess_sufficiency
# already returned is_sufficient=True THIS turn, and assess_sufficiency's
# own increment happens ONLY on its is_sufficient=False branch, which
# routes straight to ask_followup and ends the turn there — rag_retrieve
# is never even invoked. The symptom picture below (Dysmenorrhea's full,
# real symptom set, no patient_sex given) is deliberately chosen so it
# WOULD clear rag_retrieve's sex-clarification gate if rag_retrieve ever
# got a chance to run this turn — assess_sufficiency's own verdict is
# forced to "insufficient" regardless (a scripted response, not a
# naturally-occurring judgment) specifically to test the structural
# guarantee, not just the common case.


def test_a_turn_where_assess_sufficiency_is_insufficient_never_lets_rag_retrieve_increment_turn_count():
    question = "من متى بلش هالوجع بالضبط؟"
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(
                    # Dysmenorrhea's real, full symptom set (normalized) —
                    # would clear rag_retrieve's MIN_MATCHED_SYMPTOMS floor
                    # and its sex-clarification gate (no patient_sex given)
                    # if rag_retrieve ever ran this turn.
                    symptoms=[
                        {"name": "الم بطن"},
                        {"name": "الم اسفل الظهر"},
                        {"name": "غثيان"},
                    ]
                ),
                _check_red_flags_response(False),
                _sufficiency_response(False, next_question=question),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [{"role": "user", "content": "عندي ألم بطن وألم أسفل الظهر وغثيان"}],
            },
            config={"configurable": {"thread_id": "t1"}},
        )

        assert result["stage"] == "followup"
        assert result["turn_count"] == 1  # not 2 — only assess_sufficiency's own increment
        # The final next_question is assess_sufficiency's own text, not
        # rag_retrieve's sex-clarification constant — direct proof
        # rag_retrieve never ran and never overwrote it.
        assert result["messages"][-1] == {"role": "assistant", "content": question}
        assert question != _SEX_CLARIFICATION_QUESTION
        # candidate_diseases is a key ONLY rag_retrieve ever sets — its
        # absence is direct proof rag_retrieve was never invoked this turn.
        assert "candidate_diseases" not in result
    finally:
        conn.close()


def test_invoking_the_graph_with_a_red_flag_reaches_emergency_node():
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(
                    # schemas.symptoms.SymptomExtraction's name enum holds the
                    # NORMALIZED form (CLAUDE.md > Symptom vocabulary >
                    # Compare against the normalized form...) — a FakeProvider
                    # response still has to satisfy that same schema, so the
                    # authored (hamza) spelling would fail validation here.
                    symptoms=[
                        {"name": normalize("ألم في الصدر")},
                        {"name": normalize("ضيق تنفس")},
                    ]
                ),
                _check_red_flags_response(False),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [
                    {"role": "user", "content": "عندي ألم في الصدر وضيق تنفس"}
                ],
            },
            config={"configurable": {"thread_id": "t1"}},
        )
        assert result["stage"] == "emergency"
        assert result["messages"][-1]["role"] == "assistant"
        # Real rule engine, real combination — fires through
        # rules/red_flags.py, not just the (also-mocked-False) LLM layer.
        ids = {entry["id"] for entry in result["red_flags"]}
        assert ids >= {"acs_chest_pain", "pulmonary_embolism"}
    finally:
        conn.close()


# --- thread_outcome: sticky terminal safety outcome (CLAUDE.md > Non-negotiable
# safety rule 13) -------------------------------------------------------------------


def test_thread_outcome_sticky_repeat_after_emergency_reproduces_the_manual_scenario():
    # Reproduces the real manual-testing scenario that motivated safety
    # rule 13: emergency_node fires once, then a later message that
    # escalates nothing new ("ما بدي اتصل بالإسعاف" — refusing the ER,
    # not itself a crisis or red-flag signal) must get the fixed
    # reiterate_terminal_outcome reminder, not a fresh run through
    # assess_sufficiency/rag_retrieve/diagnose.
    #
    # Turn 1's red flag comes from check_red_flags's LLM layer alone
    # (has_red_flag=True with EMPTY extracted symptoms, so the
    # deterministic rule engine has nothing to match against — see
    # rules/red_flags.py: an empty confirmed-symptom set satisfies no
    # rule's requirement). This isolates the mechanism under test: a
    # RULE-matched red flag would keep re-matching every later turn on
    # its own (the causal symptoms never leave the accumulated
    # state["symptoms"]), which is a different, also-safe code path, not
    # what's being proven here.
    provider = FakeProvider(
        responses=[
            # Turn 1: emergency, via the LLM red-flag layer only.
            _crisis_check_response(False),
            _extract_symptoms_response(symptoms=[]),
            _check_red_flags_response(True),
            # Turn 2: nothing new escalates.
            _crisis_check_response(False),
            _extract_symptoms_response(symptoms=[]),
            _check_red_flags_response(False),
        ]
    )
    set_provider(provider)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        config = {"configurable": {"thread_id": "t1"}}

        turn_1 = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [{"role": "user", "content": "عندي شي مو طبيعي وخايف"}],
            },
            config=config,
        )
        assert turn_1["stage"] == "emergency"
        assert turn_1["thread_outcome"] == "emergency"
        assert turn_1["severity"] == "emergency"

        turn_2 = compiled.invoke(
            {"messages": [{"role": "user", "content": "ما بدي اتصل بالإسعاف"}]},
            config=config,
        )

        # Same standing outcome, reiterated — not a fresh pipeline run.
        assert turn_2["stage"] == "emergency"
        assert turn_2["thread_outcome"] == "emergency"
        assert turn_2["messages"][-1]["role"] == "assistant"
        # Never reached rag_retrieve/diagnose this turn.
        assert turn_2.get("diagnosis") is None
        assert not turn_2.get("candidate_diseases")

        # Exactly 6 LLM calls total (3 per turn: crisis_check,
        # extract_symptoms, check_red_flags) — proves turn 2 never
        # reached assess_sufficiency (a 7th call) or beyond.
        assert len(provider.calls) == 6
    finally:
        conn.close()


def test_thread_outcome_does_not_block_a_genuine_new_crisis_escalation():
    # A thread that already reached "emergency" must still let a real NEW
    # crisis signal reach the actual crisis_node (with its own LLM call),
    # not the fixed reiterate_terminal_outcome reminder — crisis_check
    # runs every turn regardless of thread_outcome, and crisis always
    # takes priority (safety rule 4). The reverse direction (a genuine
    # new red flag still reaching emergency_node on an already-"crisis"
    # thread) is covered at the pure routing-function level above
    # (test_a_genuine_new_red_flag_still_reaches_emergency_node_even_if_thread_outcome_already_set).
    provider = FakeProvider(
        responses=[
            # Turn 1: emergency, via the LLM red-flag layer only (same
            # isolation reasoning as the test above).
            _crisis_check_response(False),
            _extract_symptoms_response(symptoms=[]),
            _check_red_flags_response(True),
            # Turn 2: a genuine new crisis signal.
            _crisis_check_response(True),
            _crisis_node_response("فهمتك، خلينا نحكي عن هلق."),
        ]
    )
    set_provider(provider)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        config = {"configurable": {"thread_id": "t1"}}

        turn_1 = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [{"role": "user", "content": "عندي شي مو طبيعي وخايف"}],
            },
            config=config,
        )
        assert turn_1["thread_outcome"] == "emergency"

        turn_2 = compiled.invoke(
            {"messages": [{"role": "user", "content": "بدي موت، ما في فايدة"}]},
            config=config,
        )

        # The REAL crisis_node ran (its own LLM call, schema=CrisisResponse
        # among provider.calls) — not the fixed reminder — and correctly
        # overwrote thread_outcome from "emergency" to "crisis".
        assert turn_2["stage"] == "crisis"
        assert turn_2["thread_outcome"] == "crisis"
        schemas_called = [call["schema"] for call in provider.calls]
        assert CrisisResponse in schemas_called
    finally:
        conn.close()


def test_invoking_the_graph_full_chain_reaches_a_migraine_diagnosis():
    # The exact migraine-pattern message used elsewhere this session
    # (unilateral throbbing headache + nausea + photophobia +
    # phonophobia), now run through the FULL wired chain: extract_symptoms
    # -> check_red_flags (no red flag) -> assess_sufficiency (sufficient)
    # -> rag_retrieve -> diagnose -> route_specialty -> generate_reports.
    # rag_retrieve is real, unmocked, against the full 37-disease
    # knowledge base — verified directly beforehand that Migraine is
    # still the sole candidate at 1.0 match_score at this larger, denser
    # KB size (no other entry's symptom list overlaps this specific
    # compound-symptom combination), and that its real KB entry's
    # specialty is ["عصبية"] and name_ar is "الشقيقة". Neither
    # route_specialty nor generate_reports makes an LLM call of its own.
    # Only the LLM boundary is faked (crisis_check, extract_symptoms,
    # check_red_flags, assess_sufficiency, diagnose) — this is what makes
    # it a real end-to-end integration test at the GRAPH level, not just
    # a node-level check: every assertion below (specialty, reports,
    # stage) reads the final graph state compiled.invoke() actually
    # returns, not any node's return value called directly.
    migraine_symptoms = [
        normalize("صداع نابض من جهة واحدة"),
        normalize("غثيان"),
        normalize("حساسية للضوء"),
        normalize("حساسية للصوت"),
    ]
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(symptoms=[{"name": n} for n in migraine_symptoms]),
                _check_red_flags_response(False),
                _sufficiency_response(True),
                _diagnose_response("differential", differential=["Migraine"], reasoning="تطابق تام"),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [
                    {
                        "role": "user",
                        "content": "عندي صداع نابض من جهة وحدة، وغثيان، وحساسية من الضوء والصوت",
                    }
                ],
            },
            config={"configurable": {"thread_id": "t1"}},
        )

        assert result["candidate_diseases"][0]["name"] == "Migraine"
        assert result["candidate_diseases"][0]["name_ar"] == "الشقيقة"
        assert result["candidate_diseases"][0]["match_score"] == 1.0
        assert result["diagnosis"]["status"] == "differential"
        assert result["diagnosis"]["differential"][0]["name"] == "Migraine"
        assert result["diagnosis"]["differential"][0]["name_ar"] == "الشقيقة"
        assert result["diagnosis"]["differential"][0]["certainty"] == "high"
        assert result["specialty"] == "عصبية"

        # generate_reports is the real terminal node now — stage,
        # messages, and both reports are all set on this final state.
        assert result["stage"] == "diagnosis"
        assert result["messages"][-1]["role"] == "assistant"

        patient_report = result["reports"]["patient"]
        doctor_report = result["reports"]["doctor"]
        assert patient_report
        assert doctor_report
        assert "الشقيقة" in patient_report
        assert "Migraine" not in patient_report
        assert "Migraine" in doctor_report
        assert "الشقيقة" in doctor_report
    finally:
        conn.close()


def test_invoking_the_graph_full_chain_produces_an_ml_corroboration_signal_for_hypertension():
    # Real, fully unmocked ml_corroborate (real vendored bundle, real
    # feature_mapper, real density_floor) — only the LLM boundary is
    # faked, same discipline as the migraine test above. Chosen
    # deliberately: rag/knowledge_base/hypertension.json's own symptom
    # list is exactly {"صداع", "دوخة"} (CLAUDE.md > State's own
    # documented example of a short KB entry), which is also exactly
    # enough to clear ml.density_floor's MIN_REQUIRED_MATCHED=2 for
    # Hypertension (chest_pain/dizziness/headache/loss_of_balance) via
    # headache+dizziness alone — verified directly against the real
    # model during this feature's audit that this specific pair's
    # predict_proba() argmax really is Hypertension itself (CLAUDE.md's
    # XGBoost corroboration-signal section, the sparse-input finding).
    hypertension_symptoms = [normalize("صداع"), normalize("دوخة")]
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(symptoms=[{"name": n} for n in hypertension_symptoms]),
                _check_red_flags_response(False),
                _sufficiency_response(True),
                _diagnose_response("differential", differential=["Hypertension"], reasoning="تطابق"),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [{"role": "user", "content": "عندي صداع ودوخة"}],
            },
            config={"configurable": {"thread_id": "t1"}},
        )

        assert result["candidate_diseases"][0]["name"] == "Hypertension"
        assert result["candidate_diseases"][0]["ml_corroboration"] == "model_signal_present"
        # Carried through diagnose unchanged.
        assert result["diagnosis"]["differential"][0]["ml_corroboration"] == "model_signal_present"

        # Doctor-only, never patient-facing, never a raw number.
        assert "XGBoost" in result["reports"]["doctor"]
        assert "XGBoost" not in result["reports"]["patient"]
        assert "model_signal_present" not in result["reports"]["patient"]
    finally:
        conn.close()


def test_invoking_the_graph_produces_no_ml_corroboration_for_a_disease_outside_the_crosswalk():
    # Dysmenorrhea has no ml.disease_crosswalk entry at all — proves the
    # common case (no crosswalk match) completes normally with no signal
    # and no error, real ml_corroborate included in the path.
    dysmenorrhea_symptoms = [
        normalize("الم بطن"),
        normalize("الم اسفل الظهر"),
        normalize("غثيان"),
    ]
    set_provider(
        FakeProvider(
            responses=[
                _crisis_check_response(False),
                _extract_symptoms_response(symptoms=[{"name": n} for n in dysmenorrhea_symptoms]),
                _check_red_flags_response(False),
                _sufficiency_response(True),
                _diagnose_response("differential", differential=["Dysmenorrhea"], reasoning="تطابق"),
            ]
        )
    )
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        result = compiled.invoke(
            {
                "thread_id": "t1",
                "messages": [{"role": "user", "content": "عندي ألم بطن وألم أسفل الظهر وغثيان"}],
                "patient_sex": "female",
            },
            config={"configurable": {"thread_id": "t1"}},
        )

        assert result["candidate_diseases"][0]["name"] == "Dysmenorrhea"
        assert "ml_corroboration" not in result["candidate_diseases"][0]
        assert "ml_corroboration" not in result["diagnosis"]["differential"][0]
        assert "XGBoost" not in result["reports"]["doctor"]
    finally:
        conn.close()


def test_build_graph_uses_the_given_checkpointer_instance():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        compiled = build_graph(checkpointer)
        assert compiled.checkpointer is checkpointer
    finally:
        conn.close()


def test_build_graph_state_schema_is_healix_state():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        assert compiled.builder.state_schema is HealixState
    finally:
        conn.close()
