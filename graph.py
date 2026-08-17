"""graph.py — graph assembly only. No business logic.

Every node's own logic lives in nodes/; every prompt in
prompts/templates/. This file only wires nodes together and selects a
checkpointer (CLAUDE.md > Working style: build one node at a time — the
graph below is deliberately just the first slice of CLAUDE.md > Graph
flow, extended node by node as each one is built and tested).
"""

from __future__ import annotations

import logging
import os
import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg import Connection
from psycopg.rows import dict_row

from nodes.ask_followup import ask_followup
from nodes.assess_sufficiency import assess_sufficiency
from nodes.check_red_flags import check_red_flags
from nodes.crisis_check import crisis_check
from nodes.crisis_node import crisis_node
from nodes.diagnose import diagnose
from nodes.emergency_node import emergency_node
from nodes.extract_symptoms import extract_symptoms
from nodes.generate_reports import generate_reports
from nodes.ml_corroborate import ml_corroborate
from nodes.rag_retrieve import rag_retrieve
from nodes.reiterate_terminal_outcome import reiterate_terminal_outcome
from nodes.reset_stage import reset_stage
from nodes.route_specialty import route_specialty
from state import HealixState

_logger = logging.getLogger("healix.graph")

# Only used when HEALIX_POSTGRES_DSN is unset — see build_checkpointer().
_DEFAULT_SQLITE_PATH = "healix_checkpoints.sqlite"


def build_checkpointer() -> BaseCheckpointSaver:
    """Select and construct the checkpointer this service's conversation
    memory rests on: PostgresSaver if HEALIX_POSTGRES_DSN is set,
    otherwise SqliteSaver for local development.

    This is a deliberate either/or on one explicitly named setting, not a
    try-Postgres-then-fall-back: if HEALIX_POSTGRES_DSN is set and the
    connection fails, that failure propagates rather than silently
    degrading to a local SQLite file — the same "no silent fallback"
    reasoning llm_client.py applies to provider selection (CLAUDE.md >
    LLM tiers). An *unset* DSN is not a failure at all; it is local-dev
    mode, chosen on purpose, and logged as such so it is never mistaken
    for an accident.
    """
    dsn = os.getenv("HEALIX_POSTGRES_DSN", "").strip()
    if dsn:
        _logger.info("checkpointer=postgres (HEALIX_POSTGRES_DSN is set)")
        conn = Connection.connect(
            dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        checkpointer = PostgresSaver(conn)
        # Unlike SqliteSaver (which sets itself up lazily on first use and
        # documents that callers should NOT call setup() directly),
        # PostgresSaver's own docstring requires this to be called
        # explicitly on first use. Idempotent: tracks applied migrations
        # and no-ops once up to date, so calling it on every startup is
        # the correct, intended usage, not a repeated side effect.
        checkpointer.setup()
        return checkpointer

    sqlite_path = os.getenv("HEALIX_SQLITE_PATH", "").strip() or _DEFAULT_SQLITE_PATH
    _logger.info(
        "checkpointer=sqlite path=%s (HEALIX_POSTGRES_DSN not set — local development mode)",
        sqlite_path,
    )
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    return SqliteSaver(conn)


def _route_after_crisis_check(state: HealixState) -> str:
    """crisis_check -> crisis_node if it set is_crisis, else on to extract_symptoms.

    Safety rule 4 (CLAUDE.md): the crisis path bypasses everything else —
    there is no third option here. Runs identically regardless of
    state["thread_outcome"] (CLAUDE.md > Non-negotiable safety rule 13):
    a genuine new crisis signal must always be able to reach the real
    crisis_node, even on a thread that already reached "emergency" on an
    earlier turn — this function has no awareness of thread_outcome at
    all, on purpose, so there is nothing here that could suppress it.
    """
    return "crisis_node" if state.get("is_crisis") else "extract_symptoms"


def _route_after_check_red_flags(state: HealixState) -> str:
    """check_red_flags -> emergency_node if it found anything this turn,
    else -> reiterate_terminal_outcome if this thread already reached a
    terminal safety outcome on an earlier turn and nothing new escalated
    this turn, else -> assess_sufficiency.

    Safety rule 4 (CLAUDE.md): the red-flag path bypasses RAG and
    diagnosis entirely, straight to its terminal node — same reasoning as
    crisis_check's branch above. red_flags is checked BEFORE
    thread_outcome deliberately (CLAUDE.md > Non-negotiable safety rule
    13): a genuine NEW red flag must still reach emergency_node even on a
    thread that already reached "crisis" — state["thread_outcome"] (state.py)
    has no reducer, so whichever terminal node fires most recently is
    simply what gets recorded, and this ordering is what lets a later,
    real escalation actually reach it rather than being silently
    swallowed by the "already terminal" branch below.
    """
    if state.get("red_flags"):
        return "emergency_node"
    if state.get("thread_outcome") is not None:
        return "reiterate_terminal_outcome"
    return "assess_sufficiency"


def _route_after_assess_sufficiency(state: HealixState) -> str:
    """assess_sufficiency -> ask_followup if more information is needed,
    else on to rag_retrieve.

    Checks is_sufficient against True/False explicitly, not truthiness —
    unlike the two routing functions above, "field absent, default to the
    continue-forward branch" here would mean defaulting into rag_retrieve
    -> diagnose, which now does real work (a real LLM call in diagnose)
    for a state that never actually went through assess_sufficiency's own
    judgment, not just a harmless no-op the way the old END placeholder
    was. END is deliberately still the fallback for that unreached-state
    case — not expected in practice (assess_sufficiency always sets
    is_sufficient), but kept explicit rather than silently doing
    diagnostic work on an unjudged state.
    """
    if state.get("is_sufficient") is False:
        return "ask_followup"
    if state.get("is_sufficient") is True:
        return "rag_retrieve"
    return END


def _route_after_rag_retrieve(state: HealixState) -> str:
    """rag_retrieve -> ask_followup if it needs patient_sex confirmed
    before a sex-restricted candidate can be safely included, else on to
    ml_corroborate.

    rag_retrieve only ever sets state["next_question"] itself when it hit
    this ambiguity (nodes/rag_retrieve.py's module docstring, "Sex-specific
    gating") — assess_sufficiency already reset it to None this same turn
    when it decided is_sufficient=True (state.py's own field comment), so
    a truthy value here unambiguously means rag_retrieve just set one,
    not a stale value from earlier in the turn or an earlier turn.

    Routes to ml_corroborate, not straight to diagnose: a turn that's
    only asking a sex-clarification question has no finished candidate
    set yet, so there is nothing useful for the ML corroboration signal
    (CLAUDE.md's XGBoost corroboration-signal section) to annotate this
    turn — ml_corroborate always runs immediately before diagnose from
    here on, never on the ask_followup branch.
    """
    return "ask_followup" if state.get("next_question") else "ml_corroborate"


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Assemble the graph:

        START -> reset_stage -> crisis_check -> crisis_node -> END
                                             \\-> extract_symptoms -> check_red_flags -> emergency_node -> END
                                                                                      \\-> reiterate_terminal_outcome -> END
                                                                                      \\-> assess_sufficiency -> ask_followup -> END
                                                                                                              \\-> rag_retrieve -> ask_followup -> END
                                                                                                                                \\-> ml_corroborate -> diagnose -> route_specialty -> generate_reports -> END

    reset_stage runs first, unconditionally, on every turn — it clears
    state["stage"] before any node this turn could set one (CLAUDE.md >
    State: stage has no reducer, so without this a previous turn's value
    would otherwise leak into a turn that doesn't reach a stage-setting
    node of its own). It deliberately does NOT clear state["thread_outcome"]
    (CLAUDE.md > Non-negotiable safety rule 13) — see nodes/reset_stage.py's
    own docstring for why the two fields need opposite per-turn lifetimes.

    crisis_node and emergency_node both always end the turn (CLAUDE.md >
    Non-negotiable safety rule 4: both paths bypass RAG and diagnosis
    entirely, so there is nowhere else for either to route to) — and both
    also set state["thread_outcome"] to their own value, unconditionally,
    every time they fire (CLAUDE.md > Non-negotiable safety rule 13). A
    later turn on the same thread that escalates neither a fresh crisis
    signal nor a new red flag is routed by _route_after_check_red_flags
    to reiterate_terminal_outcome instead of assess_sufficiency — a fixed,
    non-LLM reminder of the standing directive, not a fresh run through
    normal symptom triage. ask_followup always ends the turn too —
    CLAUDE.md > Graph flow: it awaits the patient's next message, which
    arrives as a fresh invoke() on the same thread_id (CLAUDE.md > State:
    persistence is the checkpointer, keyed by thread_id).

    rag_retrieve -> ml_corroborate is the ONE conditional edge in this
    branch, added for sex-specific KB gating (nodes/rag_retrieve.py's
    module docstring, "Sex-specific gating"): when a sex-restricted
    candidate would otherwise qualify but state["patient_sex"] is
    unconfirmed, rag_retrieve sets state["next_question"] itself and
    _route_after_rag_retrieve sends the turn to the SAME ask_followup
    node the assess_sufficiency <-> ask_followup loop already uses, not a
    new one — reusing ask_followup's existing contract (send
    next_question, end the turn, await the next invoke() on this
    thread_id) rather than inventing a second follow-up mechanism.
    Otherwise unconditional from there: ml_corroborate -> diagnose ->
    route_specialty -> generate_reports. ml_corroborate (CLAUDE.md's
    XGBoost corroboration-signal section) only ever annotates entries
    already in state["candidate_diseases"] with an optional
    ml_corroboration field, or leaves the list completely untouched on
    any error (fail-open, non-critical path) — it never blocks or alters
    what reaches diagnose. diagnose already handles an empty
    state["candidate_diseases"] itself (short-circuits to
    insufficient_information, no LLM call — CLAUDE.md > Non-negotiable
    safety rule 6), route_specialty likewise already handles an
    insufficient_information diagnosis itself (falls back to
    GENERAL_PRACTICE, no LLM call), and generate_reports likewise
    produces an honest "couldn't determine" pair of reports for that same
    status rather than needing a routing decision here. generate_reports
    is this path's real terminal node, routed to END: it is the node that
    actually sets state["stage"] = "diagnosis" (CLAUDE.md > State — the
    field neither diagnose nor route_specialty touches) and appends this
    turn's assistant-facing reply to state["messages"], the same
    contract every other terminal node (crisis_node, emergency_node,
    ask_followup) already fulfills.

    Takes the checkpointer rather than constructing one, so callers (and
    tests) control its lifetime explicitly instead of this function
    reaching into the environment on their behalf.
    """
    builder = StateGraph(HealixState)
    builder.add_node("reset_stage", reset_stage)
    builder.add_node("crisis_check", crisis_check)
    builder.add_node("crisis_node", crisis_node)
    builder.add_node("extract_symptoms", extract_symptoms)
    builder.add_node("check_red_flags", check_red_flags)
    builder.add_node("emergency_node", emergency_node)
    builder.add_node("assess_sufficiency", assess_sufficiency)
    builder.add_node("ask_followup", ask_followup)
    builder.add_node("rag_retrieve", rag_retrieve)
    builder.add_node("ml_corroborate", ml_corroborate)
    builder.add_node("diagnose", diagnose)
    builder.add_node("route_specialty", route_specialty)
    builder.add_node("generate_reports", generate_reports)
    builder.add_node("reiterate_terminal_outcome", reiterate_terminal_outcome)

    builder.add_edge(START, "reset_stage")
    builder.add_edge("reset_stage", "crisis_check")
    builder.add_conditional_edges(
        "crisis_check",
        _route_after_crisis_check,
        {"crisis_node": "crisis_node", "extract_symptoms": "extract_symptoms"},
    )
    builder.add_edge("crisis_node", END)
    builder.add_edge("extract_symptoms", "check_red_flags")
    builder.add_conditional_edges(
        "check_red_flags",
        _route_after_check_red_flags,
        {
            "emergency_node": "emergency_node",
            "reiterate_terminal_outcome": "reiterate_terminal_outcome",
            "assess_sufficiency": "assess_sufficiency",
        },
    )
    builder.add_edge("emergency_node", END)
    builder.add_edge("reiterate_terminal_outcome", END)
    builder.add_conditional_edges(
        "assess_sufficiency",
        _route_after_assess_sufficiency,
        {"ask_followup": "ask_followup", "rag_retrieve": "rag_retrieve", END: END},
    )
    builder.add_edge("ask_followup", END)
    builder.add_conditional_edges(
        "rag_retrieve",
        _route_after_rag_retrieve,
        {"ask_followup": "ask_followup", "ml_corroborate": "ml_corroborate"},
    )
    builder.add_edge("ml_corroborate", "diagnose")
    builder.add_edge("diagnose", "route_specialty")
    builder.add_edge("route_specialty", "generate_reports")
    builder.add_edge("generate_reports", END)

    return builder.compile(checkpointer=checkpointer)
