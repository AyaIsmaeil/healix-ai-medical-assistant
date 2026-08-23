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
from nodes.verify_red_flag import verify_red_flag
from state import HealixState

_logger = logging.getLogger("healix.graph")

# Only used when HEALIX_POSTGRES_DSN is unset — see build_checkpointer().
_DEFAULT_SQLITE_PATH = "healix_checkpoints.sqlite"



#
def build_checkpointer() -> BaseCheckpointSaver:
    """Select and construct the checkpointer this service's conversation
    """
    dsn = os.getenv("HEALIX_POSTGRES_DSN", "").strip()
    if dsn:
        _logger.info("checkpointer=postgres (HEALIX_POSTGRES_DSN is set)")
        conn = Connection.connect(
            dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        checkpointer = PostgresSaver(conn)
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
    """
    return "crisis_node" if state.get("is_crisis") else "extract_symptoms"


def _route_after_check_red_flags(state: HealixState) -> str:
    if state.get("red_flags"):
        return "emergency_node"
    if state.get("thread_outcome") is not None:
        return "reiterate_terminal_outcome"
    if state.get("red_flag_candidates"):
        return "verify_red_flag"
    return "assess_sufficiency"


def _route_after_verify_red_flag(state: HealixState) -> str:
    """verify_red_flag -> ask_followup if it set next_question, else on to assess_sufficiency.
    """
    return "ask_followup" if state.get("next_question") else "assess_sufficiency"


def _route_after_assess_sufficiency(state: HealixState) -> str:
    """assess_sufficiency -> ask_followup if it set next_question, else on to rag_retrieve.
    """
    if state.get("is_sufficient") is False:
        return "ask_followup"
    if state.get("is_sufficient") is True:
        return "rag_retrieve"
    return END


def _route_after_rag_retrieve(state: HealixState) -> str:
    """rag_retrieve -> ask_followup if it set next_question, else on to ml_corroborate.
    """
    return "ask_followup" if state.get("next_question") else "ml_corroborate"


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Assemble the graph of nodes and edges, then compile it into a `CompiledStateGraph` object.
    """
    builder = StateGraph(HealixState)
    builder.add_node("reset_stage", reset_stage)
    builder.add_node("crisis_check", crisis_check)
    builder.add_node("crisis_node", crisis_node)
    builder.add_node("extract_symptoms", extract_symptoms)
    builder.add_node("check_red_flags", check_red_flags)
    builder.add_node("emergency_node", emergency_node)
    builder.add_node("verify_red_flag", verify_red_flag)
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
            "verify_red_flag": "verify_red_flag",
            "assess_sufficiency": "assess_sufficiency",
        },
    )
    builder.add_edge("emergency_node", END)
    builder.add_edge("reiterate_terminal_outcome", END)
    builder.add_conditional_edges(
        "verify_red_flag",
        _route_after_verify_red_flag,
        {"ask_followup": "ask_followup", "assess_sufficiency": "assess_sufficiency"},
    )
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
