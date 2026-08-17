"""Real, unmocked multi-turn loop: assess_sufficiency <-> ask_followup.

Same approach as tests/integration/test_graph_persistence.py — a real
SqliteSaver via graph.build_checkpointer()/build_graph(), only the LLM
calls faked. This file is about the specific conversational mechanic
CLAUDE.md > Graph flow describes for this loop: an insufficient verdict
ends the turn at ask_followup, the patient's next message arrives as a
fresh invoke() on the same thread_id, and what they say elaborating on an
already-named symptom must actually reach and merge into that symptom's
state — not get lost or treated as a brand-new complaint.

The location+quality elaboration scenario below ("من جهة وحدة، نابض"
answering a bare "صداع") is the exact one manually verified against a
real LLM earlier in this project's development, once with the raw_mention
formatting bug present (silently invisible to assess_sufficiency) and
once after the fix (visible, and correctly not re-asked for) — this test
pins that fix down permanently at the graph level, LLM calls faked so it
runs in the normal suite without cost or flakiness.

nodes.reset_stage's own "no stage-setting node reached this turn, so
don't silently inherit a previous turn's stage" property no longer has a
LIVE branch to demonstrate it against: now that route_specialty ->
generate_reports is wired in, every reachable path out of
assess_sufficiency sets its own stage (ask_followup -> "followup" or
generate_reports -> "diagnosis") — the only unstaged branch left in
graph.py's _route_after_assess_sufficiency is the bare `return END`
fallback for an is_sufficient-absent state, which is not reachable via
real assess_sufficiency output (CLAUDE.md > State: reset_stage's
clearing behavior is documented there as permanent, forward-looking
infrastructure regardless of whether every branch happens to be staged
today). reset_stage's own clearing logic stays covered at the unit level
(tests/unit/test_nodes_reset_stage.py); the graph-level test that used to
pin this down here (test_stage_does_not_carry_over_from_a_previous_turn)
was removed for exactly this reason rather than kept passing against a
premise that's no longer true.
"""

from __future__ import annotations

import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider

from graph import build_graph


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


def _extraction_response(symptoms: list[dict]) -> _ProviderResponse:
    payload = {
        "symptoms": symptoms,
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _no_red_flag_response() -> _ProviderResponse:
    payload = {"has_red_flag": False, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _sufficiency_response(is_sufficient: bool, next_question: str | None = None) -> _ProviderResponse:
    payload = {"is_sufficient": is_sufficient, "next_question": next_question, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _open_graph(sqlite_path, monkeypatch):
    monkeypatch.setenv("HEALIX_SQLITE_PATH", str(sqlite_path))
    monkeypatch.delenv("HEALIX_POSTGRES_DSN", raising=False)
    from graph import build_checkpointer

    checkpointer = build_checkpointer()
    return checkpointer, build_graph(checkpointer)


FOLLOW_UP_QUESTION = "وين مكان الصداع بالضبط، وهل هو نابض ولا ضاغط؟"


def test_full_multi_turn_followup_loop(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "thread-followup"}}

    set_provider(
        FakeProvider(
            responses=[
                # Turn 1: "عندي صداع" — bare, no distinguishing detail.
                _no_crisis_response(),
                _extraction_response([{"name": "صداع"}]),
                _no_red_flag_response(),
                _sufficiency_response(False, next_question=FOLLOW_UP_QUESTION),
                # Turn 2: "من جهة وحدة، نابض" — elaborates the SAME صداع
                # symptom via raw_mention, not a new symptom.
                _no_crisis_response(),
                _extraction_response(
                    [{"name": "صداع", "raw_mention": "من جهة وحدة، نابض"}]
                ),
                _no_red_flag_response(),
                _sufficiency_response(True),
            ]
        )
    )

    checkpointer, compiled = _open_graph(sqlite_path, monkeypatch)
    try:
        turn_1 = compiled.invoke(
            {
                "thread_id": "thread-followup",
                "messages": [{"role": "user", "content": "عندي صداع"}],
            },
            config=config,
        )

        # --- turn 1: insufficient, ends at ask_followup -----------------
        assert turn_1["stage"] == "followup"
        assert turn_1["messages"][-1] == {"role": "assistant", "content": FOLLOW_UP_QUESTION}
        assert turn_1["turn_count"] == 1
        # extract_symptoms returns ExtractedSymptom.model_dump(), which
        # includes every optional field explicitly as None, not just the
        # ones actually supplied — matching test_nodes_extract_symptoms.py's
        # own assertions for this shape. duration_days is a field added
        # alongside model_dump()'s own fields (nodes/extract_symptoms.py's
        # _with_duration_days) — None here since no duration was reported.
        assert turn_1["symptoms"] == [
            {
                "name": "صداع",
                "raw_mention": None,
                "duration": None,
                "severity": None,
                "onset": None,
                "duration_days": None,
            }
        ]

        turn_2 = compiled.invoke(
            {"messages": [{"role": "user", "content": "من جهة وحدة، نابض"}]},
            config=config,
        )

        # --- turn 2: the patient's answer reaches the SAME symptom ------
        # merge_symptoms (state.py) keys on name, filling gaps rather than
        # adding a second entry — this is what actually proves the
        # elaboration was recognized as more detail about "صداع", not a
        # brand-new, unrelated complaint.
        assert turn_2["symptoms"] == [
            {
                "name": "صداع",
                "raw_mention": "من جهة وحدة، نابض",
                "duration": None,
                "severity": None,
                "onset": None,
                "duration_days": None,
            }
        ]
        # turn_count is untouched by turn 2 (assess_sufficiency was
        # sufficient this time, so it does not increment further) — still
        # reflects exactly the one question actually asked.
        assert turn_2["turn_count"] == 1
        # Turn 2's single symptom ("صداع" alone) never clears
        # rag_retrieve.MIN_MATCHED_SYMPTOMS, so diagnose short-circuits to
        # insufficient_information (no LLM call) and route_specialty falls
        # back to general practice (also no LLM call) — but
        # generate_reports is still the real terminal node on this branch
        # now, and it DOES set stage, to "diagnosis", not None. This also
        # proves reset_stage ran correctly this turn: without it, a stale
        # "followup" from turn 1 could otherwise have leaked through
        # instead of turn 2 setting its own value (CLAUDE.md > State) —
        # see the module docstring above for why a dedicated "no
        # stage-setting node this turn" scenario no longer has a live
        # branch to test against, now that every reachable path sets one.
        assert turn_2["stage"] == "diagnosis"

        # --- the full conversation is exactly these four turns, in order.
        # generate_reports appends a 4th message (its own report text) —
        # not asserted verbatim here, that's generate_reports's own
        # test file's concern, not this multi-turn mechanic's.
        assert turn_2["messages"][:3] == [
            {"role": "user", "content": "عندي صداع"},
            {"role": "assistant", "content": FOLLOW_UP_QUESTION},
            {"role": "user", "content": "من جهة وحدة، نابض"},
        ]
        assert len(turn_2["messages"]) == 4
        assert turn_2["messages"][3]["role"] == "assistant"
    finally:
        checkpointer.conn.close()
