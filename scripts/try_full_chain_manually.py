"""MANUAL TEST SCRIPT — invokes the REAL COMPILED GRAPH
(graph.build_graph() + graph.build_checkpointer()), not a hand-chained
sequence of node function calls the way every other scripts/try_*_manually.py
script does. This is the only manual script that exercises graph.py's own
wiring/routing (crisis_check -> ... -> rag_retrieve -> diagnose) end to
end, against whatever LLM provider/model is currently configured for the
"quality" tier in .env.

*** This is NOT part of the pytest suite and must never be added to it. ***

Run it by hand:

    python scripts/try_full_chain_manually.py "عندي صداع نابض من جهة وحدة، وغثيان، وحساسية من الضوء والصوت"
    python scripts/try_full_chain_manually.py "..." --yes           # skip the confirmation
    python scripts/try_full_chain_manually.py "..." --thread-id t1  # resume a specific thread
    python scripts/try_full_chain_manually.py                       # prompts for a message

A real thread_id (random per run by default, CLAUDE.md > State: identity
that persists conversation state across invoke() calls via the real
checkpointer graph.build_checkpointer() selects — SqliteSaver locally
unless HEALIX_POSTGRES_DSN is set) — not a throwaway value, since the
whole point of this script over the node-level ones is exercising that
persistence, not just one node's logic in isolation. --thread-id lets a
later run resume an earlier one on purpose, e.g. to pick up a follow-up
loop across two separate invocations of this script rather than in one
sitting.

If the first turn's assess_sufficiency judges the picture insufficient,
this script keeps invoking the graph on the SAME thread_id, printing the
follow-up question and prompting for the patient's next message, until a
turn reaches a real terminal outcome (crisis, emergency, or diagnose) —
or the patient/tester types nothing, which exits early. This is what
"interactive" means here: a real multi-turn conversation loop against
the real checkpointer, not a single fire-and-forget call.

One turn can make up to five real LLM calls (crisis_check,
extract_symptoms, check_red_flags, assess_sufficiency, diagnose — the
first four always tier="quality"; diagnose too, when it runs at all).
The confirmation prompt below says so up front. The loop is naturally
bounded by nodes.assess_sufficiency.MAX_FOLLOW_UP_QUESTIONS (a forced
"proceed anyway" after that many follow-ups), so it cannot run forever,
but it can still be several turns' worth of real calls — this is by far
the most expensive script in this family, not "one or two calls" like
the single- or double-node ones.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from graph import build_checkpointer, build_graph  # noqa: E402
from llm_client import LLMError  # noqa: E402
from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS  # noqa: E402
from scripts._try_manually_common import (  # noqa: E402
    confirm_billable_call,
    quality_tier_config,
    reconfigure_utf8_stdio,
    resolve_message,
)

load_dotenv()
reconfigure_utf8_stdio()


def _print_candidate_diseases(candidates: list[dict]) -> None:
    print("  candidate_diseases:")
    if not candidates:
        print("    (none)")
        return
    for candidate in candidates:
        print(f"    - {candidate['name']}  (match_score={candidate['match_score']})")
        matched = "، ".join(candidate["matched_symptoms"]) or "(none)"
        missing = "، ".join(candidate["missing_symptoms"]) or "(none)"
        print(f"        matched symptoms: {matched}")
        print(f"        missing symptoms: {missing}")
        if candidate["negated_symptoms"]:
            print(f"        negated symptoms: {'، '.join(candidate['negated_symptoms'])}")


def _print_diagnosis(diagnosis: dict | None) -> None:
    print("  diagnosis:")
    if diagnosis is None:
        print("    (not reached this turn)")
        return
    print(f"    status: {diagnosis['status']}")
    if diagnosis.get("reasoning"):
        print(f"    reasoning: {diagnosis['reasoning']}")
    differential = diagnosis.get("differential") or []
    if not differential:
        print("    differential: (none)")
        return
    print("    differential:")
    for entry in differential:
        print(
            f"      - {entry['name']}  "
            f"(match_score={entry['match_score']}, certainty={entry['certainty']})"
        )


def _latest_assistant_reply(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def _print_turn_summary(turn_number: int, result: dict) -> None:
    print("=" * 72)
    print(f"TURN {turn_number} RESULT — stage={result.get('stage')!r}")
    print("=" * 72)
    reply = _latest_assistant_reply(result.get("messages", []))
    if reply:
        print(f"  assistant reply: {reply}")
    if result.get("red_flags"):
        print("  red_flags:")
        for entry in result["red_flags"]:
            print(f"    - {entry['id']}")
    print()


def _print_final_state(result: dict) -> None:
    print("=" * 72)
    print("FINAL STATE")
    print("=" * 72)
    print(f"  stage: {result.get('stage')!r}")
    print(f"  turn_count: {result.get('turn_count')}")
    print(f"  is_sufficient: {result.get('is_sufficient')}")
    print(f"  information_limited: {result.get('information_limited')}")
    print()
    _print_candidate_diseases(result.get("candidate_diseases") or [])
    print()
    _print_diagnosis(result.get("diagnosis"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message", nargs="?", default=None, help="Patient's first message (prompted for if omitted)."
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        metavar="ID",
        help="Reuse a specific thread_id (e.g. to resume a follow-up loop from an earlier run). Random per run by default.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip the billable-provider confirmation prompt."
    )
    args = parser.parse_args()

    provider, model = quality_tier_config()
    thread_id = args.thread_id or f"manual-full-chain-{uuid.uuid4().hex[:8]}"

    print("=" * 72)
    print("Manual FULL GRAPH test (graph.build_graph(), not hand-chained nodes)")
    print("=" * 72)
    print(f"  quality-tier provider : {provider or '(not set)'}")
    print(f"  quality-tier model    : {model or '(not set)'}")
    print(f"  thread_id             : {thread_id}")
    print(
        "  (one turn makes up to FIVE real LLM calls: crisis_check, "
        "extract_symptoms, check_red_flags, assess_sufficiency, diagnose — "
        f"and this may loop for several turns, bounded by "
        f"MAX_FOLLOW_UP_QUESTIONS={MAX_FOLLOW_UP_QUESTIONS})"
    )
    print()

    if not confirm_billable_call(provider, skip=args.yes):
        return 0

    message = resolve_message(args.message)
    if not message:
        print("No message given. Aborting.")
        return 1

    checkpointer = build_checkpointer()
    try:
        compiled = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        turn_number = 1
        invoke_input: dict = {
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": message}],
        }

        while True:
            print(f"  calling the graph (turn {turn_number})...")
            print()
            try:
                result = compiled.invoke(invoke_input, config=config)
            except LLMError as exc:
                print(f"  RESULT: the LLM call failed on turn {turn_number}.")
                print(f"    {type(exc).__name__}: {exc}")
                print()
                print("  Check HEALIX_LLM_PROVIDER_QUALITY / HEALIX_MODEL_QUALITY and the")
                print("  matching API key in .env.")
                return 1

            _print_turn_summary(turn_number, result)

            if result.get("stage") != "followup":
                _print_final_state(result)
                return 0

            next_message = input("  Your reply (empty to stop): ").strip()
            if not next_message:
                print("  Stopping — no reply given. Conversation left at 'followup'.")
                _print_final_state(result)
                return 0

            invoke_input = {"messages": [{"role": "user", "content": next_message}]}
            turn_number += 1
    finally:
        checkpointer.conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
