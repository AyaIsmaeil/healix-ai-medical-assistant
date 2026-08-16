"""MANUAL TEST SCRIPT — chains extract_symptoms -> assess_sufficiency for a
real end-to-end check, against whatever LLM provider/model is currently
configured for the "quality" tier in .env.

*** This is NOT part of the pytest suite and must never be added to it. ***
Neither node is wired into graph.py yet (CLAUDE.md > Working style: one
node at a time), so this calls both functions directly against a
hand-built state, feeding extract_symptoms's output straight into
assess_sufficiency's input — no checkpointer, no compiled graph. Run it
by hand:

    python scripts/try_sufficiency_manually.py "عندي صداع"
    python scripts/try_sufficiency_manually.py "عندي حرارة وسعال وضيق تنفس من يومين" --medical-record "مريض ربو"
    python scripts/try_sufficiency_manually.py "عندي صداع" --turn-count 6
    python scripts/try_sufficiency_manually.py "..." --yes   # skip the confirmation
    python scripts/try_sufficiency_manually.py               # prompts for a message

--turn-count seeds state["turn_count"] before the call, so the hard
ceiling (nodes.assess_sufficiency.MAX_FOLLOW_UP_QUESTIONS) can be
exercised without actually running six real turns first — set it to the
ceiling to watch assess_sufficiency force is_sufficient=True and
information_limited=True WITHOUT a second LLM call.

--medical-record exercises assess_sufficiency's medical_record_summary
input — otherwise dead in a script that only ever sends one message with
no record context.

Makes ONE OR TWO real LLM calls depending on --turn-count: extract_symptoms
always calls the LLM; assess_sufficiency only does if the ceiling hasn't
already been reached. The confirmation prompt below states which applies
before asking.

See scripts/try_red_flags_manually.py for the same chaining idea against
check_red_flags, and scripts/try_extract_manually.py /
scripts/try_crisis_manually.py for the other nodes in this family of
scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from llm_client import LLMError  # noqa: E402
from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS, assess_sufficiency  # noqa: E402
from nodes.extract_symptoms import extract_symptoms  # noqa: E402
from scripts._try_manually_common import (  # noqa: E402
    confirm_billable_call,
    print_string_list,
    print_symptom_list,
    quality_tier_config,
    reconfigure_utf8_stdio,
    resolve_message,
)

load_dotenv()
reconfigure_utf8_stdio()

TEST_THREAD_ID = "manual-sufficiency-test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message", nargs="?", default=None, help="Patient message to test (prompted for if omitted)."
    )
    parser.add_argument(
        "--medical-record",
        default="",
        metavar="SUMMARY",
        help="Optional medical record summary, fed to assess_sufficiency. Empty by default.",
    )
    parser.add_argument(
        "--turn-count",
        type=int,
        default=0,
        metavar="N",
        help=(
            f"Seed state['turn_count'] with this value before calling "
            f"assess_sufficiency, to exercise the hard ceiling "
            f"(MAX_FOLLOW_UP_QUESTIONS={MAX_FOLLOW_UP_QUESTIONS}) without "
            f"running that many real turns first. Default 0."
        ),
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip the billable-provider confirmation prompt."
    )
    args = parser.parse_args()

    provider, model = quality_tier_config()
    ceiling_already_reached = args.turn_count >= MAX_FOLLOW_UP_QUESTIONS

    print("=" * 72)
    print("Manual extract_symptoms -> assess_sufficiency test")
    print("=" * 72)
    print(f"  quality-tier provider : {provider or '(not set)'}")
    print(f"  quality-tier model    : {model or '(not set)'}")
    print(f"  starting turn_count   : {args.turn_count}  (ceiling = {MAX_FOLLOW_UP_QUESTIONS})")
    if ceiling_already_reached:
        print(
            "  (turn_count is already at/past the ceiling — assess_sufficiency "
            "will short-circuit and NOT call the LLM; only extract_symptoms will)"
        )
    else:
        print('  (both nodes call tier="quality" — this makes TWO real LLM calls)')
    print()

    if not confirm_billable_call(provider, skip=args.yes):
        return 0

    message = resolve_message(args.message)
    if not message:
        print("No message given. Aborting.")
        return 1

    state = {
        "thread_id": TEST_THREAD_ID,
        "messages": [{"role": "user", "content": message}],
        "medical_record_summary": args.medical_record,
        "turn_count": args.turn_count,
    }

    print("  calling extract_symptoms...")
    print()
    try:
        extraction = extract_symptoms(state)
    except LLMError as exc:
        print("  RESULT: the LLM call failed during extract_symptoms.")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  Check HEALIX_LLM_PROVIDER_QUALITY / HEALIX_MODEL_QUALITY and the")
        print("  matching API key in .env.")
        return 1

    print("=" * 72)
    print("extract_symptoms RESULT")
    print("=" * 72)
    print_symptom_list("symptoms", extraction["symptoms"])
    print()
    print_symptom_list("negated_symptoms", extraction["negated_symptoms"])
    print()
    print_string_list("unmatched_mentions", extraction["unmatched_mentions"])
    print()

    # Feed extract_symptoms's output straight into assess_sufficiency's
    # input — this IS the chaining asked for, done by hand since neither
    # node is wired into graph.py yet to do it for us.
    state.update(extraction)

    print("  calling assess_sufficiency...")
    print()
    try:
        result = assess_sufficiency(state)
    except LLMError as exc:
        print("  RESULT: the LLM call failed during assess_sufficiency.")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  Check HEALIX_LLM_PROVIDER_QUALITY / HEALIX_MODEL_QUALITY and the")
        print("  matching API key in .env.")
        return 1

    # turn_count is only in `result` when assess_sufficiency actually
    # incremented it (the insufficient, not-at-ceiling path) — otherwise
    # it's unchanged from what state already had, same as the real graph
    # would see it (CLAUDE.md > State: assess_sufficiency is turn_count's
    # sole writer).
    final_turn_count = result.get("turn_count", state["turn_count"])

    print("=" * 72)
    print("assess_sufficiency RESULT")
    print("=" * 72)
    print(f"  is_sufficient       : {result['is_sufficient']}")
    print(f"  next_question       : {result['next_question'] or '(none)'}")
    print(f"  information_limited : {result['information_limited']}")
    print(f"  turn_count          : {final_turn_count}  (started at {args.turn_count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
