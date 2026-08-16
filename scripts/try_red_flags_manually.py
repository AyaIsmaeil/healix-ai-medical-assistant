"""MANUAL TEST SCRIPT — chains extract_symptoms -> check_red_flags for a
real end-to-end check, against whatever LLM provider/model is currently
configured for the "quality" tier in .env.

*** This is NOT part of the pytest suite and must never be added to it. ***
Neither node is wired into graph.py yet (CLAUDE.md > Working style: one
node at a time), so this calls both functions directly against a
hand-built state, feeding extract_symptoms's output straight into
check_red_flags's input — no checkpointer, no compiled graph. Run it by
hand:

    python scripts/try_red_flags_manually.py "عندي ألم في الصدر وضيق تنفس من نص ساعة"
    python scripts/try_red_flags_manually.py "عندي ألم في الصدر" --medical-record "مريض سكري من النوع الثاني"
    python scripts/try_red_flags_manually.py "..." --yes   # skip the confirmation
    python scripts/try_red_flags_manually.py               # prompts for a message

--medical-record exercises the rule engine's chronic-condition threshold
lowering (e.g. a diabetes mention drops acs_chest_pain's any_of clause) —
otherwise dead in a script that only ever sends one message with no
record context.

Makes TWO real LLM calls (extract_symptoms, then check_red_flags) — the
confirmation prompt below says so up front, not just "one call" like the
single-node scripts.

See scripts/try_extract_manually.py and scripts/try_crisis_manually.py
for the other nodes in this family of scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from llm_client import LLMError  # noqa: E402
from nodes.check_red_flags import check_red_flags  # noqa: E402
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

TEST_THREAD_ID = "manual-red-flags-test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message", nargs="?", default=None, help="Patient message to test (prompted for if omitted)."
    )
    parser.add_argument(
        "--medical-record",
        default="",
        metavar="SUMMARY",
        help=(
            "Optional chronic-condition record summary, to exercise a rule's "
            "chronic_requirement (e.g. a diabetes mention lowering "
            "acs_chest_pain's threshold). Empty by default."
        ),
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip the billable-provider confirmation prompt."
    )
    args = parser.parse_args()

    provider, model = quality_tier_config()

    print("=" * 72)
    print("Manual extract_symptoms -> check_red_flags test")
    print("=" * 72)
    print(f"  quality-tier provider : {provider or '(not set)'}")
    print(f"  quality-tier model    : {model or '(not set)'}")
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

    # Feed extract_symptoms's output straight into check_red_flags's
    # input — this IS the chaining asked for, done by hand since neither
    # node is wired into graph.py yet to do it for us.
    state.update(extraction)

    print("  calling check_red_flags...")
    print()
    try:
        result = check_red_flags(state)
    except LLMError as exc:
        print("  RESULT: the LLM call failed during check_red_flags.")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  Check HEALIX_LLM_PROVIDER_QUALITY / HEALIX_MODEL_QUALITY and the")
        print("  matching API key in .env.")
        return 1

    print("=" * 72)
    print("check_red_flags RESULT")
    print("=" * 72)
    print_string_list("red_flags", result["red_flags"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
