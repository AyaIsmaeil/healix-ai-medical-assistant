"""MANUAL TEST SCRIPT — calls nodes.extract_symptoms.extract_symptoms
directly, against whatever LLM provider/model is currently configured for
the "quality" tier in .env.

*** This is NOT part of the pytest suite and must never be added to it. ***
This calls extract_symptoms directly against a hand-built state rather
than invoking the compiled graph — useful for isolating one LLM call.
Run it by hand:

    python scripts/try_extract_manually.py "عندي صداع وحرارة من يومين، بس ما عندي سعال"
    python scripts/try_extract_manually.py "..." --yes   # skip the confirmation
    python scripts/try_extract_manually.py               # prompts for a message

See scripts/try_crisis_manually.py for the same idea against the crisis
path — a separate script, not a flag on this one, because that one
invokes a real compiled graph and this one doesn't.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from llm_client import LLMError  # noqa: E402
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

TEST_THREAD_ID = "manual-extract-test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message", nargs="?", default=None, help="Patient message to test (prompted for if omitted)."
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip the billable-provider confirmation prompt."
    )
    args = parser.parse_args()

    provider, model = quality_tier_config()

    print("=" * 72)
    print("Manual extract_symptoms test")
    print("=" * 72)
    print(f"  quality-tier provider : {provider or '(not set)'}")
    print(f"  quality-tier model    : {model or '(not set)'}")
    print('  (extract_symptoms calls tier="quality" — CLAUDE.md safety rule 12)')
    print()

    if not confirm_billable_call(provider, skip=args.yes):
        return 0

    message = resolve_message(args.message)
    if not message:
        print("No message given. Aborting.")
        return 1

    state = {"thread_id": TEST_THREAD_ID, "messages": [{"role": "user", "content": message}]}

    print("  calling extract_symptoms...")
    print()
    try:
        result = extract_symptoms(state)
    except LLMError as exc:
        print("  RESULT: the LLM call failed.")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  Check HEALIX_LLM_PROVIDER_QUALITY / HEALIX_MODEL_QUALITY and the")
        print("  matching API key in .env.")
        return 1

    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print_symptom_list("symptoms", result["symptoms"])
    print()
    print_symptom_list("negated_symptoms", result["negated_symptoms"])
    print()
    print_string_list("unmatched_mentions", result["unmatched_mentions"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
