"""MANUAL TEST SCRIPT — invokes the real crisis path (crisis_check ->
crisis_node) end to end, against whatever LLM provider/model is
currently configured for the "quality" tier in .env.

*** This is NOT part of the pytest suite and must never be added to it. ***
Unlike scripts/verify_*_enum.py, this has no pass/fail verdict — it exists
to let a person read the actual response a real message produces. Run it
by hand:

    python scripts/try_crisis_manually.py "بدي موت"
    python scripts/try_crisis_manually.py "بدي موت" --yes   # skip the confirmation
    python scripts/try_crisis_manually.py                   # prompts for a message

The graph runs against a throwaway in-memory SQLite checkpointer built
directly here, not graph.build_checkpointer() — a one-off manual run has
no business writing into whatever HEALIX_POSTGRES_DSN / HEALIX_SQLITE_PATH
point at in .env.

See scripts/try_extract_manually.py for the same idea against
extract_symptoms — a separate script, not a flag on this one, because
that node isn't wired into graph.py yet and is called directly instead of
through a compiled graph.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from graph import build_graph  # noqa: E402
from llm_client import LLMError  # noqa: E402
from scripts._try_manually_common import (  # noqa: E402
    confirm_billable_call,
    quality_tier_config,
    reconfigure_utf8_stdio,
    resolve_message,
)

load_dotenv()
reconfigure_utf8_stdio()

TEST_THREAD_ID = "manual-crisis-test"


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
    print("Manual crisis-path test")
    print("=" * 72)
    print(f"  quality-tier provider : {provider or '(not set)'}")
    print(f"  quality-tier model    : {model or '(not set)'}")
    print('  (crisis_check and crisis_node both call tier="quality")')
    print()

    if not confirm_billable_call(provider, skip=args.yes):
        return 0

    message = resolve_message(args.message)
    if not message:
        print("No message given. Aborting.")
        return 1

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        compiled = build_graph(SqliteSaver(conn))
        print(f"  invoking graph (thread_id={TEST_THREAD_ID!r})...")
        print()
        try:
            result = compiled.invoke(
                {
                    "thread_id": TEST_THREAD_ID,
                    "messages": [{"role": "user", "content": message}],
                },
                config={"configurable": {"thread_id": TEST_THREAD_ID}},
            )
        except LLMError as exc:
            print("  RESULT: the LLM call failed.")
            print(f"    {type(exc).__name__}: {exc}")
            print()
            print("  Check HEALIX_LLM_PROVIDER_QUALITY / HEALIX_MODEL_QUALITY and the")
            print("  matching API key in .env.")
            return 1
    finally:
        conn.close()

    last_assistant = next(
        (m["content"] for m in reversed(result["messages"]) if m.get("role") == "assistant"),
        None,
    )

    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  is_crisis : {result.get('is_crisis')}")
    print(f"  stage     : {result.get('stage')}")
    print()
    print("  last assistant message:")
    print("  " + "-" * 68)
    if last_assistant is None:
        print("  (none — the graph ended without crisis_node producing a reply)")
    else:
        for line in last_assistant.splitlines():
            print(f"  {line}")
    print("  " + "-" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
