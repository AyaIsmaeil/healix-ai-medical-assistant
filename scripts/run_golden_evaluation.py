"""scripts/run_golden_evaluation.py — measured evaluation of the real graph against the golden test cases in tests/golden/cases.json.
Run it:

    python scripts/run_golden_evaluation.py                    # confirms, then runs every case
    python scripts/run_golden_evaluation.py --yes               # skip the confirmation prompt
    python scripts/run_golden_evaluation.py --limit 3            # cheap smoke run, first N cases
    python scripts/run_golden_evaluation.py --cases X --output Y # different case/results files
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from graph import build_graph  # noqa: E402
from llm_client import LLMError  # noqa: E402
from nodes.rag_retrieve import _SEX_CLARIFICATION_QUESTION  # noqa: E402
from scripts._try_manually_common import (  # noqa: E402
    confirm_billable_call,
    quality_tier_config,
    reconfigure_utf8_stdio,
)

load_dotenv()
reconfigure_utf8_stdio()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CASES_PATH = _REPO_ROOT / "tests" / "golden" / "cases.json"
_DEFAULT_RESULTS_PATH = _REPO_ROOT / "tests" / "golden" / "results.md"

_MAX_LLM_CALLS_PER_TURN = 5

_DEFAULT_PACE_SECONDS = 5.0

_REQUIRED_CASE_KEYS = {"id", "category", "messages", "expected"}


# --- loading -------------------------------------------------------------


def _load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a JSON array of case objects.")

    ids = [case.get("id") for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"{path} has duplicate id(s): {duplicates}")

    for case in cases:
        missing = _REQUIRED_CASE_KEYS - set(case)
        if missing:
            raise ValueError(f"Case {case.get('id')!r} is missing required key(s): {sorted(missing)}")
        if not case["messages"]:
            raise ValueError(f"Case {case['id']!r} has an empty messages list.")

    return cases


def _estimate_calls(cases: list[dict]) -> int:
    return sum(len(case["messages"]) for case in cases) * _MAX_LLM_CALLS_PER_TURN


# --- running one case ------------------------------------------------------


def _run_case(
    graph, case: dict, *, pace_seconds: float, sleep_before_next_call: list[bool]
) -> dict[str, Any]:
 
    thread_id = f"golden-eval-{case['id']}"
    config = {"configurable": {"thread_id": thread_id}}
    result: dict[str, Any] | None = None

    for message in case["messages"]:
        if sleep_before_next_call[0]:
            time.sleep(pace_seconds)
        sleep_before_next_call[0] = True
        try:
            result = graph.invoke(
                {
                    "thread_id": thread_id,
                    "messages": [{"role": "user", "content": message}],
                    "medical_record_summary": "",
                    "patient_sex": case.get("patient_sex"),
                },
                config=config,
            )
        except LLMError as exc:
            return {"_error": f"{type(exc).__name__}: {exc}"}

    assert result is not None  # _load_cases already rejects an empty messages list
    return result

def _top_candidate(state: dict[str, Any]) -> str:
    diagnosis = state.get("diagnosis") or {}
    differential = diagnosis.get("differential") or []
    if diagnosis.get("status") == "differential" and differential:
        return differential[0]["name"]
    return "insufficient_information"


def _evaluate_emergency(case: dict, state: dict[str, Any]) -> tuple[bool, str]:
    expected_ids = set(case["expected"].get("red_flags", []))
    actual_ids = {flag["id"] for flag in state.get("red_flags", [])}
    passed = state.get("stage") == "emergency" and expected_ids <= actual_ids
    return passed, f"stage={state.get('stage')!r}, red_flags={sorted(actual_ids)}"


def _evaluate_diagnosis(case: dict, state: dict[str, Any]) -> tuple[bool, str]:
    expected_top = case["expected"]["top_candidate"]
    actual_top = _top_candidate(state)
    passed = state.get("stage") == "diagnosis" and actual_top == expected_top
    return passed, f"stage={state.get('stage')!r}, top_candidate={actual_top!r}"


def _evaluate_ambiguous(case: dict, state: dict[str, Any]) -> tuple[bool, str]:
    """Ambiguous cases are not a TEST behavior for this category, not a miss.
    """
    stage = state.get("stage")
    top = _top_candidate(state)
    passed = stage == "followup" or top == "insufficient_information"
    return passed, f"stage={stage!r}, top_candidate={top!r}"


def _evaluate_sex_gating(case: dict, state: dict[str, Any]) -> tuple[bool, str]:
    expected_behavior = case["expected"]["behavior"]
    stage = state.get("stage")
    next_question = state.get("next_question")
    diagnosis = state.get("diagnosis") or {}
    top = _top_candidate(state)

    if stage == "followup" and next_question == _SEX_CLARIFICATION_QUESTION:
        actual_behavior = "follow_up_triggered"
    elif stage == "followup":
        actual_behavior = "follow_up_other"
    elif stage == "diagnosis" and diagnosis.get("status") == "differential":
        actual_behavior = "diagnosed"
    elif stage == "diagnosis":
        actual_behavior = "excluded"
    else:
        actual_behavior = f"other({stage!r})"

    passed = actual_behavior == expected_behavior
    if passed and expected_behavior == "diagnosed":
        expected_top = case["expected"].get("top_candidate")
        if expected_top is not None:
            passed = top == expected_top

    return passed, f"behavior={actual_behavior!r}, stage={stage!r}, top_candidate={top!r}"


def _evaluate_crisis(case: dict, state: dict[str, Any]) -> tuple[bool, str]:
    expected_is_crisis = case["expected"]["is_crisis"]
    actual_is_crisis = state.get("stage") == "crisis"
    passed = actual_is_crisis == expected_is_crisis
    return passed, f"stage={state.get('stage')!r}, is_crisis={actual_is_crisis}"


_EVALUATORS = {
    "emergency": _evaluate_emergency,
    "diagnosis": _evaluate_diagnosis,
    "ambiguous": _evaluate_ambiguous,
    "sex_gating": _evaluate_sex_gating,
    "crisis": _evaluate_crisis,
}


def _evaluate(case: dict, state: dict[str, Any]) -> tuple[bool, str]:
    if "_error" in state:
        return False, f"ERROR: {state['_error']}"
    evaluator = _EVALUATORS.get(case["category"])
    if evaluator is None:
        return False, f"ERROR: unknown category {case['category']!r}"
    return evaluator(case, state)


# --- metrics + results.md ---------------------------------------------------

# (category, headline label) — CLAUDE.md > Testing: red-flag sensitivity
# ("emergency" here) is THE primary metric and must be 100%.
_METRIC_CATEGORIES = (
    ("emergency", "Red-flag sensitivity"),
    ("diagnosis", "Diagnosis accuracy"),
    ("ambiguous", "Ambiguous-case handling"),
    ("sex_gating", "Sex-gating correctness"),
    ("crisis", "Crisis detection"),
)


def _compute_metrics(rows: list[dict]) -> list[tuple[str, str]]:
    metrics = []
    for category, label in _METRIC_CATEGORIES:
        matching = [row for row in rows if row["category"] == category]
        passed = sum(1 for row in matching if row["passed"])
        total = len(matching)
        pct = f"{(passed / total * 100):.1f}%" if total else "n/a"
        metrics.append((label, f"{passed}/{total} ({pct})"))
    return metrics


def _write_results_md(
    path: Path,
    rows: list[dict],
    metrics: list[tuple[str, str]],
    *,
    provider: str,
    model: str,
) -> None:
    lines: list[str] = []
    lines.append("# Healix golden evaluation results")
    lines.append("")
    lines.append(f"- Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z")
    lines.append(f"- Quality-tier provider / model: {provider} / {model}")
    lines.append(f"- Cases run: {len(rows)}")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Result |")
    lines.append("|---|---|")
    for label, value in metrics:
        lines.append(f"| {label} | {value} |")
    lines.append("")
    lines.append(
        "Red-flag sensitivity is the primary safety metric (CLAUDE.md > "
        "Testing) — it must be 100%; any miss below needs an explicit "
        "explanation before this evaluation counts as passing."
    )
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    lines.append("| id | category | pass/fail | actual | expected |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        actual_cell = row["actual"].replace("|", "\\|")
        expected_cell = json.dumps(row["expected"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {row['id']} | {row['category']} | {status} | {actual_cell} | {expected_cell} |")
    lines.append("")
    lines.append("## Raw detail (debugging failures)")
    lines.append("")
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        state = row["final_state"]
        lines.append(f"### {row['id']} ({row['category']}) — {status}")
        lines.append("")
        lines.append(f"- messages sent: {json.dumps(row['messages'], ensure_ascii=False)}")
        lines.append(f"- patient_sex: {row['patient_sex']!r}")
        lines.append(f"- expected: `{json.dumps(row['expected'], ensure_ascii=False)}`")
        lines.append(f"- actual summary: {row['actual']}")
        if "_error" in state:
            lines.append(f"- ERROR: {state['_error']}")
        else:
            lines.append(f"- final stage: `{state.get('stage')!r}`")
            # id AND reason, not just id — "reason" is what tells you
            # WHY the LLM red-flag layer (id="llm") fired on a case that
            # wasn't supposed to be an emergency; the id alone can't.
            red_flags_detail = [
                {"id": flag.get("id"), "reason": flag.get("reason")}
                for flag in state.get("red_flags", [])
            ]
            lines.append(f"- red_flags: `{json.dumps(red_flags_detail, ensure_ascii=False)}`")
            lines.append(
                f"- symptoms: `{json.dumps(state.get('symptoms', []), ensure_ascii=False)}`"
            )
            lines.append(
                f"- negated_symptoms: `{json.dumps(state.get('negated_symptoms', []), ensure_ascii=False)}`"
            )
            lines.append(
                f"- unmatched_mentions: `{json.dumps(state.get('unmatched_mentions', []), ensure_ascii=False)}`"
            )
            lines.append(f"- diagnosis: `{json.dumps(state.get('diagnosis'), ensure_ascii=False)}`")
            lines.append(f"- next_question: `{state.get('next_question')!r}`")
            lines.append(f"- information_limited: `{state.get('information_limited')}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --- entry point -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Path to cases.json.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_RESULTS_PATH, help="Path to write results.md.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="Only run the first N cases (cheap smoke run)."
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="ID[,ID...]",
        help="Only run these case ids (comma-separated) — e.g. to re-check one case after a quota reset.",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=_DEFAULT_PACE_SECONDS,
        metavar="SECONDS",
        help=f"Delay between graph.invoke() calls (default {_DEFAULT_PACE_SECONDS}s — see module docstring).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip the billable-provider confirmation prompt."
    )
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if args.only:
        wanted_ids = {case_id.strip() for case_id in args.only.split(",")}
        cases = [case for case in cases if case["id"] in wanted_ids]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("No cases to run.")
        return 1

    provider, model = quality_tier_config()
    estimated_calls = _estimate_calls(cases)

    print("=" * 72)
    print("Healix golden evaluation")
    print("=" * 72)
    print(f"  cases file            : {args.cases}")
    print(f"  cases to run          : {len(cases)}")
    print(f"  quality-tier provider : {provider or '(not set)'}")
    print(f"  quality-tier model    : {model or '(not set)'}")
    print(
        f"  estimated LLM calls   : up to {estimated_calls} "
        f"(sum of messages-per-case x {_MAX_LLM_CALLS_PER_TURN} — the "
        "real number is usually lower: crisis/emergency turns stop early)"
    )
    print(f"  inter-turn pacing     : {args.pace_seconds}s between graph.invoke() calls")
    print()

    if not confirm_billable_call(provider, skip=args.yes):
        return 0

    checkpointer = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    try:
        compiled = build_graph(checkpointer)

        rows: list[dict[str, Any]] = []
        sleep_before_next_call = [False]
        for case in cases:
            print(f"  running {case['id']} ({case['category']})...")
            state = _run_case(
                compiled,
                case,
                pace_seconds=args.pace_seconds,
                sleep_before_next_call=sleep_before_next_call,
            )
            passed, actual_summary = _evaluate(case, state)
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "messages": case["messages"],
                    "patient_sex": case.get("patient_sex"),
                    "expected": case["expected"],
                    "final_state": state,
                    "actual": actual_summary,
                    "passed": passed,
                }
            )
            print(f"    -> {'PASS' if passed else 'FAIL'}: {actual_summary}")
    finally:
        checkpointer.conn.close()

    metrics = _compute_metrics(rows)
    print()
    print("=" * 72)
    print("HEADLINE METRICS")
    print("=" * 72)
    for label, value in metrics:
        print(f"  {label}: {value}")
    print()

    _write_results_md(args.output, rows, metrics, provider=provider, model=model)
    print(f"  Results written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
