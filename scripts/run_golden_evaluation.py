"""scripts/run_golden_evaluation.py — measured evaluation of the real
compiled graph (graph.build_graph()) against tests/golden/cases.json.

Distinct from tests/unit and tests/integration (CLAUDE.md > Project
layout: tests/golden/ holds "gold-standard test cases" for measured
evaluation, not pass/fail CI). This script is not part of the pytest
suite and must never be added to it: every case makes REAL LLM calls
against whatever provider .env currently configures for the "quality"
tier (CLAUDE.md > Non-negotiable safety rule 12 — every node that reads
the patient's raw Arabic message uses that tier: crisis_check,
extract_symptoms, check_red_flags, assess_sufficiency, diagnose). Same
"real graph, real calls" reasoning as scripts/try_full_chain_manually.py,
just run over a fixed, scored case set instead of driven interactively.

One turn can make up to five quality-tier LLM calls (crisis/emergency
turns stop earlier — see graph.py's own routing). A case with N messages
sends up to N turns on the same thread_id, so total calls scale with the
whole case file, not just its length. The estimate is printed and, like
every other scripts/*_manually.py script, gated behind
scripts._try_manually_common.confirm_billable_call before anything runs
— the quality tier this script exercises is never Ollama (CLAUDE.md > LLM
tiers: "HEALIX_LLM_PROVIDER_QUALITY must never be ollama"), so treat the
estimate as always-real quota against a hosted provider's free tier
(CLAUDE.md: "Free-tier providers only"), not literally free to run
without limit.

Isolation: a dedicated in-memory SqliteSaver, never
graph.build_checkpointer()'s own dev/prod file — a golden run's
thread_ids (golden-eval-<case id>) must never mix with real conversation
state, and re-running the same case twice must start that thread fresh
rather than resuming leftover state from a previous run.

--- Case schema (tests/golden/cases.json): a JSON array of objects —

    id: str                    unique; used to build this run's thread_id
    category: "emergency" | "diagnosis" | "ambiguous" | "sex_gating" | "crisis"
    messages: list[str]        patient turns, sent in order on one thread_id
    patient_sex: "male" | "female" | null
    expected: dict — shape depends on category:
        emergency:   {"red_flags": [rule_id, ...]}
            Pass: the turn reached emergency_node (stage == "emergency")
            AND every listed rule_id is among the ones that actually
            fired. Extra rule_ids firing alongside the expected ones are
            not penalized — real presentations often satisfy more than
            one rule at once.
        diagnosis:   {"top_candidate": "<Disease name>"}
            Pass: the turn reached generate_reports (stage == "diagnosis")
            AND diagnosis["differential"][0]["name"] equals this exactly.
            "name", not "name_ar" — the English/Latin key every
            rag/knowledge_base/*.json entry is keyed by, so the case file
            doesn't have to duplicate the Arabic translation to state
            what disease is expected.
        ambiguous:   {"top_candidate": "insufficient_information"} (informational —
                      see _evaluate_ambiguous for the actual, more lenient
                      pass condition)
        sex_gating:  {"behavior": "excluded" | "follow_up_triggered" | "diagnosed",
                       "top_candidate": "<Disease name>"}      (only used when behavior == "diagnosed")
            "diagnosed": reached generate_reports with a real differential
            (diagnosis["status"] == "differential") — top_candidate checked
            the same way as the diagnosis category above, when given.
            "follow_up_triggered": stage == "followup" AND next_question
            is EXACTLY nodes.rag_retrieve._SEX_CLARIFICATION_QUESTION —
            not just any follow-up (see "follow_up_other" below).
            "excluded": reached generate_reports but diagnosis["status"]
            == "insufficient_information" — the sex-restricted candidate
            was excluded outright (never asked about, never diagnosed).
            No excluded-candidate name is required in the case file:
            candidate_diseases never contains an excluded entry in the
            first place (nodes/rag_retrieve.py excludes it before
            construction, not merely downranks it), so there is nothing
            to name-check here beyond the outcome itself.
        crisis:      {"is_crisis": true}

Only the FINAL message's outcome is scored. A multi-turn case is free to
have assess_sufficiency ask a real follow-up on an earlier turn — that's
expected, normal behavior, not something to special-case. What matters
is the state after the last message in the list.

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

# Matches scripts/try_full_chain_manually.py's own worst-case-per-turn
# estimate (crisis_check, extract_symptoms, check_red_flags,
# assess_sufficiency, diagnose) — an upper bound, not an exact count:
# crisis/emergency turns stop well before all five run.
_MAX_LLM_CALLS_PER_TURN = 5

# Default inter-turn pacing. Observed LIVE against the real
# HEALIX_LLM_PROVIDER_QUALITY=gemini / HEALIX_MODEL_QUALITY=gemini-3.1-flash-lite
# free tier while validating this script: back-to-back cases hit
# "429 RESOURCE_EXHAUSTED ... generativelanguage.googleapis.com/generate_content_free_tier_requests,
# limit: 15" within the first four cases — a single turn already spends up
# to _MAX_LLM_CALLS_PER_TURN calls, so a handful of turns run
# back-to-back exhausts a 15-per-minute budget almost immediately.
# 60s / 15 = 4s minimum spacing; padded to 5s for margin. This only
# throttles the gap BETWEEN graph.invoke() calls (i.e. between turns/
# cases) — the up-to-five calls INSIDE one invoke() still fire back-to-
# back, since llm_client.call_llm() is what would need to change to pace
# those, which is out of scope here. Configurable because a different
# provider/model has a different real limit, not because this number is
# a guess.
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
    """Send every message in case["messages"] in order on one thread_id,
    returning the FINAL turn's full state dict — or {"_error": ...} if a
    call failed partway through (a real LLMError, not caught by this
    script's own logic elsewhere).

    sleep_before_next_call is a one-element mutable flag shared across
    every case in the run: False only for the very first graph.invoke()
    of the whole run (nothing to pace against yet), True for every one
    after — see _DEFAULT_PACE_SECONDS for why this pacing exists at all.
    """
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


# --- per-category evaluators ------------------------------------------------
# Each returns (passed, actual_summary) — actual_summary is a short,
# category-specific string for the results.md table; the full state dict
# is kept separately for the raw debug section.


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
    """Pass = asked a clarifying question OR honestly returned
    insufficient_information — never a confident guess (CLAUDE.md >
    Testing: "Ambiguous cases must produce a clarifying question or
    insufficient_information — not a guess"). A strict exact-match
    against expected["top_candidate"] would wrongly fail a case that
    correctly kept asking instead of settling on
    insufficient_information within the turns given — that is the
    RIGHT behavior for this category, not a miss.
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
        # Asked something, but not the sex-clarification question —
        # e.g. assess_sufficiency's own follow-up loop, not
        # rag_retrieve's. Distinct from "follow_up_triggered" on
        # purpose: a sex_gating case expecting that behavior means
        # specifically the sex question, not any question.
        actual_behavior = "follow_up_other"
    elif stage == "diagnosis" and diagnosis.get("status") == "differential":
        actual_behavior = "diagnosed"
    elif stage == "diagnosis":
        # Reached the terminal node but with no differential — for a
        # sex_gating case, this is the sex-restricted candidate having
        # been excluded outright (state.py: excluded, not merely
        # downranked, so nothing else clears the match floor here since
        # these cases share one symptom set with only that one KB match).
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
