"""Audit logging entry point.

CLAUDE.md > Conventions: every LLM call is logged with prompt name +
version, model, raw output, timestamp, thread_id. This module is the
sink-agnostic front for that — callers should not need to change when
the backing store becomes something other than stdlib logging.

Nothing here may raise into a caller. An audit failure must never be the
reason a medical turn fails, so `llm_client` wraps these calls too; the
belt-and-braces is deliberate.

*** WHAT THESE RECORDS CONTAIN ***

Audit records hold patient data and are handled like medical records, not
like telemetry: `raw_response` is the patient's own account of their
symptoms in Arabic, and `thread_id` maps back through Laravel to a
specific user. Do not commit them, forward them to an external log
aggregator or error tracker, or quote them in the project report without
redaction. See CLAUDE.md > Audit logs and patient data — that section is
binding, not advisory. Any new field added here should be assumed to
carry patient data unless it demonstrably cannot.

*** DELIVERY MECHANISM (fixed after being a no-op from initial
implementation until this session's investigation — see CLAUDE.md >
Audit logs and patient data for the full incident writeup) ***

Every function below existed and was called correctly from day one, but
`logging.getLogger("healix.audit")` had no handler attached anywhere in
its hierarchy, and nothing in this project ever ran `basicConfig()` or
otherwise configured one — not `graph.py`, not `api/main.py`, not even
under `uvicorn api.main:app` (uvicorn's own default LOGGING_CONFIG only
touches its own "uvicorn"/"uvicorn.error"/"uvicorn.access" loggers,
verified directly against uvicorn/config.py). Consequence: every
`log_llm_call`/`log_crisis_detection`/`log_red_flag_detection` call
(logged at INFO) was silently discarded by Python's `logging` module —
formatted, dispatched, and dropped with no error, since a logger with no
handler anywhere in its chain falls back to `logging.lastResort`, a bare
stderr handler fixed at WARNING and therefore blind to INFO. Worse,
`log_malformed_output` (logged at WARNING, at or above that floor) was
NOT silently dropped — it was live-printing its `payload` (which can
carry a patient's verbatim `raw_mention` text, e.g. via
`state.merge_symptoms`) unredacted to stderr, the opposite failure mode
from "no logs at all."

_configure_handler() below attaches a real logging.FileHandler to
`"healix.audit"` SPECIFICALLY — not root, not any other logger — at
INFO, so all four functions' records now land in one file instead of
either being dropped (INFO) or leaking to stderr (WARNING).
`propagate = False` is set so records never reach root, or anything a
future `logging.basicConfig()`/third-party import might attach to root —
this fix cannot be silently reopened by an unrelated later change the
way the original gap was invisible for this long.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger("healix.audit")

# One of the exact patterns .gitignore already reserves for this (see
# that file's own comment there, written in anticipation of this fix:
# "The current sink is stdlib logging, which writes nowhere by default;
# these patterns are here so that adding a file handler later cannot
# accidentally commit patient data."). Overridable via
# HEALIX_AUDIT_LOG_PATH — same "configuration, never hardcoded" pattern
# as graph.py's HEALIX_SQLITE_PATH — but the override changes only WHERE
# the file lives, not the level, format, or what gets logged; retention/
# rotation policy is deliberately NOT decided here (CLAUDE.md > Audit
# logs and patient data: "Log retention is a decision, not a default" —
# still true, still open, not something this fix invents an answer to).
_DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.log"


def _configure_handler() -> None:
    """Attach a file handler to "healix.audit" so its records actually
    persist. Runs once at import time (module-level side effect, same
    pattern as rules/red_flags.py's _validate_rule_symptoms_are_canonical)
    — Python only executes a module's top level on the first import per
    process, so every caller across the codebase (nodes/, llm_client.py,
    schemas/, state.py) shares this one configuration, not a duplicate
    handler per import site. The handler-presence check below is extra
    insurance against a second configuration (e.g. a module reload)
    attaching a duplicate handler and doubling every record written.
    """
    if _logger.handlers:
        return

    log_path = Path(os.getenv("HEALIX_AUDIT_LOG_PATH", "").strip() or _DEFAULT_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    # Isolates this logger from root (and anything attached to root
    # later) — see module docstring's "DELIVERY MECHANISM" section for
    # why this matters specifically for log_malformed_output's WARNING-
    # level, patient-bearing records.
    _logger.propagate = False


_configure_handler()


def log_malformed_output(*, node: str, reason: str, payload: dict[str, Any]) -> None:
    """Record output that failed validation instead of raising.

    node: where the bad data was caught (e.g. "merge_symptoms").
    reason: short machine-readable cause (e.g. "missing_name").
    payload: the offending data, as-is, for later inspection.
    """
    _logger.warning("malformed_output node=%s reason=%s payload=%r", node, reason, payload)


def log_llm_call(
    *,
    prompt_name: str,
    tier: str,
    model: str,
    provider: str,
    latency_ms: float,
    outcome: str,
    raw_response: str | None = None,
    usage: dict[str, int] | None = None,
    attempts: int = 1,
    thread_id: str | None = None,
    error: str | None = None,
    ended_by: str | None = None,
) -> None:
    """Record one LLM call attempt-chain.

    outcome: "success" | "validation_error" | "unavailable".
    ended_by: which limit ended the call — "success", "validation",
        "attempts" (retry budget spent), "budget" (wall-clock budget
        spent), or "provider_error" (non-retryable). Distinguishing these
        is what tells you whether to raise the retry count, raise the
        budget, or fix the request; latency_ms alone cannot.
    raw_response: the provider's raw text, kept verbatim — the evaluation
        chapter needs the unedited output, not a summary of it.
    usage: token counts when the provider reports them; providers are not
        required to, so this stays optional rather than being faked.
    """
    _logger.info(
        "llm_call ts=%s prompt=%s tier=%s provider=%s model=%s outcome=%s "
        "ended_by=%s latency_ms=%.1f attempts=%d thread_id=%s usage=%r "
        "error=%s raw=%r",
        datetime.now(timezone.utc).isoformat(),
        prompt_name,
        tier,
        provider,
        model,
        outcome,
        ended_by,
        latency_ms,
        attempts,
        thread_id,
        usage,
        error,
        raw_response,
    )


def log_crisis_detection(
    *,
    thread_id: str | None,
    rule_matched: bool,
    rule_categories: list[str],
    llm_matched: bool,
    llm_reasoning: str | None,
    combined: bool,
) -> None:
    """Record both crisis-detection layers' verdicts side by side.

    CLAUDE.md > Non-negotiable safety rule 3: the deterministic rule layer
    and the LLM are combined with OR, and neither is ever weakened by the
    other. Logging them separately here, not just `combined`, is what lets
    later analysis measure how often each layer catches something the
    other missed — the combined bool alone can't answer that, since it
    collapses "both agreed" and "only one fired" into the same True.

    This is distinct from the log_llm_call record the LLM call itself
    already produces (prompt/model/raw response) — this record is about
    the two verdicts and how they compare, not the call that produced one
    of them.
    """
    _logger.info(
        "crisis_detection ts=%s thread_id=%s rule_matched=%s rule_categories=%r "
        "llm_matched=%s llm_reasoning=%r combined=%s",
        datetime.now(timezone.utc).isoformat(),
        thread_id,
        rule_matched,
        rule_categories,
        llm_matched,
        llm_reasoning,
        combined,
    )


def log_red_flag_detection(
    *,
    thread_id: str | None,
    rule_matched: bool,
    rule_ids: list[str],
    llm_matched: bool,
    llm_reasoning: str | None,
    combined: bool,
    candidate_rule_ids: list[str] | None = None,
    safety_decision: str | None = None,
) -> None:
    """Record every red-flag-detection layer's verdict side by side, plus
    the final explicit disposition (nodes/check_red_flags.py's module
    docstring: candidate/confirmed architecture, not an implicit OR).

    rule_matched/rule_ids/combined: the deterministic hard-match layer
    (rules.red_flags.check_red_flags) — unchanged meaning from before this
    module's disposition rewrite. llm_matched/llm_reasoning: the LLM's
    potential_red_flag SCREEN — logged for audit/analysis (measuring how
    often it flags something the deterministic layers miss entirely), but
    since nodes/check_red_flags.py's rewrite it no longer independently
    drives `combined` or the disposition below — see that module's own
    docstring for why (schemas/red_flags.py: "not authorized to confirm
    an emergency").

    candidate_rule_ids/safety_decision are new (previously absent):
    candidate_rule_ids is the deterministic Layer 1b's own output
    (rules.red_flags.find_incomplete_combination_candidates, not-yet-
    rejected only); safety_decision is the explicit
    HARD_EMERGENCY/NEEDS_CLARIFICATION/NO_RED_FLAG disposition
    (state.SafetyDecision) graph.py's routing actually reads. Optional
    with a None default so this function's signature does not break any
    other caller that predates this field — there is none today (checked:
    nodes/check_red_flags.py is this function's only caller), but the
    same "additive, not breaking" discipline applies here as everywhere
    else in this project.

    rule_ids, not rule_categories: rules/red_flags.py has multiple rules
    sharing a category (e.g. two "neuro" rules), so the rule_id is the
    more specific, traceable identifier of what actually fired.
    """
    _logger.info(
        "red_flag_detection ts=%s thread_id=%s rule_matched=%s rule_ids=%r "
        "llm_matched=%s llm_reasoning=%r combined=%s candidate_rule_ids=%r "
        "safety_decision=%s",
        datetime.now(timezone.utc).isoformat(),
        thread_id,
        rule_matched,
        rule_ids,
        llm_matched,
        llm_reasoning,
        combined,
        candidate_rule_ids or [],
        safety_decision,
    )


def log_negation_detection(
    *,
    thread_id: str | None,
    rule_negated: list[str],
    llm_negated: list[str],
    combined: list[str],
) -> None:
    """Record both negation-detection layers' verdicts side by side.

    Same rationale as log_crisis_detection/log_red_flag_detection
    (CLAUDE.md > Non-negotiable safety rule 3's rule-based-first,
    LLM-second, OR-combined pattern — applied here to
    nodes/extract_symptoms.py's negation detection, not just
    crisis/red-flags): logging the two verdicts separately, not just
    `combined`, is what lets later analysis measure how often each layer
    catches a negation the other missed.

    No rule_matched/llm_matched booleans and no reasoning field, unlike
    the two sibling functions above — the shapes genuinely differ, not
    an inconsistency: negation detection's verdict IS the set of names
    each layer found. rules.negation.detect_negated_symptoms returns a
    bare frozenset with no reasoning text of its own, and
    schemas.symptoms.NegatedSymptom carries no reasoning field either.
    A derived "did anything fire" boolean would just be
    `bool(rule_negated)` / `bool(llm_negated)` — redundant with the
    lists themselves, not an independent signal the way a crisis/red-flag
    reasoning string is.
    """
    _logger.info(
        "negation_detection ts=%s thread_id=%s rule_negated=%r llm_negated=%r combined=%r",
        datetime.now(timezone.utc).isoformat(),
        thread_id,
        rule_negated,
        llm_negated,
        combined,
    )
