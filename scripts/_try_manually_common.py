"""Shared plumbing for scripts/try_*_manually.py.

Not a node, not part of the pytest suite. Factored out once a second
try_*_manually.py script needed the exact UTF-8/confirmation-gate
handling the first one already had — see scripts/try_crisis_manually.py
and scripts/try_extract_manually.py for what each script actually tests.
"""

from __future__ import annotations

import os
import sys


def reconfigure_utf8_stdio() -> None:
    """Windows' default console codepage (cp1252/cp850, not UTF-8)
    otherwise raises or mangles Arabic on the way out (print) and the way
    in (typed or pasted interactive input) — reconfigure both explicitly
    rather than relying on whatever the terminal happens to be set to.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")


# Ollama is free/local; every other provider is billed per call. Default to
# asking for confirmation unless the configured provider is positively
# known to be free — safer than the reverse (skip confirming unless known
# billable), which would silently stop asking the moment an unrecognized
# or misspelled provider name showed up.
FREE_PROVIDERS = {"ollama"}


def quality_tier_config() -> tuple[str, str]:
    """(provider, model) currently configured for the "quality" tier.

    Empty string, not None, when unset — every caller just wants
    something printable and comparable against FREE_PROVIDERS.
    """
    provider = os.getenv("HEALIX_LLM_PROVIDER_QUALITY", "").strip()
    model = os.getenv("HEALIX_MODEL_QUALITY", "").strip()
    return provider, model


def confirm_billable_call(provider: str, *, skip: bool) -> bool:
    """Prints its own prompt/messaging. False means "abort, don't call"."""
    if provider in FREE_PROVIDERS or skip:
        return True
    print(f"  This makes a REAL, BILLABLE call to {provider or '(unset provider)'}.")
    if input("  Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("  Aborted. No API call made.")
        return False
    print()
    return True


def resolve_message(argv_message: str | None, *, prompt: str = "Patient message: ") -> str:
    """The positional argv message, or an interactive prompt if omitted."""
    return argv_message or input(prompt).strip()


def print_symptom_list(label: str, symptoms: list[dict]) -> None:
    """Pretty-print a list of symptom dicts — state["symptoms"]/["negated_symptoms"] shape."""
    print(f"  {label}:")
    if not symptoms:
        print("    (none)")
        return
    for symptom in symptoms:
        details = ", ".join(
            f"{key}={value!r}"
            for key, value in symptom.items()
            if key != "name" and value is not None
        )
        print(f"    - {symptom['name']}" + (f"  ({details})" if details else ""))


def print_string_list(label: str, items: list[str]) -> None:
    """Pretty-print a flat list of strings — unmatched_mentions/red_flags shape."""
    print(f"  {label}:")
    if not items:
        print("    (none)")
        return
    for item in items:
        print(f"    - {item}")
