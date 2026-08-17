"""Shared helpers for scripts/verify_*_enum.py.

Not a standalone script — imported by the per-provider verification
scripts (verify_gemini_enum.py, verify_groq_enum.py). Factored out
specifically because the mismatch-detection and vocabulary-coverage logic
here is safety-relevant: it is what tells you whether exact-string
matching in rules/red_flags.py is actually safe against a given
provider. A fix to that logic belongs in one place, not copy-pasted into
every provider's script where it can drift out of sync silently.

Each verify_*_enum.py script owns its own main(): which provider/tier it
targets, its own header framing, and any provider-specific caveats (e.g.
Groq's structured-output support being model-gated). Only the genuinely
provider-agnostic pieces live here.
"""

from __future__ import annotations

import unicodedata
from contextlib import contextmanager

import llm_client
from rules.crisis import normalize
from schemas.symptoms import SYMPTOM_NAMES
from vocabulary.symptoms import CANONICAL_SYMPTOMS, is_canonical

# Deliberately short and unambiguous: this is testing schema plumbing, not
# extraction quality. Syrian colloquial, roughly "I've had a headache and a
# fever for two days, no cough."
TEST_MESSAGE = "من يومين عندي وجع براسي وحرارة، بس ما في سعال."

# What TEST_MESSAGE should yield if the vocabulary covered it. Kept beside
# the message so the two cannot drift apart. A concept that is absent from
# the vocabulary is one the model is structurally unable to return — so a
# missing extraction is explained by the gap, not by the prompt. Checking
# this BEFORE the call means an uninformative run costs no quota.
EXPECTED_CONCEPTS: tuple[tuple[str, str, str], ...] = (
    ("صداع", "headache", "confirmed"),
    ("حمى", "fever", "confirmed"),
    ("سعال", "cough", "negated"),
)


@contextmanager
def capture_audit_records():
    """Capture what llm_client hands the audit layer, without suppressing it.

    call_llm returns the validated model, not the provider's raw text — but
    the raw text is exactly what these scripts need to see, including on
    the validation-failure path where no model is returned at all. The
    audit layer already receives it, so this taps that rather than
    widening call_llm's public interface for a diagnostic script.
    """
    records: list[dict] = []
    original = llm_client.log_llm_call

    def capturing(**kwargs):
        records.append(kwargs)
        original(**kwargs)

    llm_client.log_llm_call = capturing
    try:
        yield records
    finally:
        llm_client.log_llm_call = original


def report_vocabulary_coverage() -> list[str]:
    """Print which of TEST_MESSAGE's concepts the vocabulary can express."""
    print("  Vocabulary coverage of the test message:")
    absent = []
    for name, gloss, role in EXPECTED_CONCEPTS:
        present = is_canonical(name)
        if not present:
            absent.append(f"{name} ({gloss})")
        near = [v for v in SYMPTOM_NAMES if normalize(name) in v and v != normalize(name)]
        note = f"  ~ closest entry: {near[0]!r}" if near and not present else ""
        print(f"     [{'OK ' if present else 'GAP'}] {name!r} ({gloss}, {role}){note}")

    if absent:
        print()
        print(f"     {len(absent)} of {len(EXPECTED_CONCEPTS)} concepts are NOT in the")
        print("     vocabulary. The enum makes them unreturnable, so their absence")
        print("     from the response is expected and is NOT a prompt defect.")
    print()
    return absent


def codepoints(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def describe_mismatch(returned: str) -> None:
    """Explain *how* a returned value differs from every vocabulary entry."""
    print(f"      returned : {returned!r}")
    print(f"      codepoints: {codepoints(returned)}")

    normalized_hit = next(
        (name for name in SYMPTOM_NAMES if normalize(name) == normalize(returned)), None
    )
    if normalized_hit:
        print("      NOTE: matches a vocabulary entry only AFTER normalization.")
        print(f"      vocabulary: {normalized_hit!r}")
        print(f"      codepoints: {codepoints(normalized_hit)}")
        print("      -> the provider altered the spelling; exact-match would fail.")
        return

    nfc = unicodedata.normalize("NFC", returned)
    nfd = unicodedata.normalize("NFD", returned)
    if nfc in CANONICAL_SYMPTOMS or nfd in CANONICAL_SYMPTOMS:
        print("      NOTE: differs only by Unicode normalization form (NFC/NFD).")
        return

    print("      -> no vocabulary entry corresponds, even loosely.")


def print_raw_response(records: list[dict]) -> None:
    """Dump the provider's response verbatim, before any parsing."""
    print("  " + "-" * 68)
    print("  RAW PROVIDER RESPONSE (verbatim, pre-parse)")
    print("  " + "-" * 68)
    if not records:
        print("    <nothing captured — the call failed before reaching the provider>")
        print()
        return

    record = records[-1]
    raw = record.get("raw_response")
    print(f"    outcome : {record.get('outcome')}")
    print(f"    attempts: {record.get('attempts')}   usage: {record.get('usage')}")
    if record.get("error"):
        print(f"    error   : {record['error']}")
    print()
    if raw is None:
        print("    <no body — the provider never returned one>")
    else:
        print(f"    length  : {len(raw)} chars")
        for line in raw.splitlines() or [""]:
            print(f"    | {line}")
    print()
