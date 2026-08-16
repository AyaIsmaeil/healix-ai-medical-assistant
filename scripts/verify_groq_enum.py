"""MANUAL VERIFICATION SCRIPT — MAKES ONE REAL, BILLABLE GROQ API CALL.

*** This is NOT part of the pytest suite and must never be added to it. ***
Run it by hand, deliberately:

    python scripts/verify_groq_enum.py                      # gpt-oss-20b, asks for confirmation
    python scripts/verify_groq_enum.py --yes                 # skips the prompt
    python scripts/verify_groq_enum.py --model openai/gpt-oss-120b --yes

Companion to scripts/verify_gemini_enum.py — same three questions, same
checklist, against Groq instead:

  1. Does the provider ACCEPT a schema with an enum this large?
  2. Do non-Latin (Arabic) enum values survive the round trip?
  3. Does the returned value match a vocabulary entry BYTE FOR BYTE?

*** Why this is a SEPARATE script, not a --provider flag on the Gemini
one: Groq's structured-output support is NOT provider-wide. As of this
writing, native JSON Schema enforcement (response_format: json_schema,
strict: true) is documented as supported only on openai/gpt-oss-20b and
openai/gpt-oss-120b — see llm_client._GROQ_STRICT_SCHEMA_MODELS and
https://console.groq.com/docs/structured-outputs. Every other Groq model
either has no schema enforcement or best-effort-only, which is exactly
the kind of silent downgrade this project refuses to build around. A
PASS here is a claim about these two specific models, not about "Groq."

This script forces the "fast" tier to groq / the chosen model for its own
process only (HEALIX_LLM_PROVIDER_FAST / HEALIX_MODEL_FAST are set in
os.environ after load_dotenv(), never written to .env) — so it verifies
the same two models regardless of whatever your .env currently has each
tier pointed at.

Requires GROQ_API_KEY in the environment (.env). See
https://console.groq.com/keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Forced before importing llm_client, so the module-level provider/config
# state it builds is consistent with these from the very first call —
# llm_client itself doesn't cache anything at import time, but keeping the
# override adjacent to load_dotenv() keeps the "this process only, this
# tier only" scope obvious to a reader.
_DEFAULT_MODEL = "openai/gpt-oss-20b"
os.environ["HEALIX_LLM_PROVIDER_FAST"] = "groq"
os.environ.setdefault("HEALIX_MODEL_FAST", _DEFAULT_MODEL)

from llm_client import (  # noqa: E402
    LLMError,
    _GROQ_STRICT_SCHEMA_MODELS,
    call_llm,
)
from schemas.symptoms import SymptomExtraction  # noqa: E402
from scripts._verify_enum_common import (  # noqa: E402
    EXPECTED_CONCEPTS,
    TEST_MESSAGE,
    capture_audit_records,
    describe_mismatch,
    print_raw_response,
    report_vocabulary_coverage,
)
from vocabulary.symptoms import CANONICAL_SYMPTOMS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        choices=sorted(_GROQ_STRICT_SCHEMA_MODELS),
        help="Which of the two verified-support models to test (default: %(default)s).",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    os.environ["HEALIX_MODEL_FAST"] = args.model

    schema_json = json.dumps(SymptomExtraction.model_json_schema(), ensure_ascii=False)
    enum_values = SymptomExtraction.model_json_schema()["$defs"]["ExtractedSymptom"][
        "properties"
    ]["name"]["enum"]

    print("=" * 72)
    print("Groq structured-output enum verification")
    print("=" * 72)
    print(f"  model              : {args.model}")
    print(f"  vocabulary entries : {len(CANONICAL_SYMPTOMS)}")
    print(f"  enum length        : {len(enum_values)}")
    print(f"  schema size        : {len(schema_json.encode('utf-8'))} bytes "
          f"({len(schema_json)} chars)")
    print()
    print("  Watch these two numbers as the vocabulary grows toward 52 —")
    print("  provider limits surface as a rejected request, not a warning.")
    print()
    print("  Structured-output support on Groq is MODEL-SPECIFIC, not")
    print(f"  provider-wide. Verified models: {sorted(_GROQ_STRICT_SCHEMA_MODELS)}.")
    print("  A PASS below is a claim about this one model, not about Groq")
    print("  in general — llm_client refuses schema + any other Groq model")
    print("  outright (see _GroqProvider.generate), so that combination")
    print("  cannot silently reach production regardless of this result.")
    print()

    absent_concepts = report_vocabulary_coverage()

    if not args.yes:
        print("  This makes ONE REAL API CALL against your Groq quota.")
        if input("  Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("  Aborted. No API call made.")
            return 0
        print()

    print(f"  message: {TEST_MESSAGE}")
    print(f"  calling provider ({args.model})...")
    print()

    try:
        with capture_audit_records() as records:
            result = call_llm(
                TEST_MESSAGE,
                schema=SymptomExtraction,
                tier="fast",
                prompt_name="verify_groq_enum",
            )
    except LLMError as exc:
        print_raw_response(records)
        print("  RESULT: provider call FAILED")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  If this is a schema-related 400, this model's strict support may")
        print("  have changed since this script's model list was written — recheck")
        print("  https://console.groq.com/docs/structured-outputs before assuming")
        print("  the key, quota, or vocabulary size is at fault.")
        return 1

    assert isinstance(result, SymptomExtraction)

    print_raw_response(records)

    print("  1. Provider accepted the strict json_schema request : YES")
    print(f"  2. Response parsed and validated                    : YES "
          f"({len(result.symptoms)} symptom(s), "
          f"{len(result.negated_symptoms)} negated)")
    print()

    returned_names = [s.name for s in result.symptoms] + [
        n.name for n in result.negated_symptoms
    ]

    if not returned_names:
        print("  3. Byte-for-byte match: NO NAMES RETURNED — inconclusive.")
        if absent_concepts:
            print(f"     Every concept in the test message is missing from the")
            print(f"     vocabulary ({', '.join(absent_concepts)}), so there was")
            print("     nothing the model was permitted to return. Reconcile the")
            print("     vocabulary before reading anything into this.")
        else:
            print("     The vocabulary covers this message, so the extraction")
            print("     prompt is the thing to look at.")
        return 1

    print("  3. Byte-for-byte vocabulary match:")
    all_exact = True
    for name in returned_names:
        exact = name in CANONICAL_SYMPTOMS
        all_exact &= exact
        print(f"     [{'OK ' if exact else 'BAD'}] {name!r}")
        if not exact:
            describe_mismatch(name)

    print()
    non_latin = [n for n in returned_names if any(ord(ch) > 0x7F for ch in n)]
    print(f"  4. Non-Latin values returned intact  : "
          f"{'YES' if non_latin and all_exact else 'SEE ABOVE'} "
          f"({len(non_latin)}/{len(returned_names)} contain non-ASCII)")

    print()
    if absent_concepts:
        print("  5. Completeness of this extraction   : NOT ASSESSABLE")
        print(f"     {len(absent_concepts)} concept(s) in the test message are absent")
        print(f"     from the vocabulary ({', '.join(absent_concepts)}).")
        print("     Anything missing from the response is explained by that gap.")
        print("     Do not tune the extraction prompt on this run.")
    else:
        expected_total = len(EXPECTED_CONCEPTS)
        print(f"  5. Completeness of this extraction   : "
              f"{len(returned_names)}/{expected_total} expected concept(s) returned")
        if len(returned_names) < expected_total:
            print("     The vocabulary covers every concept in the test message, so a")
            print("     shortfall here IS an extraction-prompt problem.")

    print()
    print("=" * 72)
    if all_exact:
        print(f"VERDICT: enum survives the round trip on {args.model}.")
        print("Exact-string matching is safe for THIS model. Re-run with")
        print("--model for the other one before relying on both.")
    else:
        print(f"VERDICT: MISMATCH on {args.model} — a returned value is not")
        print("byte-identical to the vocabulary. rules/red_flags.py matching")
        print("would silently miss it.")
    print("=" * 72)

    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
