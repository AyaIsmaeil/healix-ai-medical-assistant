"""MANUAL VERIFICATION SCRIPT — MAKES ONE REAL, BILLABLE GEMINI API CALL.

*** This is NOT part of the pytest suite and must never be added to it. ***
Run it by hand, deliberately:

    python scripts/verify_gemini_enum.py            # asks for confirmation
    python scripts/verify_gemini_enum.py --yes      # skips the prompt

What it answers, none of which a mocked test can:

  1. Does the provider ACCEPT a schema with an enum this large? Providers
     impose schema/enum size limits that are not documented per-field and
     only appear as the vocabulary grows toward its full 52 entries.
  2. Do non-Latin (Arabic) enum values survive the round trip, or does
     something along the way mangle, transliterate, or escape them?
  3. Does the returned value match a vocabulary entry BYTE FOR BYTE?
     A visually identical string that differs by one codepoint — a hamza
     form, a normalization form — parses as a valid enum member only if it
     is byte-identical, and would otherwise fail validation. Downstream,
     exact-string matching in rules/red_flags.py depends on this.

Requires GEMINI_API_KEY, HEALIX_LLM_PROVIDER_FAST=gemini, and
HEALIX_MODEL_FAST in the environment (.env).

See scripts/verify_groq_enum.py for the same check against Groq — a
separate script, not a flag on this one, because the two providers'
structured-output support differs enough (Groq's is model-gated) to want
their own framing rather than a shared --provider switch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from llm_client import LLMError, call_llm  # noqa: E402
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

load_dotenv()


def main() -> int:
    schema_json = json.dumps(SymptomExtraction.model_json_schema(), ensure_ascii=False)
    enum_values = SymptomExtraction.model_json_schema()["$defs"]["ExtractedSymptom"][
        "properties"
    ]["name"]["enum"]

    print("=" * 72)
    print("Gemini structured-output enum verification")
    print("=" * 72)
    print(f"  vocabulary entries : {len(CANONICAL_SYMPTOMS)}")
    print(f"  enum length        : {len(enum_values)}")
    print(f"  schema size        : {len(schema_json.encode('utf-8'))} bytes "
          f"({len(schema_json)} chars)")
    print()
    print("  Watch these two numbers as the vocabulary grows toward 52 —")
    print("  provider limits surface as a rejected request, not a warning.")
    print()

    absent_concepts = report_vocabulary_coverage()

    if "--yes" not in sys.argv and "-y" not in sys.argv:
        print("  This makes ONE REAL API CALL against your quota.")
        if input("  Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("  Aborted. No API call made.")
            return 0
        print()

    print(f"  message: {TEST_MESSAGE}")
    print("  calling provider...")
    print()

    try:
        with capture_audit_records() as records:
            result = call_llm(
                TEST_MESSAGE,
                schema=SymptomExtraction,
                tier="fast",
                prompt_name="verify_gemini_enum",
            )
    except LLMError as exc:
        print_raw_response(records)
        print("  RESULT: provider call FAILED")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  If this is a schema-size or invalid-argument error, the enum")
        print("  may have exceeded a provider limit. Re-run with a trimmed")
        print("  vocabulary to confirm before assuming the key or model is at fault.")
        return 1

    assert isinstance(result, SymptomExtraction)

    print_raw_response(records)

    print("  1. Provider accepted the enum schema : YES")
    print(f"  2. Response parsed and validated     : YES "
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
        print("VERDICT: enum survives the round trip. Exact-string matching is safe.")
    else:
        print("VERDICT: MISMATCH — a returned value is not byte-identical to the")
        print("vocabulary. rules/red_flags.py matching would silently miss it.")
    print("=" * 72)

    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
