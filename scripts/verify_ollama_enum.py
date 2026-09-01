from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

_DEFAULT_MODEL = "qwen3:4b"
os.environ["HEALIX_LLM_PROVIDER_FAST"] = "ollama"
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("HEALIX_MODEL_FAST", _DEFAULT_MODEL)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Any locally-pulled Ollama model (default: {_DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    os.environ["HEALIX_MODEL_FAST"] = args.model
    base_url = os.environ["OLLAMA_BASE_URL"]

    schema_json = json.dumps(SymptomExtraction.model_json_schema(), ensure_ascii=False)
    enum_values = SymptomExtraction.model_json_schema()["$defs"]["ExtractedSymptom"][
        "properties"
    ]["name"]["enum"]

    print("=" * 72)
    print("Ollama structured-output enum verification (local, free)")
    print("=" * 72)
    print(f"  server             : {base_url}")
    print(f"  model              : {args.model}")
    print(f"  vocabulary entries : {len(CANONICAL_SYMPTOMS)}")
    print(f"  enum length        : {len(enum_values)}")
    print(f"  schema size        : {len(schema_json.encode('utf-8'))} bytes "
          f"({len(schema_json)} chars)")
    print()
    print("  Structured output here is enforced by llama.cpp's grammar system")
    print("  beneath the model, not by the model itself — unlike Groq, this is")
    print("  NOT limited to specific models. A PASS is still specific to this")
    print("  server/model combination, e.g. a corrupt local model file or a")
    print("  very old Ollama version could still misbehave.")
    print()

    absent_concepts = report_vocabulary_coverage()

    print(f"  message: {TEST_MESSAGE}")
    print(f"  calling {args.model} on {base_url} (no confirmation needed — free, local)...")
    print()

    started = time.monotonic()
    try:
        with capture_audit_records() as records:
            result = call_llm(
                TEST_MESSAGE,
                schema=SymptomExtraction,
                tier="fast",
                prompt_name="verify_ollama_enum",
            )
    except LLMError as exc:
        elapsed = time.monotonic() - started
        print_raw_response(records)
        print(f"  RESULT: provider call FAILED after {elapsed:.1f}s")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("  If this is a connection error, confirm Ollama is running")
        print("  (`ollama serve`) and the model is pulled (`ollama pull "
              f"{args.model}`). If it's a timeout, this may be exactly the")
        print("  tight-default problem documented in .env.example — try again")
        print("  after raising HEALIX_LLM_TIMEOUT_SECONDS / "
              "HEALIX_LLM_TOTAL_BUDGET_SECONDS.")
        return 1

    elapsed = time.monotonic() - started
    assert isinstance(result, SymptomExtraction)

    print_raw_response(records)

    print(f"  Round-trip time                      : {elapsed:.2f}s")
    if records:
        print(f"    attempts: {records[-1].get('attempts')}   "
              f"usage: {records[-1].get('usage')}")
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
        print(f"VERDICT: enum survives the round trip on {args.model} ({elapsed:.1f}s).")
        print("Exact-string matching is safe for this model/server.")
    else:
        print(f"VERDICT: MISMATCH on {args.model} — a returned value is not")
        print("byte-identical to the vocabulary. rules/red_flags.py matching")
        print("would silently miss it.")
    print("=" * 72)

    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
