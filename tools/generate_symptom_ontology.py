"""
Healix - Symptom Ontology Template Generator (offline tool).

Builds the GROUND-TRUTH template that the runtime symptom-coding layer consumes:
maps every ``HEALIX_SYMPTOM_####`` id (as frozen by ``OntologyMapper``) to its
DDXPlus evidence code and the English / French labels from the DDXPlus release
dictionaries. Nothing is invented — labels are copied verbatim from
``release_evidences.json``; the Arabic synonym list (``ar``) is emitted EMPTY on
purpose, to be filled by clinical review, never machine-guessed.

Run:
    venv/Scripts/python.exe tools/generate_symptom_ontology.py

Output:
    app/dictionaries/symptom_ontology.json   (only creates it if absent; use
    --force to overwrite, which PRESERVES any Arabic terms already filled in.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.ontology_mapper import OntologyMapper

_ROOT = Path(__file__).resolve().parent.parent
_DDXPLUS_DIR = _ROOT / "app" / "data" / "raw" / "ddxplus"
_OUT_PATH = _ROOT / "app" / "dictionaries" / "symptom_ontology.json"


def _label(evidences: dict, code: str, key: str) -> str:
    """Verbatim label from the DDXPlus release dictionary ('' if missing)."""
    return str((evidences.get(code) or {}).get(key, "")).strip()


def build_template(preserve_from: dict | None = None) -> dict:
    mapper = OntologyMapper.from_directory(_DDXPLUS_DIR)
    evidences = mapper._evidences  # read-only access to the release dictionary

    preserved = (preserve_from or {}).get("symptoms", {})

    symptoms: dict[str, dict] = {}
    # id -> code (invert the frozen code -> id table); HEALIX-only ids have no code
    code_by_id = {sid: code for code, sid in mapper.symptom_id_by_code.items()}

    for healix_id in mapper.symptom_ids:
        code = code_by_id.get(healix_id)
        existing_ar = preserved.get(healix_id, {}).get("ar", []) if preserved else []
        symptoms[healix_id] = {
            "ddxplus_code": code,  # None for HEALIX-only symptoms
            "en": _label(evidences, code, "question_en") if code else "",
            "fr": _label(evidences, code, "question_fr") if code else "",
            "ar": list(existing_ar),  # keep any clinically-reviewed terms
        }

    return {
        "ontology_version": "healix-ontology-v1.0.0",
        "description": (
            "Runtime symptom-coding lexicon. Maps HEALIX_SYMPTOM ids to Arabic "
            "synonyms for free-text interview matching. 'en'/'fr' are DDXPlus "
            "reference labels (do not edit); fill 'ar' under clinical review."
        ),
        "symptom_count": len(symptoms),
        "symptoms": symptoms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing file (Arabic terms already present are preserved)",
    )
    args = parser.parse_args()

    preserve = None
    if _OUT_PATH.exists():
        preserve = json.loads(_OUT_PATH.read_text(encoding="utf-8"))
        if not args.force:
            filled = sum(1 for s in preserve.get("symptoms", {}).values() if s.get("ar"))
            print(f"[skip] {_OUT_PATH} already exists "
                  f"({len(preserve.get('symptoms', {}))} symptoms, {filled} with Arabic). "
                  f"Use --force to regenerate (Arabic terms are preserved).")
            return

    template = build_template(preserve_from=preserve)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    filled = sum(1 for s in template["symptoms"].values() if s["ar"])
    print(f"[ok] wrote {_OUT_PATH}")
    print(f"     {template['symptom_count']} symptoms | {filled} with Arabic terms filled")


if __name__ == "__main__":
    main()
