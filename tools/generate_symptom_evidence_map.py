"""
Healix — Symptom Evidence Map Generator (offline, reproducible).

يُولِّد ``app/dictionaries/symptom_evidence_map.yaml`` من مصادر حقيقية فقط:
  1. ``app/data/raw/ddxplus/release_evidences.json`` — نصّ السؤال الإنجليزي الرسمي
  2. ``app/dictionaries/symptom_ontology.json`` — مرادفات عربية (بعد seed_symptom_ar)

لا يُختلَع أي evidence_source ولا أي رمز E_* — كل شيء قابل للتحقّق من DDXPlus.
أسماء المفاهيم (concept) تُحفَظ من الملف الحالي عند تطابق ``evidence_codes``،
وإلا تُشتَقّ من ``HEALIX_SYMPTOM_####`` (مثلاً ``symptom_0001``).

Run:
    python tools/generate_symptom_evidence_map.py
    python tools/generate_symptom_evidence_map.py --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_DDXPLUS_DIR = _ROOT / "app" / "data" / "raw" / "ddxplus"
_ONTOLOGY_PATH = _ROOT / "app" / "dictionaries" / "symptom_ontology.json"
_OUT_PATH = _ROOT / "app" / "dictionaries" / "symptom_evidence_map.yaml"


def _load_evidences() -> Dict[str, Any]:
    path = _DDXPLUS_DIR / "release_evidences.json"
    if not path.exists():
        raise SystemExit(
            f"[error] {path} missing — run: python tools/download_ddxplus.py --quick"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_source(evidences: Dict[str, Any], code: str) -> str:
    entry = evidences.get(code) or {}
    question = str(entry.get("question_en", "")).strip()
    return f"{code}: {question}" if question else code


def _preserve_concepts(existing: Dict[str, Any]) -> Dict[Tuple[str, ...], str]:
    """evidence_codes tuple -> concept name from current yaml."""
    out: Dict[Tuple[str, ...], str] = {}
    for entry in existing.get("mappings", []):
        codes = tuple(entry.get("evidence_codes") or ())
        concept = entry.get("concept")
        if codes and concept:
            out[codes] = concept
    return out


def _derive_concept(healix_id: str, codes: List[str], preserved: Dict[Tuple[str, ...], str]) -> str:
    key = tuple(codes)
    if key in preserved:
        return preserved[key]
    # fallback: symptom slug from healix id
    suffix = healix_id.replace("HEALIX_SYMPTOM_", "").lstrip("0") or "0"
    return f"symptom_{suffix}"


def build_map(
    ontology: Dict[str, Any],
    evidences: Dict[str, Any],
    preserved: Dict[Tuple[str, ...], str],
) -> Dict[str, Any]:
    mappings: List[Dict[str, Any]] = []

    for healix_id, info in sorted(ontology.get("symptoms", {}).items()):
        code = info.get("ddxplus_code")
        arabic = [t.strip() for t in (info.get("ar") or []) if t and str(t).strip()]
        if not code or not arabic:
            continue

        codes = [code]
        concept = _derive_concept(healix_id, codes, preserved)
        mappings.append({
            "concept": concept,
            "names": arabic,
            "evidence_codes": codes,
            "evidence_source": _evidence_source(evidences, code),
            "_healix_id": healix_id,
            "_en": info.get("en", ""),
        })

    # preserve legacy multi-code entries (e.g. headache/chest_pain/abdominal_pain -> E_53+E_55)
    legacy_by_concept = {
        entry["concept"]: entry
        for entry in (ontology.get("_legacy") or [])  # unused; read from existing yaml instead
    }
    _ = legacy_by_concept  # placeholder for future extension

    return {
        "version": "1.1.0",
        "source": {
            "ddxplus": "app/data/raw/ddxplus/release_evidences.json",
            "ontology": "app/dictionaries/symptom_ontology.json",
            "generator": "tools/generate_symptom_evidence_map.py",
        },
        "mappings": [
            {k: v for k, v in entry.items() if not k.startswith("_")}
            for entry in mappings
        ],
    }


def _merge_legacy_pain_entries(
    result: Dict[str, Any],
    existing: Dict[str, Any],
    evidences: Dict[str, Any],
) -> None:
    """يُبقي مدخلات الألم متعددة الرموز (E_53+E_55) إن وُجدت بالملف الحالي."""
    current_concepts = {m["concept"] for m in result["mappings"]}
    for entry in existing.get("mappings", []):
        codes = entry.get("evidence_codes") or []
        if len(codes) <= 1:
            continue
        concept = entry.get("concept")
        if concept in current_concepts:
            continue
        names = entry.get("names") or []
        if not names:
            continue
        sources = " / ".join(_evidence_source(evidences, c) for c in codes)
        result["mappings"].append({
            "concept": concept,
            "names": names,
            "evidence_codes": codes,
            "evidence_source": sources,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing yaml")
    args = parser.parse_args()

    if _OUT_PATH.exists() and not args.force:
        print(f"[skip] {_OUT_PATH} exists — use --force to regenerate")
        return

    if not _ONTOLOGY_PATH.exists():
        raise SystemExit(
            f"[error] {_ONTOLOGY_PATH} missing — run: python tools/generate_symptom_ontology.py"
        )

    existing: Dict[str, Any] = {}
    if _OUT_PATH.exists():
        existing = yaml.safe_load(_OUT_PATH.read_text(encoding="utf-8")) or {}

    ontology = json.loads(_ONTOLOGY_PATH.read_text(encoding="utf-8"))
    evidences = _load_evidences()
    preserved = _preserve_concepts(existing)

    result = build_map(ontology, evidences, preserved)
    _merge_legacy_pain_entries(result, existing, evidences)

    # sort by concept for stable diffs
    result["mappings"].sort(key=lambda m: m["concept"])

    header = (
        "# Healix — قاموس ربط الأعراض العربية برموز أدلة DDXPlus\n"
        "# ⚠️ ملف مُولَّد آلياً — لا تُعدِّله يدوياً.\n"
        "# أعد توليده: python tools/generate_symptom_evidence_map.py --force\n"
        "# المصادر: release_evidences.json + symptom_ontology.json\n\n"
    )
    _OUT_PATH.write_text(
        header + yaml.dump(result, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    print(f"[ok] wrote {_OUT_PATH} ({len(result['mappings'])} mappings)")


if __name__ == "__main__":
    main()
