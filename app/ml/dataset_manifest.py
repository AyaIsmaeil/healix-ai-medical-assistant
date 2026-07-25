"""
Healix - Dataset Manifest (Phase 5).

Builds the reproducibility record for a dataset build:
``dataset_version`` is a deterministic function of
*(source file hashes + ontology version + schema version + builder version + seed)*.
Re-running with the same inputs must reproduce byte-identical artifacts, and any
change to any input yields a new immutable ``dataset_version``.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

BUILDER_VERSION = "dataset-builder-v1.0.0"


def file_digest(path: Path, chunk: int = 8 << 20) -> Dict[str, Any]:
    """Content hash + size of a source file (streamed; never loads it whole)."""
    h = hashlib.blake2b(digest_size=16)
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            size += len(block)
            h.update(block)
    return {"file": path.name, "sha": h.hexdigest(), "size_bytes": size}


def compute_dataset_version(source_digests: List[Dict[str, Any]],
                            ontology_version: str, schema_version: str,
                            seed: int) -> str:
    payload = "|".join(
        [d["sha"] for d in sorted(source_digests, key=lambda x: x["file"])]
        + [ontology_version, schema_version, BUILDER_VERSION, str(seed)]
    )
    short = hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()
    return f"ddxplus-v2.0-{short}"


def build_manifest(*, dataset_version: str, ontology_version: str,
                   schema_version: str, seed: int, chunk_size: int,
                   source_digests: List[Dict[str, Any]],
                   splits: Dict[str, Any], artifacts: List[Dict[str, Any]],
                   timings: Dict[str, float]) -> Dict[str, Any]:
    return {
        "dataset_version": dataset_version,
        "builder_version": BUILDER_VERSION,
        "ontology_version": ontology_version,
        "feature_schema_version": schema_version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "chunk_size": chunk_size,
        "determinism": {
            "split_policy": "official DDXPlus split preserved; no reshuffling",
            "row_order": "source order preserved",
            "reproducible": True,
        },
        # The single most important interoperability field: DDXPlus is a
        # complete simulated interview, so an unlisted evidence is a true
        # negative (0). Healix runtime is open-world (unasked -> null).
        "absence_semantics": "closed_world",
        "source": {"dataset": "DDXPlus", "files": source_digests},
        "splits": splits,
        "artifacts": artifacts,
        "timings_seconds": timings,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "guarantees": [
            "No rows were removed, altered, or reshuffled.",
            "Leaked rows are flagged (leakage_flag), never dropped.",
            "No imputation, scaling, or class rebalancing was applied.",
            "Nulls are preserved as nulls (never coerced to 0).",
        ],
    }


def feature_dictionary(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-readable description of every feature column."""
    groups: Dict[str, str] = {}
    entries: Dict[str, Any] = {}

    for name, spec in schema.get("numeric_fields", {}).items():
        entries[name] = {"kind": "numeric", "encoding": "raw_value",
                         "source": spec.get("source"), "nullable": True,
                         "ontology_ref": spec.get("ontology_ref")}
    for name, spec in schema.get("boolean_fields", {}).items():
        entries[name] = {"kind": "boolean", "encoding": "tri_state_1_0_null",
                         "source": spec.get("source"), "nullable": True,
                         "ontology_ref": spec.get("ontology_ref")}
    for field, spec in schema.get("categorical_fields", {}).items():
        for value in spec["values"]:
            col = f"{spec['one_hot_prefix']}_{value}"
            entries[col] = {
                "kind": "categorical_one_hot", "encoding": "one_hot_null_propagating",
                "parent_field": field, "category": value,
                "source": spec.get("source"), "nullable": True,
                "ontology_ref": spec.get("ontology_ref"),
            }

    for name in schema["feature_order"]:
        if name.startswith("sym_"):
            groups[name] = "G3_symptoms"
        elif name.startswith("hx_chronic_"):
            groups[name] = "G5_chronic_diseases"
        elif name.startswith("hx_med_"):
            groups[name] = "G6_medications"
        elif name.startswith("hx_allergy_"):
            groups[name] = "G7_allergies"
        elif name.startswith("hx_family_"):
            groups[name] = "G8_family_history"
        elif name.startswith("hx_surgery_"):
            groups[name] = "G9_surgeries"
        elif name.startswith(("travel_", "contact_", "vaccination_", "recent_")):
            groups[name] = "G11_exposure"
        elif name.startswith(("bmi_",)):
            groups[name] = "G12_anthropometric"
        elif name.startswith(("repro_",)):
            groups[name] = "G13_reproductive"
        elif name in ("age", "gender_male", "gender_female", "pregnancy_possible"):
            groups[name] = "G1_demographics"
        elif name == "temperature_c":
            groups[name] = "G2_vitals"
        elif name in ("symptom_count", "positive_negative_ratio", "has_red_flag",
                      "validity_score", "interview_completeness_ratio"):
            groups[name] = "G14_derived"
        elif name.startswith(("smoking", "alcohol", "occupation_", "substance_",
                              "caffeine", "energy_drinks", "physical_activity",
                              "residence_", "household_")):
            groups[name] = "G10_lifestyle"
        else:
            groups[name] = "G4_descriptors"

    for name, group in groups.items():
        entries.setdefault(name, {})["group"] = group

    return {
        "feature_schema_version": schema["schema_version"],
        "ontology_version": schema.get("ontology_version"),
        "n_features": len(schema["feature_order"]),
        "feature_order": schema["feature_order"],
        "features": entries,
        "encoding_rules": {
            "tri_state_1_0_null": "1 = present, 0 = explicitly absent, null = unknown",
            "one_hot_null_propagating": "unknown/unmapped category => ALL members null "
                                        "(never zeroed, which would fabricate a negative)",
            "raw_value": "numeric passthrough; no scaling or imputation in the schema",
        },
    }
