"""
Healix - Feature Schema v2 generator (Phase 5) — OFFLINE, run once.

Emits ``app/dictionaries/feature_schemas/v2.json`` exactly as frozen in
``docs/research/FEATURE_SCHEMA_V2_SPECIFICATION.md`` (286 columns,
healix-ontology-v1.0.0).

Keeping the generator in the repository makes the frozen schema reproducible:
the file is data, but the derivation of that data is auditable.

Run:  python -m app.ml.schema_v2_generator
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ml.ontology_mapper import (ANTHROPOMETRIC, EXPOSURE_BOOLEAN,
                                    LIFESTYLE_BOOLEAN, REPRODUCTIVE,
                                    OntologyMapper)

REPO = Path(__file__).resolve().parents[2]
DDXPLUS_DIR = REPO / "app" / "data" / "raw" / "ddxplus"
OUT_PATH = REPO / "app" / "dictionaries" / "feature_schemas" / "v2.json"

SCHEMA_VERSION = "assessment-features-v2"
ONTOLOGY_VERSION = "healix-ontology-v1.0.0"
EXPECTED_COLUMNS = 286

LOCATIONS = ["head", "face", "mouth_throat", "neck", "chest", "abdomen", "back",
             "pelvis_groin", "genital", "anorectal", "upper_limb", "lower_limb", "none"]
QUALITIES = ["sharp", "burning", "pressure", "pulsating", "cramping", "tender",
             "affective_distress"]
TRAVEL_REGIONS = ["none", "north_africa", "west_africa", "south_africa",
                  "central_america", "north_america", "south_america", "asia",
                  "south_east_asia", "caribbean", "europe", "oceania"]

V1_COLUMNS = ["age", "gender_male", "gender_female", "severity", "smoking",
              "temperature_c", "progression_worsening", "progression_stable",
              "progression_improving"]


def build_schema(mapper: OntologyMapper) -> Dict[str, Any]:
    numeric: "OrderedDict[str, Any]" = OrderedDict()
    boolean: "OrderedDict[str, Any]" = OrderedDict()
    categorical: "OrderedDict[str, Any]" = OrderedDict()
    order: List[str] = []

    def num(name: str, source: str, ref: Optional[str] = None) -> None:
        numeric[name] = {"source": source, "nullable": True,
                         **({"ontology_ref": ref} if ref else {})}
        order.append(name)

    def boo(name: str, source: str, ref: Optional[str] = None) -> None:
        boolean[name] = {"source": source, "nullable": True,
                         **({"ontology_ref": ref} if ref else {})}
        order.append(name)

    def cat(field: str, prefix: str, values: List[str], source: str,
            ref: Optional[str] = None) -> None:
        categorical[field] = {"source": source, "values": values,
                              "one_hot_prefix": prefix, "nullable": True,
                              **({"ontology_ref": ref} if ref else {})}
        order.extend(f"{prefix}_{v}" for v in values)

    # G1 Demographics (4)
    num("age", "demographics.age")
    cat("gender", "gender", ["male", "female"], "demographics.gender")
    boo("pregnancy_possible", "demographics.pregnancy_possible", "E_167")
    # G2 Vitals (1)
    num("temperature_c", "temperature_c")
    # G3 Symptoms (98)
    for sid in mapper.symptom_ids:
        code = next((c for c, s in mapper.symptom_id_by_code.items() if s == sid), None)
        boo(f"sym_{sid}", f"symptom_presence.{sid}", code or "HEALIX_ONLY")
    # G4 Descriptors (53)
    num("severity", "primary_symptom.descriptors.severity_0_10", "E_56")
    num("onset_days_ago", "primary_symptom.descriptors.onset_days_ago")
    num("onset_speed", "primary_symptom.descriptors.onset_speed", "E_59")
    num("pain_precision", "primary_symptom.descriptors.pain_precision", "E_58")
    cat("progression", "progression", ["worsening", "stable", "improving"],
        "primary_symptom.descriptors.progression")
    cat("quality", "quality", QUALITIES, "primary_symptom.descriptors.quality", "E_54")
    cat("location", "location", LOCATIONS, "primary_symptom.descriptors.location", "E_55")
    cat("radiation", "radiation", LOCATIONS, "primary_symptom.descriptors.radiation", "E_57")
    cat("laterality", "laterality", ["left", "right", "bilateral", "midline"],
        "primary_symptom.descriptors.laterality")
    cat("rash_color", "rash_color", ["dark", "yellow", "pale", "pink", "red"],
        "primary_symptom.descriptors.rash_color", "E_130")
    num("rash_swollen", "primary_symptom.descriptors.rash_swollen", "E_132")
    num("rash_itch", "primary_symptom.descriptors.rash_itch", "E_136")
    boo("lesion_gt_1cm", "primary_symptom.descriptors.lesion_gt_1cm", "E_135")
    boo("lesion_peels", "primary_symptom.descriptors.lesion_peels", "E_131")
    # G5..G9 History (80) — source points INTO the collection, unique per column.
    for code, feature in mapper.history_feature_by_code.items():
        boo(feature, f"medical_history.{mapper.history_path_by_code[code]}", code)
    # G10 Lifestyle (20)
    boo("smoking", "lifestyle.smoking", "E_79")          # v1 legacy column preserved
    cat("smoking_status", "smoking_status", ["never", "current", "former", "passive"],
        "lifestyle.smoking_status", "E_79|E_191|E_222")
    boo("alcohol", "lifestyle.alcohol", LIFESTYLE_BOOLEAN["alcohol"])
    cat("occupation", "occupation",
        ["agriculture", "construction", "mining", "daycare", "other"],
        "lifestyle.occupation", "E_198|E_199|E_200|E_49")
    for name in ("substance_iv", "substance_stimulant", "caffeine", "energy_drinks",
                 "physical_activity", "household_size_ge4"):
        boo(name, f"lifestyle.{name}", LIFESTYLE_BOOLEAN[name])
    cat("residence", "residence", ["rural", "suburb", "city"], "lifestyle.residence",
        "E_183|E_195|E_207")
    # G11 Exposure (21)
    cat("travel_region", "travel", TRAVEL_REGIONS, "exposure.travel_region", "E_204")
    for name, code in EXPOSURE_BOOLEAN.items():
        boo(name, f"exposure.{name}", code)
    # G12 / G13
    for name, code in ANTHROPOMETRIC.items():
        boo(name, f"anthropometric.{name}", code)
    for name, code in REPRODUCTIVE.items():
        boo(name, f"reproductive.{name}", code)
    # G14 Derived (5)
    num("symptom_count", "derived.symptom_count")
    num("positive_negative_ratio", "derived.positive_negative_ratio")
    boo("has_red_flag", "derived.has_red_flag")
    num("validity_score", "validation.validity_score")
    num("interview_completeness_ratio", "derived.interview_completeness_ratio")

    return OrderedDict([
        ("schema_version", SCHEMA_VERSION),
        ("ontology_version", ONTOLOGY_VERSION),
        ("supersedes", "assessment-features-v1"),
        ("description",
         "Feature Schema v2 - 286 columns, frozen in Phase 4.7 "
         "(docs/research/FEATURE_SCHEMA_V2_SPECIFICATION.md). Strict additive "
         "superset of v1: all 9 v1 columns preserved verbatim."),
        ("feature_order", order),
        ("numeric_fields", numeric),
        ("boolean_fields", boolean),
        ("categorical_fields", categorical),
    ])


def validate(schema: Dict[str, Any]) -> None:
    """Structural + semantic guards. A failure here must abort generation."""
    order = schema["feature_order"]
    produced = list(schema["numeric_fields"]) + list(schema["boolean_fields"])
    for spec in schema["categorical_fields"].values():
        produced += [f"{spec['one_hot_prefix']}_{v}" for v in spec["values"]]

    assert len(order) == len(set(order)), "duplicate column names"
    assert set(produced) == set(order), f"column mismatch: {set(produced) ^ set(order)}"
    assert len(order) == EXPECTED_COLUMNS, f"expected {EXPECTED_COLUMNS}, got {len(order)}"

    missing_v1 = [c for c in V1_COLUMNS if c not in order]
    assert not missing_v1, f"v1 columns lost: {missing_v1}"

    # Guard against the Phase-5 defect class: a shared `source` path silently
    # produced 80 always-null history columns. Every scalar column must resolve
    # from a DISTINCT path, or the encoder cannot tell the columns apart.
    sources = [spec["source"] for spec in schema["numeric_fields"].values()]
    sources += [spec["source"] for spec in schema["boolean_fields"].values()]
    duplicates = {s for s in sources if sources.count(s) > 1}
    assert not duplicates, f"scalar columns share a source path: {sorted(duplicates)}"


def main() -> None:
    mapper = OntologyMapper.from_directory(DDXPLUS_DIR)
    schema = build_schema(mapper)
    validate(schema)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO)} | columns={len(schema['feature_order'])} "
          f"numeric={len(schema['numeric_fields'])} "
          f"boolean={len(schema['boolean_fields'])} "
          f"categorical={len(schema['categorical_fields'])}")


if __name__ == "__main__":
    main()
