"""
Healix - Ontology Mapper (Phase 5, Stage 1).

Builds the deterministic lookup tables that translate DDXPlus identifiers into
canonical Healix ontology identifiers, exactly as frozen in
``docs/research/FEATURE_SCHEMA_V2_SPECIFICATION.md`` (healix-ontology-v1.0.0).

Nothing here is invented: every mapping is derived from the DDXPlus release
dictionaries plus the curated groupings frozen in Phase 4.7. Any DDXPlus code
that fails to resolve raises immediately — silent skipping is forbidden,
because a quietly wrong ontology is the exact failure this project exists to
prevent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class OntologyError(RuntimeError):
    """Raised when a DDXPlus identifier cannot be resolved to the ontology."""


# --- Descriptor evidences: handled by the descriptor block, not as symptoms ---
DESCRIPTOR_CODES = frozenset({
    "E_54", "E_55", "E_56", "E_57", "E_58", "E_59", "E_133", "E_152",
    "E_130", "E_131", "E_132", "E_134", "E_135", "E_136",
})

# --- Curated antecedent buckets (frozen Phase 4.3 / 4.4 / 4.7) ---
MEDICATIONS = ["E_15", "E_44", "E_71", "E_100", "E_104", "E_124", "E_146", "E_149",
               "E_184", "E_1", "E_10"]
ALLERGIES = ["E_12", "E_226"]
FAMILY = ["E_4", "E_25", "E_26", "E_28", "E_86", "E_87", "E_142", "E_99", "E_165",
          "E_223", "E_224", "E_225", "E_29"]
SURGERY = ["E_137"]
ETHNICITY = ["E_17"]          # excluded from the schema by design (privacy/fairness)
PREGNANCY = ["E_167"]
TRAVEL = ["E_204"]

LIFESTYLE_BOOLEAN = {           # feature_id -> DDXPlus code
    "alcohol": "E_78",
    "substance_iv": "E_61",
    "substance_stimulant": "E_62",
    "caffeine": "E_35",
    "energy_drinks": "E_60",
    "physical_activity": "E_143",
    "household_size_ge4": "E_48",
}
SMOKING_STATUS = {"current": "E_79", "former": "E_191", "passive": "E_222"}
OCCUPATION = {"agriculture": "E_198", "construction": "E_199", "mining": "E_200",
              "daycare": "E_49"}
RESIDENCE = {"rural": "E_183", "suburb": "E_195", "city": "E_207"}

EXPOSURE_BOOLEAN = {
    "contact_similar": "E_41", "contact_pertussis": "E_40", "contact_ebola": "E_73",
    "vaccination_up_to_date": "E_209", "recent_viral_infection": "E_0",
    "recent_cold": "E_116", "recent_surgery": "E_196",
    "recent_hospital_treatment": "E_147", "recent_stimulant_meds": "E_213",
}
ANTHROPOMETRIC = {"bmi_overweight": "E_70", "bmi_underweight": "E_208"}
REPRODUCTIVE = {"repro_breastfed_gt9mo": "E_11", "repro_menarche_lt12": "E_141"}

# --- Body-location curation: 165 DDXPlus codes -> 12 regions + none (Phase 4.7 §3) ---
_REGION_RULES = (
    ("none", (r"^nowhere$",)),
    ("head", (r"top of the head", r"back of head", r"forehead", r"occiput", r"temple")),
    ("face", (r"cheek", r"chin", r"^nose$", r"^jaw$", r"^eye", r"^ear", r"commissure",
              r"lip", r"vermilion")),
    ("mouth_throat", (r"^mouth$", r"tonsil", r"teeth", r"gum", r"uvula", r"pharynx",
                      r"palace", r"tongue", r"thyroid cartilage", r"trachea",
                      r"under the jaw")),
    ("neck", (r"neck", r"cervical spine", r"trapezius")),
    ("chest", (r"chest", r"breast", r"thoracic spine", r"scapula")),
    ("abdomen", (r"^belly$", r"epigastric", r"hypochondrium", r"flank", r"iliac fossa",
                 r"renal fossa")),
    ("back", (r"lumbar spine", r"coccyx", r"buttock")),
    ("pelvis_groin", (r"groin", r"^pubis$", r"iliac wing", r"iliac crest", r"^hip")),
    ("genital", (r"glans", r"^penis$", r"scrotum", r"testicle", r"labia", r"clitoris",
                 r"vagina", r"vulval", r"hymen", r"urethra")),
    ("anorectal", (r"^anus$",)),
    ("upper_limb", (r"shoulder", r"axilla", r"biceps", r"triceps", r"elbow", r"forearm",
                    r"wrist", r"hand", r"palm", r"finger", r"thumb")),
    ("lower_limb", (r"thigh", r"ischio", r"knee", r"popliteal", r"calf", r"tibia",
                    r"ankle", r"foot", r"sole", r"heel", r"toe")),
)

# --- Pain-quality curation: 15 usable DDXPlus values -> 7 categories (Phase 4.7 §4) ---
_QUALITY_MAP = {
    "a knife stroke": "sharp", "sharp": "sharp",
    "burning": "burning",
    "heavy": "pressure",
    "a pulse": "pulsating",
    "a cramp": "cramping", "tugging": "cramping",
    "sensitive": "tender",
    "heartbreaking": "affective_distress", "haunting": "affective_distress",
    "violent": "affective_distress", "sickening": "affective_distress",
    "scary": "affective_distress", "exhausting": "affective_distress",
    "tedious": "affective_distress",
}

_TRAVEL_MAP = {
    "N": "none", "North Africa": "north_africa", "West Africa": "west_africa",
    "South Africa": "south_africa", "Central America": "central_america",
    "North America": "north_america", "South America": "south_america",
    "Asia": "asia", "South East Asia": "south_east_asia", "Caraibes": "caribbean",
    "Europe": "europe", "Oceania": "oceania",
}

_RASH_COLOR_MAP = {"dark": "dark", "yellow": "yellow", "pale": "pale",
                   "pink": "pink", "red": "red"}

# "NA" is a placeholder value, never a category (Phase 4.7 §0 N3).
_NA_VALUES = frozenset({"NA", ""})

# Healix-only symptoms with no DDXPlus equivalent (Phase 4.7 §2.2).
HEALIX_ONLY_SYMPTOMS = ("HEALIX_SYMPTOM_0097", "HEALIX_SYMPTOM_0098")


def _label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("en", "")).strip()
    return str(value).strip()


class OntologyMapper:
    """Deterministic DDXPlus -> Healix ontology translation tables."""

    def __init__(self, evidences: Dict[str, Any], conditions: Dict[str, Any]):
        self._evidences = evidences
        self._conditions = conditions

        # --- symptom presence: E_* -> HEALIX_SYMPTOM_#### (declaration order) ---
        symptom_codes = [c for c, e in evidences.items()
                         if not e.get("is_antecedent") and c not in DESCRIPTOR_CODES]
        self.symptom_id_by_code: Dict[str, str] = {
            code: f"HEALIX_SYMPTOM_{i:04d}" for i, code in enumerate(symptom_codes, start=1)
        }
        self.symptom_ids: List[str] = (
            list(self.symptom_id_by_code.values()) + list(HEALIX_ONLY_SYMPTOMS)
        )

        # --- antecedent buckets: E_* -> feature_id ---
        assigned = set(MEDICATIONS + ALLERGIES + FAMILY + SURGERY + ETHNICITY
                       + PREGNANCY + TRAVEL
                       + list(LIFESTYLE_BOOLEAN.values())
                       + list(SMOKING_STATUS.values())
                       + list(OCCUPATION.values())
                       + list(RESIDENCE.values())
                       + list(EXPOSURE_BOOLEAN.values())
                       + list(ANTHROPOMETRIC.values())
                       + list(REPRODUCTIVE.values()))
        antecedents = [c for c, e in evidences.items() if e.get("is_antecedent")]
        self.chronic_codes = sorted(
            (c for c in antecedents if c not in assigned),
            key=lambda x: int(x.split("_")[1]),
        )

        # feature-column name, and the dotted path *into* the ClinicalFeatureSet
        # collection that holds it. The two must stay in lockstep: the schema
        # declares the path, the parser writes it, the encoder resolves it.
        self.history_feature_by_code: Dict[str, str] = {}
        self.history_path_by_code: Dict[str, str] = {}
        for bucket, codes in (("chronic_diseases", self.chronic_codes),
                              ("medications", MEDICATIONS),
                              ("allergies", ALLERGIES),
                              ("family_history", FAMILY),
                              ("surgeries", SURGERY)):
            prefix = {"chronic_diseases": "hx_chronic", "medications": "hx_med",
                      "allergies": "hx_allergy", "family_history": "hx_family",
                      "surgeries": "hx_surgery"}[bucket]
            for code in codes:
                self.history_feature_by_code[code] = f"{prefix}_{code}"
                self.history_path_by_code[code] = f"{bucket}.{code}"

        self.lifestyle_feature_by_code = {v: k for k, v in LIFESTYLE_BOOLEAN.items()}
        self.exposure_feature_by_code = {v: k for k, v in EXPOSURE_BOOLEAN.items()}
        self.anthro_feature_by_code = {v: k for k, v in ANTHROPOMETRIC.items()}
        self.repro_feature_by_code = {v: k for k, v in REPRODUCTIVE.items()}
        self.smoking_status_by_code = {v: k for k, v in SMOKING_STATUS.items()}
        self.occupation_by_code = {v: k for k, v in OCCUPATION.items()}
        self.residence_by_code = {v: k for k, v in RESIDENCE.items()}

        # --- value vocabularies (V_* -> curated category) ---
        self.location_by_value = self._build_location_map()
        self.laterality_by_value = self._build_laterality_map()
        self.quality_by_value = self._build_value_map("E_54", _QUALITY_MAP)
        self.travel_by_value = self._build_value_map("E_204", _TRAVEL_MAP)
        self.rash_color_by_value = self._build_value_map("E_130", _RASH_COLOR_MAP)

        # --- diseases: pathology name -> HEALIX_DISEASE_#### (+ metadata) ---
        self.disease_by_pathology: Dict[str, Dict[str, Any]] = {}
        for i, (name, cond) in enumerate(sorted(conditions.items()), start=1):
            self.disease_by_pathology[name] = {
                "healix_id": f"HEALIX_DISEASE_{i:04d}",
                "icd10": cond.get("icd10-id"),
                "severity": cond.get("severity"),
                "urgency_prior": self.severity_to_urgency(cond.get("severity")),
            }

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_directory(cls, ddxplus_dir: Path) -> "OntologyMapper":
        ddxplus_dir = Path(ddxplus_dir)
        evidences = json.loads(
            (ddxplus_dir / "release_evidences.json").read_text(encoding="utf-8"))
        conditions = json.loads(
            (ddxplus_dir / "release_conditions.json").read_text(encoding="utf-8"))
        return cls(evidences, conditions)

    def _build_location_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for value, meaning in self._evidences["E_55"]["value_meaning"].items():
            label = _label(meaning).lower()
            region = next(
                (name for name, patterns in _REGION_RULES
                 if any(re.search(p, label) for p in patterns)),
                None,
            )
            if region is None:
                raise OntologyError(
                    f"Location value {value} ({label!r}) matches no frozen region.")
            mapping[value] = region
        return mapping

    def _build_laterality_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for value, meaning in self._evidences["E_55"]["value_meaning"].items():
            label = _label(meaning)
            if label.endswith("(R)"):
                mapping[value] = "right"
            elif label.endswith("(L)"):
                mapping[value] = "left"
            else:
                mapping[value] = "midline"
        return mapping

    def _build_value_map(self, code: str, curated: Dict[str, str]) -> Dict[str, Optional[str]]:
        mapping: Dict[str, Optional[str]] = {}
        for value, meaning in self._evidences[code]["value_meaning"].items():
            label = _label(meaning)
            if label in _NA_VALUES:
                mapping[value] = None          # placeholder -> null, never a category
                continue
            if label not in curated:
                raise OntologyError(
                    f"{code} value {value} ({label!r}) is not in the frozen vocabulary.")
            mapping[value] = curated[label]
        return mapping

    # ------------------------------------------------------------------
    # public lookups
    # ------------------------------------------------------------------
    @staticmethod
    def severity_to_urgency(severity: Optional[int]) -> Optional[str]:
        """DDXPlus severity is INVERTED: 1 = most urgent (Phase 4.7 §2.4)."""
        return {1: "EMERGENCY", 2: "URGENT", 3: "SEMI_URGENT",
                4: "NON_URGENT", 5: "NON_URGENT"}.get(severity)

    def resolve_disease(self, pathology: str) -> Dict[str, Any]:
        try:
            return self.disease_by_pathology[pathology]
        except KeyError as exc:
            raise OntologyError(f"Unknown pathology: {pathology!r}") from exc

    def is_known_evidence(self, code: str) -> bool:
        return code in self._evidences

    def evidence_target(self, code: str) -> Optional[str]:
        """Coarse routing label for a binary evidence (diagnostics/QC)."""
        if code in self.symptom_id_by_code:
            return "symptom"
        if code in self.history_feature_by_code:
            return "history"
        if code in self.lifestyle_feature_by_code or code in self.smoking_status_by_code \
                or code in self.occupation_by_code or code in self.residence_by_code:
            return "lifestyle"
        if code in self.exposure_feature_by_code:
            return "exposure"
        if code in self.anthro_feature_by_code:
            return "anthropometric"
        if code in self.repro_feature_by_code:
            return "reproductive"
        if code in PREGNANCY:
            return "demographics"
        if code in ETHNICITY:
            return "excluded"
        if code in DESCRIPTOR_CODES or code in TRAVEL:
            return "descriptor"
        return None

    def summary(self) -> Dict[str, int]:
        return {
            "evidences": len(self._evidences),
            "conditions": len(self._conditions),
            "symptom_ids": len(self.symptom_ids),
            "chronic": len(self.chronic_codes),
            "locations": len(self.location_by_value),
            "qualities": len({v for v in self.quality_by_value.values() if v}),
        }
