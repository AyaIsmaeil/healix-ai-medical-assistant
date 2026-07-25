"""
Healix - Evidence Parser (Phase 5, Stage 2).

Decodes the DDXPlus ``EVIDENCES`` token grammar into a typed, nested raw-value
mapping whose key paths mirror ``ClinicalFeatureSet`` (``demographics.age``,
``symptom_presence.HEALIX_SYMPTOM_0001``, ...). That mapping is the
"ClinicalFeatureSet-equivalent" stage of the pipeline: it carries exactly the
same field paths the Feature Schema declares in its ``source`` strings, so the
encoder can resolve columns declaratively.

Token grammar (verified in docs/research/DDXPLUS_EDA.md §3.1):
    E_XX             -> binary evidence present
    E_XX_@_N         -> ordinal / integer value
    E_XX_@_V_YY      -> coded value (repeatable for multi-valued evidences)

Absence semantics: DDXPlus simulates a *complete* interview, so an evidence
that is not listed is a true negative (``0``), not "unknown". This is the
``closed_world`` convention and it is recorded in the manifest — the runtime is
``open_world`` and the difference must never be silently mixed.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

from app.ml.ontology_mapper import OntologyMapper


class EvidenceParseError(ValueError):
    """Raised when a row cannot be parsed; the row is quarantined, never skipped."""


# Descriptor evidence -> (raw-value path, kind)
_DESCRIPTOR_ROUTES = {
    "E_56": ("primary_symptom.descriptors.severity_0_10", "int"),
    "E_59": ("primary_symptom.descriptors.onset_speed", "int"),
    "E_58": ("primary_symptom.descriptors.pain_precision", "int"),
    "E_132": ("primary_symptom.descriptors.rash_swollen", "int"),
    "E_136": ("primary_symptom.descriptors.rash_itch", "int"),
    "E_135": ("primary_symptom.descriptors.lesion_gt_1cm", "yesno"),
    "E_131": ("primary_symptom.descriptors.lesion_peels", "yesno"),
    "E_130": ("primary_symptom.descriptors.rash_color", "rash_color"),
    "E_54": ("primary_symptom.descriptors.quality", "quality"),
    "E_55": ("primary_symptom.descriptors.location", "location"),
    "E_133": ("primary_symptom.descriptors.location", "location"),
    "E_152": ("primary_symptom.descriptors.location", "location"),
    "E_57": ("primary_symptom.descriptors.radiation", "location"),
    "E_204": ("exposure.travel_region", "travel"),
}

_YESNO = {"V_12": True, "V_10": False}   # Y / N


def _set_path(target: Dict[str, Any], path: str, value: Any) -> None:
    node = target
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def split_token(token: str) -> Tuple[str, Optional[str]]:
    """``E_56_@_4`` -> ("E_56", "4"); ``E_55_@_V_161`` -> ("E_55", "V_161")."""
    if "_@_" not in token:
        return token, None
    code, _, value = token.partition("_@_")
    return code, value


class EvidenceParser:
    """Turns one DDXPlus CSV row into a nested raw-value mapping."""

    def __init__(self, mapper: OntologyMapper):
        self._m = mapper

    # ------------------------------------------------------------------
    def parse_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}

        # --- demographics -------------------------------------------------
        _set_path(raw, "demographics.age", self._as_int(row.get("AGE")))
        sex = (row.get("SEX") or "").strip().upper()
        _set_path(raw, "demographics.gender",
                  {"M": "male", "F": "female"}.get(sex))

        # --- closed-world symptom presence: default every symptom to False --
        presence: Dict[str, Optional[bool]] = {
            sid: False for sid in self._m.symptom_id_by_code.values()
        }
        # Healix-only symptoms have no DDXPlus evidence -> unknown, not absent.
        for sid in ("HEALIX_SYMPTOM_0097", "HEALIX_SYMPTOM_0098"):
            presence[sid] = None
        raw["symptom_presence"] = presence

        # --- closed-world history / lifestyle / exposure defaults -----------
        # Written at the exact dotted path the schema declares as `source`
        # (medical_history.chronic_diseases.E_69, ...), so the encoder resolves it.
        for code, path in self._m.history_path_by_code.items():
            _set_path(raw, f"medical_history.{path}", False)
        for feature in self._m.lifestyle_feature_by_code.values():
            _set_path(raw, f"lifestyle.{feature}", False)
        for feature in self._m.exposure_feature_by_code.values():
            _set_path(raw, f"exposure.{feature}", False)
        for feature in self._m.anthro_feature_by_code.values():
            _set_path(raw, f"anthropometric.{feature}", False)
        for feature in self._m.repro_feature_by_code.values():
            _set_path(raw, f"reproductive.{feature}", False)
        _set_path(raw, "demographics.pregnancy_possible", False)
        _set_path(raw, "lifestyle.smoking", False)
        _set_path(raw, "lifestyle.smoking_status", "never")

        locations: List[str] = []
        radiations: List[str] = []
        lateralities: List[str] = []

        # --- decode the evidence tokens ------------------------------------
        for token in self._literal_list(row.get("EVIDENCES"), "EVIDENCES"):
            code, value = split_token(str(token))
            if not self._m.is_known_evidence(code):
                raise EvidenceParseError(f"Unknown evidence code: {code!r}")

            # 1) binary presence
            if value is None:
                self._apply_binary(raw, code, presence)
                continue

            # 2) valued evidence
            route = _DESCRIPTOR_ROUTES.get(code)
            if route is None:
                # A valued token on a non-descriptor evidence still means "present".
                self._apply_binary(raw, code, presence)
                continue

            path, kind = route
            if kind == "int":
                _set_path(raw, path, self._as_int(value))
            elif kind == "yesno":
                _set_path(raw, path, _YESNO.get(value))
            elif kind == "rash_color":
                _set_path(raw, path, self._m.rash_color_by_value.get(value))
            elif kind == "quality":
                _set_path(raw, path, self._m.quality_by_value.get(value))
            elif kind == "travel":
                _set_path(raw, path, self._m.travel_by_value.get(value))
            elif kind == "location":
                region = self._m.location_by_value.get(value)
                lat = self._m.laterality_by_value.get(value)
                if region is None:
                    raise EvidenceParseError(f"Unknown location value: {value!r}")
                (radiations if code == "E_57" else locations).append(region)
                if lat:
                    lateralities.append(lat)

        # --- reduce multi-valued location/radiation/laterality --------------
        _set_path(raw, "primary_symptom.descriptors.location",
                  self._dominant(locations))
        _set_path(raw, "primary_symptom.descriptors.radiation",
                  self._dominant(radiations))
        _set_path(raw, "primary_symptom.descriptors.laterality",
                  self._laterality(lateralities))

        # --- primary symptom (chief complaint) ------------------------------
        initial = (row.get("INITIAL_EVIDENCE") or "").strip()
        _set_path(raw, "primary_symptom.healix_id",
                  self._m.symptom_id_by_code.get(initial))

        # --- derived --------------------------------------------------------
        positives = sum(1 for v in presence.values() if v is True)
        negatives = sum(1 for v in presence.values() if v is False)
        _set_path(raw, "derived.symptom_count", positives)
        _set_path(raw, "derived.positive_negative_ratio",
                  round(positives / negatives, 4) if negatives else None)
        # Not derivable from DDXPlus (Phase 4.6 F5) -> explicit nulls.
        _set_path(raw, "derived.has_red_flag", None)
        _set_path(raw, "derived.interview_completeness_ratio", None)
        _set_path(raw, "validation.validity_score", None)
        # DDXPlus has no numeric temperature (fever is a boolean symptom).
        raw["temperature_c"] = None
        _set_path(raw, "primary_symptom.descriptors.onset_days_ago", None)
        _set_path(raw, "primary_symptom.descriptors.progression", None)

        return raw

    # ------------------------------------------------------------------
    def _apply_binary(self, raw: Dict[str, Any], code: str,
                      presence: Dict[str, Optional[bool]]) -> None:
        m = self._m
        if code in m.symptom_id_by_code:
            presence[m.symptom_id_by_code[code]] = True
        elif code in m.history_path_by_code:
            _set_path(raw, f"medical_history.{m.history_path_by_code[code]}", True)
        elif code in m.lifestyle_feature_by_code:
            _set_path(raw, f"lifestyle.{m.lifestyle_feature_by_code[code]}", True)
        elif code in m.smoking_status_by_code:
            _set_path(raw, "lifestyle.smoking_status", m.smoking_status_by_code[code])
            if m.smoking_status_by_code[code] == "current":
                _set_path(raw, "lifestyle.smoking", True)
        elif code in m.occupation_by_code:
            _set_path(raw, "lifestyle.occupation", m.occupation_by_code[code])
        elif code in m.residence_by_code:
            _set_path(raw, "lifestyle.residence", m.residence_by_code[code])
        elif code in m.exposure_feature_by_code:
            _set_path(raw, f"exposure.{m.exposure_feature_by_code[code]}", True)
        elif code in m.anthro_feature_by_code:
            _set_path(raw, f"anthropometric.{m.anthro_feature_by_code[code]}", True)
        elif code in m.repro_feature_by_code:
            _set_path(raw, f"reproductive.{m.repro_feature_by_code[code]}", True)
        elif code == "E_167":
            _set_path(raw, "demographics.pregnancy_possible", True)
        # E_17 (ethnicity) is deliberately excluded from the schema; ignore silently.

    @staticmethod
    def _dominant(values: List[str]) -> Optional[str]:
        """Pick a single region deterministically: most frequent, then alphabetical.

        Multi-valued location evidences may name several regions; the schema
        declares one categorical column, so a deterministic reduction is
        required. Ties break alphabetically so the result never depends on
        token order.
        """
        if not values:
            return None
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    @staticmethod
    def _laterality(values: List[str]) -> Optional[str]:
        sides = {v for v in values if v in ("left", "right")}
        if len(sides) == 2:
            return "bilateral"
        if sides:
            return sides.pop()
        return "midline" if values else None

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _literal_list(value: Any, field: str) -> List[Any]:
        """DDXPlus list columns are *Python* literals (single quotes), not JSON."""
        text = (value or "").strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise EvidenceParseError(f"Malformed {field} literal: {exc}") from exc
        if not isinstance(parsed, list):
            raise EvidenceParseError(f"{field} is not a list")
        return parsed

    # ------------------------------------------------------------------
    def parse_differential(self, value: Any) -> List[Tuple[str, float]]:
        """``[['Bronchitis', 0.19], ...]`` -> [(pathology, probability), ...]."""
        out: List[Tuple[str, float]] = []
        for item in self._literal_list(value, "DIFFERENTIAL_DIAGNOSIS"):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise EvidenceParseError("Malformed differential entry")
            out.append((str(item[0]), float(item[1])))
        return out
