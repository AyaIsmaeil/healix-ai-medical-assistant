"""
Healix - Quality Validator (Phase 5, Stage 5).

Implements the quality gates from ``docs/research/DATASET_BUILDER_DESIGN.md`` §7
and the mitigation adopted in ``docs/research/CROSS_SPLIT_LEAKAGE_REPORT.md``.

Guiding principle: **flag, never silently drop.** Rows are never removed and the
official split is never altered. Problems are recorded, counted, and surfaced;
rows that cannot be parsed at all go to a quarantine table with a reason.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Set

# Ranges reused verbatim from app/dictionaries/feature_validation_rules.json so
# that offline validation and runtime validation cannot drift apart.
RANGE_RULES: Dict[str, Dict[str, Any]] = {
    "age": {"min": 0, "max": 120, "on_out_of_range": "reject"},
    "temperature_c": {"min": 30.0, "max": 45.0, "on_out_of_range": "reject"},
    "severity": {"min": 0, "max": 10, "on_out_of_range": "clip"},
    "onset_days_ago": {"min": 0, "max": 3650, "on_out_of_range": "reject"},
    "onset_speed": {"min": 0, "max": 10, "on_out_of_range": "clip"},
    "pain_precision": {"min": 0, "max": 10, "on_out_of_range": "clip"},
    "rash_swollen": {"min": 0, "max": 10, "on_out_of_range": "clip"},
    "rash_itch": {"min": 0, "max": 10, "on_out_of_range": "clip"},
    "validity_score": {"min": 0.0, "max": 1.0, "on_out_of_range": "clip"},
    "interview_completeness_ratio": {"min": 0.0, "max": 1.0, "on_out_of_range": "clip"},
    "symptom_count": {"min": 0, "max": None, "on_out_of_range": "reject"},
}

FEATURE_HASH_COLUMNS = ("AGE", "SEX", "EVIDENCES", "INITIAL_EVIDENCE")


def content_hash(row: Dict[str, str],
                 columns: Iterable[str] = FEATURE_HASH_COLUMNS) -> str:
    """Stable 128-bit content hash of the model *inputs* (leakage definition)."""
    payload = "\x1f".join((row.get(c) or "") for c in columns)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


class QualityValidator:
    """Accumulates quality findings across chunks; emits a report at the end."""

    def __init__(self, split: str, leaked_hashes: Optional[Set[str]] = None):
        self.split = split
        self._leaked = leaked_hashes or set()
        self.rows = 0
        self.quarantined = 0
        self.leaked_rows = 0
        self.hash_counts: Counter = Counter()
        self.range_violations: Counter = Counter()
        self.clipped: Counter = Counter()
        self.impossible: Counter = Counter()
        self.unknown_ids: Counter = Counter()
        self.missing_mandatory = 0
        self.ambiguous_inputs: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    def observe(self, raw_row: Dict[str, str], encoded: Dict[str, Any],
                labels: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one row. Returns per-row QC annotations (never drops it)."""
        self.rows += 1
        h = content_hash(raw_row)
        self.hash_counts[h] += 1

        is_leaked = h in self._leaked
        if is_leaked:
            self.leaked_rows += 1

        # --- mandatory fields ------------------------------------------
        if not labels.get("y_disease"):
            self.missing_mandatory += 1

        # --- numeric ranges (mirrors FeatureValidator semantics) --------
        for column, rule in RANGE_RULES.items():
            value = encoded.get(column)
            if value is None:
                continue
            low, high = rule["min"], rule["max"]
            if (low is not None and value < low) or (high is not None and value > high):
                if rule["on_out_of_range"] == "clip":
                    self.clipped[column] += 1
                    encoded[column] = max(low, min(high, value))
                else:
                    self.range_violations[column] += 1
                    encoded[column] = None

        # --- impossible combinations -----------------------------------
        # Grounded on the cross-field rule the runtime FeatureValidator enforces.
        if encoded.get("gender_male") == 1 and encoded.get("pregnancy_possible") == 1:
            self.impossible["pregnancy_with_male"] += 1

        # --- label ambiguity: identical inputs, conflicting ground truth --
        self.ambiguous_inputs.setdefault(h, set()).add(labels.get("y_disease", ""))

        return {
            "content_hash": h,
            "duplicate_group_id": h,
            "leakage_flag": 1 if is_leaked else 0,
        }

    def note_unknown(self, kind: str) -> None:
        self.unknown_ids[kind] += 1

    def note_quarantine(self) -> None:
        self.quarantined += 1

    # ------------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        redundant = sum(v - 1 for v in self.hash_counts.values() if v > 1)
        ambiguous = {h: sorted(d) for h, d in self.ambiguous_inputs.items()
                     if len(d) > 1}
        return {
            "split": self.split,
            "rows": self.rows,
            "unique_records": len(self.hash_counts),
            "duplicate_groups": sum(1 for v in self.hash_counts.values() if v > 1),
            "redundant_rows": redundant,
            "duplicate_pct": round(100 * redundant / self.rows, 4) if self.rows else 0.0,
            "leaked_rows": self.leaked_rows,
            "leaked_pct": round(100 * self.leaked_rows / self.rows, 4) if self.rows else 0.0,
            "quarantined_rows": self.quarantined,
            "missing_mandatory": self.missing_mandatory,
            "range_rejected": dict(self.range_violations),
            "range_clipped": dict(self.clipped),
            "impossible_combinations": dict(self.impossible),
            "unknown_ontology_ids": dict(self.unknown_ids),
            "ambiguous_input_groups": len(ambiguous),
            "ambiguous_input_rows": sum(self.hash_counts[h] for h in ambiguous),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def compute_leaked_hashes(hashes_by_split: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Hashes that appear in more than one split (the leakage gate)."""
        leaked: Dict[str, Set[str]] = {s: set() for s in hashes_by_split}
        names = list(hashes_by_split)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = hashes_by_split[a] & hashes_by_split[b]
                leaked[a] |= shared
                leaked[b] |= shared
        return leaked
