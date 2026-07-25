"""
Healix - Statistics Report (Phase 5, Stage 6).

Accumulates distribution, class-balance, null-coverage and feature-coverage
statistics while the builder streams. Everything is computed incrementally so
the full matrix is never held in memory.

The builder never rebalances or imputes — imbalance is *reported* so the model
pipeline can choose a weighting strategy (Phase 6).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class StatisticsAccumulator:
    """Streaming statistics for one split."""

    def __init__(self, split: str, feature_order: List[str]):
        self.split = split
        self.feature_order = feature_order
        self.rows = 0
        self._non_null: Counter = Counter()
        self._positive: Counter = Counter()          # boolean/one-hot == 1
        self._num_min: Dict[str, float] = {}
        self._num_max: Dict[str, float] = {}
        self._num_sum: Dict[str, float] = {}
        self._num_n: Counter = Counter()
        self.disease: Counter = Counter()
        self.urgency: Counter = Counter()
        self.specialty: Counter = Counter()
        self.leaked_by_disease: Counter = Counter()
        self.age: Counter = Counter()
        self.sex: Counter = Counter()
        self.evidence_counts: List[int] = []
        self.differential_lengths: List[int] = []
        self.truth_in_differential = 0

    # ------------------------------------------------------------------
    def observe(self, encoded: Dict[str, Any], labels: Dict[str, Any],
                qc: Dict[str, Any]) -> None:
        self.rows += 1

        for name, value in encoded.items():
            if value is None:
                continue
            self._non_null[name] += 1
            if value == 1:
                self._positive[name] += 1
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                v = float(value)
                self._num_n[name] += 1
                self._num_sum[name] = self._num_sum.get(name, 0.0) + v
                if name not in self._num_min or v < self._num_min[name]:
                    self._num_min[name] = v
                if name not in self._num_max or v > self._num_max[name]:
                    self._num_max[name] = v

        disease = labels.get("y_disease")
        self.disease[disease] += 1
        self.urgency[labels.get("y_urgency_prior")] += 1
        self.specialty[labels.get("y_specialty")] += 1
        if qc.get("leakage_flag"):
            self.leaked_by_disease[disease] += 1
        if labels.get("truth_in_differential"):
            self.truth_in_differential += 1
        self.differential_lengths.append(labels.get("differential_length", 0))

        age = encoded.get("age")
        if age is not None:
            self.age[int(age) // 10 * 10] += 1
        if encoded.get("gender_male") == 1:
            self.sex["male"] += 1
        elif encoded.get("gender_female") == 1:
            self.sex["female"] += 1
        self.evidence_counts.append(int(encoded.get("symptom_count") or 0))

    # ------------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        counts = sorted(self.disease.values(), reverse=True)
        imbalance = round(counts[0] / counts[-1], 2) if counts and counts[-1] else None

        coverage = {
            name: {
                "fill_rate": round(self._non_null[name] / self.rows, 6) if self.rows else 0.0,
                "positive_rate": (round(self._positive[name] / self.rows, 6)
                                  if self.rows else 0.0),
            }
            for name in self.feature_order
        }
        always_null = [n for n in self.feature_order if self._non_null[n] == 0]
        constant = [n for n in self.feature_order
                    if self._non_null[n] == self.rows and self._positive[n] in (0, self.rows)
                    and n not in always_null]

        numeric_stats = {
            name: {
                "min": self._num_min[name],
                "max": self._num_max[name],
                "mean": round(self._num_sum[name] / self._num_n[name], 4),
                "count": self._num_n[name],
            }
            for name in sorted(self._num_n)
        }

        ev = self.evidence_counts
        diff = self.differential_lengths
        return {
            "split": self.split,
            "rows": self.rows,
            "class_distribution": {
                "disease": dict(self.disease.most_common()),
                "urgency_prior": dict(self.urgency),
                "specialty": dict(self.specialty.most_common()),
                "imbalance_ratio_max_over_min": imbalance,
                "n_classes_present": len(self.disease),
            },
            "leakage_by_disease": dict(self.leaked_by_disease.most_common()),
            "demographics": {
                "sex": dict(self.sex),
                "age_decade_histogram": dict(sorted(self.age.items())),
            },
            "symptom_count": {
                "min": min(ev) if ev else None,
                "max": max(ev) if ev else None,
                "mean": round(sum(ev) / len(ev), 3) if ev else None,
            },
            "differential_length": {
                "min": min(diff) if diff else None,
                "max": max(diff) if diff else None,
                "mean": round(sum(diff) / len(diff), 3) if diff else None,
            },
            "truth_in_differential_rate": (
                round(self.truth_in_differential / self.rows, 6) if self.rows else None),
            "numeric_statistics": numeric_stats,
            "feature_coverage": coverage,
            "always_null_columns": always_null,
            "always_null_count": len(always_null),
            "constant_columns": constant,
        }
