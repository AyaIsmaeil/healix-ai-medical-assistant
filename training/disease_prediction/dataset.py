"""
Healix - Disease Prediction dataset loader (Phase 6.1).

Loads the Phase-5 Parquet dataset (Feature Schema v2, 286 columns) and turns
it into a plain float32 feature matrix + integer disease labels. This module
NEVER regenerates the dataset, NEVER touches the ontology, and NEVER modifies
Feature Schema v2 — it only reads what Phase 5 already produced.

Label encoding is derived from ``label_dictionary.json`` (the ontology's own
49-disease list), not from whichever classes happen to appear in a split —
so the mapping HEALIX_DISEASE_XXXX -> class index is identical and
reproducible across train/validation/test regardless of class presence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from training.disease_prediction.config import (LABEL_AND_QC_COLUMNS,
                                                 LEAKAGE_FLAG_COLUMN,
                                                 TARGET_COLUMN)

SPLIT_FILES = {"train": "train.parquet", "validation": "validation.parquet",
              "test": "test.parquet"}


@dataclass
class DatasetBundle:
    """Everything needed to train/evaluate one split, nothing more."""

    X: np.ndarray                  # float32, shape (n_rows, n_features)
    y: np.ndarray                  # int64 class indices, shape (n_rows,)
    leakage_flag: np.ndarray       # int8, shape (n_rows,) — 1 = flagged (Phase 5)
    feature_order: List[str]
    n_rows: int

    @property
    def clean_mask(self) -> np.ndarray:
        """Rows with NO cross-split leakage flag (docs/research/
        CROSS_SPLIT_LEAKAGE_REPORT.md §5.1 — the PRIMARY evaluation subset)."""
        return self.leakage_flag == 0


class DiseaseLabelEncoder:
    """HEALIX_DISEASE_XXXX <-> class index, sourced from the ontology's own
    label_dictionary.json — deterministic regardless of which classes are
    physically present in any one split."""

    def __init__(self, disease_ids: List[str]):
        self.classes_: List[str] = sorted(disease_ids)
        self._index_by_id: Dict[str, int] = {
            cid: i for i, cid in enumerate(self.classes_)}

    @classmethod
    def from_label_dictionary(cls, dataset_dir: Path) -> "DiseaseLabelEncoder":
        label_dict = json.loads(
            (dataset_dir / "metadata" / "label_dictionary.json")
            .read_text(encoding="utf-8"))
        return cls(list(label_dict["diseases"].keys()))

    def transform(self, disease_ids) -> np.ndarray:
        try:
            return np.array([self._index_by_id[d] for d in disease_ids],
                            dtype=np.int64)
        except KeyError as exc:
            raise ValueError(
                f"Unknown disease id in data, not present in the ontology's "
                f"label_dictionary.json: {exc}") from exc

    def inverse_transform(self, indices) -> List[str]:
        return [self.classes_[i] for i in indices]

    @property
    def n_classes(self) -> int:
        return len(self.classes_)


def load_feature_order(dataset_dir: Path) -> List[str]:
    feature_dict = json.loads(
        (dataset_dir / "metadata" / "feature_dictionary.json")
        .read_text(encoding="utf-8"))
    order = feature_dict["feature_order"]
    assert len(order) == feature_dict["n_features"], (
        "feature_dictionary.json is internally inconsistent: "
        f"len(feature_order)={len(order)} != n_features={feature_dict['n_features']}")
    return order


def load_split(dataset_dir: Path, split: str, feature_order: List[str],
               limit: Optional[int] = None) -> pd.DataFrame:
    """Read only the columns actually needed (features + target + leakage
    flag) — avoids loading the list-typed differential columns or the
    string hash columns into memory at all."""
    path = dataset_dir / "datasets" / SPLIT_FILES[split]
    columns = list(feature_order) + [TARGET_COLUMN, LEAKAGE_FLAG_COLUMN]
    table = pq.read_table(path, columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pandas()


def to_bundle(df: pd.DataFrame, feature_order: List[str],
             encoder: DiseaseLabelEncoder) -> DatasetBundle:
    """Vectorized, memory-lean conversion to a float32 matrix.

    Nulls are preserved as NaN (never coerced to 0) — this is the same
    null-honest contract Feature Schema v2 declares for the runtime encoder;
    tree-ensemble models consume NaN natively, and only the Logistic
    Regression baseline needs a downstream imputer (applied in models.py,
    not here, so the raw matrix stays a faithful copy of the schema).
    """
    X = df[feature_order].to_numpy(dtype=np.float32, na_value=np.nan)
    y = encoder.transform(df[TARGET_COLUMN].to_numpy())
    leakage_flag = df[LEAKAGE_FLAG_COLUMN].to_numpy(dtype=np.int8)
    return DatasetBundle(X=X, y=y, leakage_flag=leakage_flag,
                        feature_order=feature_order, n_rows=len(df))


def load_bundle(dataset_dir: Path, split: str, encoder: DiseaseLabelEncoder,
                feature_order: Optional[List[str]] = None,
                limit: Optional[int] = None) -> DatasetBundle:
    """One-call convenience: read a split straight into a DatasetBundle."""
    feature_order = feature_order or load_feature_order(dataset_dir)
    df = load_split(dataset_dir, split, feature_order, limit=limit)
    return to_bundle(df, feature_order, encoder)


def verify_no_label_leak_into_features(feature_order: List[str]) -> None:
    """Guard against the trivial-but-fatal mistake of a label column slipping
    into the feature list (e.g. through a future feature_dictionary.json
    edit). Cheap, always run before training."""
    overlap = set(feature_order) & set(LABEL_AND_QC_COLUMNS)
    if overlap:
        raise RuntimeError(
            f"Label/QC columns found inside feature_order — would leak the "
            f"target directly into the model: {sorted(overlap)}")
