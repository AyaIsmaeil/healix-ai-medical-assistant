"""model_loader: loads the XGBoost disease-prediction bundle
(ml/models/xgboost-healix-arabic-v1.0/) and verifies its integrity before
anything in this package trusts it.

Unlike the earlier vendored bundle this replaces, this one is first-party:
produced in-repo by ml/training/disease_symptom_checklist_training.ipynb
directly from rag/knowledge_base/*.json (plus a supplementary webteb.com
scrape), synthetic symptom combinations filtered to
vocabulary/symptoms.py's canonical Arabic terms only. See that notebook
and CLAUDE.md's XGBoost corroboration-signal section for the full
provenance. This module still owns nothing about disease/feature
semantics; that is nodes.ml_corroborate's job. It only answers "is this
bundle the one it claims to be, and internally consistent."

--- Fails loudly, once, on first use ---

load_bundle() verifies SHA256 checksums against checksums.json and
cross-checks model.n_features_in_ / feature_schema.json's column count /
metadata.json's n_features (and the label encoder's class count against
metadata.json's n_classes) before returning anything. Any mismatch raises
MLModelError — a corrupt or substituted bundle must never be used
silently. This deliberately mirrors the "verify before trust" discipline
CLAUDE.md > Symptom vocabulary already applies to rules/red_flags.py's
own import-time validation, just at a different layer.

Unlike that import-time validation, this is NOT run at Python import
time: loading model.joblib (via joblib/xgboost) is real I/O and CPU work
that every test importing this module would otherwise pay for, even a
test that only touches ml.disease_crosswalk and never runs inference —
the same reasoning rag.schema.load_all() is an explicit function call,
not an import-time side effect. Call load_bundle() once, explicitly,
from wherever this package's first real use happens
(nodes.ml_corroborate); the result is cached module-globally after the
first successful call, since re-loading a 6MB joblib file per graph turn
would be wasteful — the knowledge-base precedent (rag.schema.load_all()
reloading every call) does not apply here because that file is small
JSON, not a multi-megabyte serialized model.

--- Inference-time errors are a different node's problem ---

This module only concerns itself with the bundle's integrity at load
time. Whether predict_proba() itself later raises on a particular input
is nodes.ml_corroborate's fail-open responsibility (CLAUDE.md: XGBoost is
explicitly non-critical-path at inference time, even though a corrupt
bundle at startup is treated as fatal here).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

BUNDLE_DIR = Path(__file__).parent / "models" / "xgboost-healix-arabic-v1.0"


class MLModelError(Exception):
    """Raised when the vendored model bundle is missing, corrupt, or
    internally inconsistent. Never caught to "fall back" to anything —
    a bad bundle must be fixed, not silently worked around."""


@dataclass(frozen=True)
class MLBundle:
    model: Any  # xgboost.sklearn.XGBClassifier
    label_encoder: Any  # sklearn.preprocessing.LabelEncoder
    feature_order: tuple[str, ...]  # feature_schema.json's model_feature_order, in order
    metadata: dict[str, Any]


_CHECKSUM_FILES = ("model.joblib", "label_encoder.joblib", "feature_schema.json", "metadata.json")

_cached_bundle: MLBundle | None = None


def _verify_checksums(directory: Path) -> None:
    checksums_path = directory / "checksums.json"
    if not checksums_path.exists():
        raise MLModelError(f"missing checksums.json in {directory}")

    expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    for filename in _CHECKSUM_FILES:
        expected_hash = expected.get(filename)
        if not expected_hash:
            raise MLModelError(f"checksums.json has no entry for {filename}")

        file_path = directory / filename
        if not file_path.exists():
            raise MLModelError(f"bundle is missing {filename} (expected by checksums.json)")

        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_hash.lower() != expected_hash.lower():
            raise MLModelError(
                f"checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )


def _load_feature_order(directory: Path) -> tuple[str, ...]:
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    try:
        feature_order = schema["layers"]["model_feature_order"]
    except KeyError as exc:
        raise MLModelError("feature_schema.json missing layers.model_feature_order") from exc
    if not feature_order:
        raise MLModelError("feature_schema.json's model_feature_order is empty")
    return tuple(feature_order)


def load_bundle(*, directory: Path = BUNDLE_DIR) -> MLBundle:
    """Load, verify, and cache the vendored model bundle.

    Raises MLModelError on any checksum mismatch or internal
    inconsistency (feature/class count disagreement between the model
    object, feature_schema.json, and metadata.json) — a bundle that fails
    this check is never returned, cached, or partially trusted.
    """
    global _cached_bundle
    if _cached_bundle is not None and directory == BUNDLE_DIR:
        return _cached_bundle

    _verify_checksums(directory)

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    feature_order = _load_feature_order(directory)
    model = joblib.load(directory / "model.joblib")
    label_encoder = joblib.load(directory / "label_encoder.joblib")

    n_features_in = getattr(model, "n_features_in_", None)
    if n_features_in != len(feature_order):
        raise MLModelError(
            f"model.n_features_in_={n_features_in} does not match "
            f"feature_schema.json's {len(feature_order)} columns"
        )
    if n_features_in != metadata.get("n_features"):
        raise MLModelError(
            f"model.n_features_in_={n_features_in} does not match "
            f"metadata.json's n_features={metadata.get('n_features')}"
        )

    n_classes = len(getattr(label_encoder, "classes_", ()))
    if n_classes != metadata.get("n_classes"):
        raise MLModelError(
            f"label_encoder has {n_classes} classes, metadata.json says "
            f"n_classes={metadata.get('n_classes')}"
        )

    bundle = MLBundle(
        model=model,
        label_encoder=label_encoder,
        feature_order=feature_order,
        metadata=metadata,
    )
    if directory == BUNDLE_DIR:
        _cached_bundle = bundle
    return bundle


def _reset_cache_for_tests() -> None:
    """Test-only: clears the module-level cache so a test can force a
    fresh load_bundle() call (e.g. to exercise a checksum-mismatch path
    against a temp directory without a stale real bundle leaking in)."""
    global _cached_bundle
    _cached_bundle = None
