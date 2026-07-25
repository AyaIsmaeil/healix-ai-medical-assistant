"""
Healix - Phase 6.2 explainability (OFFLINE ONLY).

Three complementary views of feature importance, because each alone is
misleading:

* **Impurity (Gini) importance** - free from a fitted forest, but biased
  toward high-cardinality features and computed on TRAINING data only.
* **Permutation importance** - measures the actual drop in macro-F1 on HELD
  OUT data when a column is shuffled. Slower, but it answers "does this
  feature carry real predictive signal?" rather than "did the trees like
  splitting on it?".
* **SHAP** - per-prediction attribution; the only one that shows direction
  and interaction. Sampled, because exact TreeSHAP on a 200-tree/49-class
  forest over 286 features is prohibitively expensive at full scale.

Every reported feature is mapped back to the HEALIX ontology (symptom IDs ->
human-readable question text, history/lifestyle/exposure codes -> their
clinical meaning) using the dataset's own feature_dictionary.json plus the
DDXPlus release dictionaries, so the output is clinically readable rather
than a list of opaque column names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from training.disease_prediction.config import SEED


# ----------------------------------------------------------------------
# Ontology-aware feature naming
# ----------------------------------------------------------------------
class FeatureOntologyResolver:
    """Maps a Feature-Schema-v2 column name to human-readable clinical text."""

    def __init__(self, dataset_dir: Path, ddxplus_dir: Optional[Path] = None):
        fd = json.loads((dataset_dir / "metadata" / "feature_dictionary.json")
                        .read_text(encoding="utf-8"))
        self.features: Dict[str, Any] = fd["features"]
        self.feature_order: List[str] = fd["feature_order"]

        self._evidence_text: Dict[str, str] = {}
        if ddxplus_dir and (ddxplus_dir / "release_evidences.json").exists():
            ev = json.loads((ddxplus_dir / "release_evidences.json")
                            .read_text(encoding="utf-8"))
            for code, spec in ev.items():
                q = (spec.get("question_en") or "").strip()
                if q:
                    self._evidence_text[code] = q

    def describe(self, column: str) -> Dict[str, Any]:
        spec = self.features.get(column, {})
        ontology_ref = spec.get("ontology_ref")
        group = spec.get("group")
        kind = spec.get("kind")

        # symptom columns carry their HEALIX id in the name
        healix_id = None
        if column.startswith("sym_HEALIX_SYMPTOM_"):
            healix_id = column[len("sym_"):]

        # history/lifestyle/exposure columns embed the DDXPlus evidence code
        evidence_code = ontology_ref
        if evidence_code is None:
            for prefix in ("hx_chronic_", "hx_med_", "hx_allergy_",
                           "hx_family_", "hx_surgery_"):
                if column.startswith(prefix):
                    evidence_code = column[len(prefix):]
                    break

        question = self._evidence_text.get(evidence_code) if evidence_code else None
        return {
            "column": column,
            "group": group,
            "kind": kind,
            "healix_symptom_id": healix_id,
            "ddxplus_evidence": evidence_code,
            "clinical_meaning": question,
        }


# ----------------------------------------------------------------------
# Importance computations
# ----------------------------------------------------------------------
def impurity_importance(model, feature_order: List[str],
                        top_n: int = 50) -> List[Dict[str, Any]]:
    if not hasattr(model, "feature_importances_"):
        return []
    imp = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(imp)[::-1][:top_n]
    total = float(imp.sum()) or 1.0
    return [{"rank": r + 1, "column": feature_order[i],
             "importance": round(float(imp[i]), 8),
             "importance_pct": round(100 * float(imp[i]) / total, 4)}
            for r, i in enumerate(order)]


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                   feature_order: List[str],
                                   n_repeats: int = 3, top_n: int = 50,
                                   seed: int = SEED,
                                   n_jobs: int = 1) -> List[Dict[str, Any]]:
    """Permutation importance on HELD-OUT data, scored by macro-F1.

    Scored with macro-F1 (not accuracy) to stay consistent with the primary
    metric used everywhere else in Phases 6.1/6.2, given the 247:1 imbalance.
    """
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    result = permutation_importance(
        model, X, y, scoring=scorer, n_repeats=n_repeats,
        random_state=seed, n_jobs=n_jobs)
    means = result.importances_mean
    stds = result.importances_std
    order = np.argsort(means)[::-1][:top_n]
    return [{"rank": r + 1, "column": feature_order[i],
             "importance_mean": round(float(means[i]), 8),
             "importance_std": round(float(stds[i]), 8)}
            for r, i in enumerate(order)]


def compute_shap_importance(model, X_background: np.ndarray,
                            X_explain: np.ndarray, feature_order: List[str],
                            top_n: int = 50) -> Dict[str, Any]:
    """Mean |SHAP| per feature, aggregated across classes and samples.

    Uses INTERVENTIONAL TreeSHAP with the supplied background sample. Per
    SHAP's own docs, omitting ``data=`` makes ``TreeExplainer`` silently fall
    back to ``tree_path_dependent`` (background-free) even if a background
    array is drawn and reported elsewhere — that mismatch between the
    reported and actual method was a reproducibility bug in an earlier
    version of this function, fixed here.

    The additivity check (Shapley values sum to the model's output) is run
    with its default of ``True`` first. It is only disabled, and the
    triggering exception recorded, if it actually fails — never pre-emptively
    — since silently disabling it can hide real computation errors, not just
    the known hist-tree numerical false positive (see SHAP issues #1105,
    #1744).

    Returns a structured error instead of raising if SHAP is unavailable or
    the model is unsupported — explainability must never break the pipeline.
    """
    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"shap not importable: {exc}"}

    try:
        explainer = shap.TreeExplainer(
            model, data=X_background, feature_perturbation="interventional")

        additivity_check_enabled = True
        additivity_check_disabled_reason: Optional[str] = None
        try:
            values = explainer.shap_values(X_explain, check_additivity=True)
        except Exception as exc:  # noqa: BLE001
            additivity_check_enabled = False
            additivity_check_disabled_reason = f"{type(exc).__name__}: {exc}"
            values = explainer.shap_values(X_explain, check_additivity=False)

        # shap returns either a list (one array per class) or a 3-D array
        if isinstance(values, list):
            stacked = np.stack([np.abs(v) for v in values], axis=0)   # (C,N,F)
            mean_abs = stacked.mean(axis=(0, 1))
        else:
            arr = np.abs(np.asarray(values))
            if arr.ndim == 3:      # (N, F, C) or (C, N, F)
                axes = tuple(i for i in range(arr.ndim)
                             if arr.shape[i] != len(feature_order))
                mean_abs = arr.mean(axis=axes)
            else:
                mean_abs = arr.mean(axis=0)
        mean_abs = np.asarray(mean_abs, dtype=float).ravel()
        if mean_abs.shape[0] != len(feature_order):
            return {"available": False,
                    "error": f"SHAP shape mismatch: got {mean_abs.shape[0]} "
                             f"values for {len(feature_order)} features"}

        order = np.argsort(mean_abs)[::-1][:top_n]
        return {
            "available": True,
            "feature_perturbation": "interventional",
            "n_background": int(len(X_background)),
            "n_explained": int(len(X_explain)),
            "additivity_check_enabled": additivity_check_enabled,
            "additivity_check_disabled_reason": additivity_check_disabled_reason,
            "top_features": [
                {"rank": r + 1, "column": feature_order[i],
                 "mean_abs_shap": round(float(mean_abs[i]), 8)}
                for r, i in enumerate(order)],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def plot_top_features(entries: List[Dict[str, Any]], value_key: str,
                      title: str, path, top_n: int = 25) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    try:
        subset = entries[:top_n][::-1]
        labels = [e["column"] for e in subset]
        vals = [e[value_key] for e in subset]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(subset))))
        ax.barh(range(len(subset)), vals, color="#2b8cbe")
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(value_key)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001
        return None
