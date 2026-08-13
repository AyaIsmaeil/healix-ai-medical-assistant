"""
Healix - Disease Prediction training pipeline (Phase 6.1) — OFFLINE ONLY.

Trains every configured model family on the Phase-5 DDXPlus train split,
evaluates each on the validation split (full + leakage-free subset), and
writes a full comparison report. This script performs NO hyperparameter
search, NO cross-validation, and NO threshold calibration (Phase 6.2).

This script is never imported by app.main, never touches a FastAPI route,
and never replaces RuleBasedDiseasePredictor. It is a standalone, offline
experiment runner.

Run:
    python -m training.disease_prediction.train                  # full run
    python -m training.disease_prediction.train --limit 50000    # smoke test
    python -m training.disease_prediction.train --models xgboost lightgbm
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pyarrow.parquet as pq

from training.disease_prediction import config
from training.disease_prediction.dataset import (SPLIT_FILES,
                                                  DiseaseLabelEncoder,
                                                  load_bundle,
                                                  load_feature_order,
                                                  verify_no_label_leak_into_features)
from training.disease_prediction.evaluate import evaluate_split
from training.disease_prediction.models import model_registry
from training.disease_prediction.utils import (PeakMemorySampler, Timer,
                                                balanced_sample_weights,
                                                directory_size_bytes,
                                                set_global_seed, setup_logger,
                                                write_json)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # noqa: BLE001 - confusion-matrix plots are best-effort only
    _HAS_MPL = False


def _save_confusion_matrix_plot(cm: np.ndarray, class_names: List[str],
                                path) -> Optional[str]:
    if not _HAS_MPL:
        return None
    try:
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="viridis")
        ax.set_title("Confusion matrix (validation, full split)")
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001 - never fail the run over a plot
        return None


def _class_balancing_meta(name: str, y_train: np.ndarray) -> Dict[str, Any]:
    """Describe how imbalance handling is applied for this model family."""
    if not config.USE_BALANCED_CLASS_WEIGHTS:
        return {"enabled": False}

    if name in config.SAMPLE_WEIGHT_AT_FIT_MODELS:
        return {
            "enabled": True,
            "method": "sample_weight_balanced",
            "source": "sklearn.utils.class_weight.compute_sample_weight",
        }

    params = config.MODEL_DEFAULTS.get(name, {})
    if params.get("class_weight") == config.SKLEARN_CLASS_WEIGHT:
        return {
            "enabled": True,
            "method": "class_weight_balanced",
            "class_weight": config.SKLEARN_CLASS_WEIGHT,
        }

    return {"enabled": False}


def _fit_one_model(name: str, spec, X_train: np.ndarray, y_train: np.ndarray,
                   logger) -> Dict[str, Any]:
    logger.info("fitting %s ...", spec.display_name)
    model = spec.build()

    fit_kwargs: Dict[str, Any] = {}
    balancing = _class_balancing_meta(name, y_train)
    if balancing.get("enabled") and name in config.SAMPLE_WEIGHT_AT_FIT_MODELS:
        fit_kwargs["sample_weight"] = balanced_sample_weights(y_train)
        logger.info("%s: balanced sample_weight enabled", spec.display_name)

    with PeakMemorySampler() as mem, Timer() as t:
        model.fit(X_train, y_train, **fit_kwargs)

    model_path = config.MODELS_DIR / f"{name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    size_mb = round(directory_size_bytes(model_path) / 1e6, 3)

    logger.info("%s fit in %.2fs | peak_mem=%.1fMB | size=%.2fMB",
               spec.display_name, t.elapsed, mem.peak_mb, size_mb)

    return {"model": model, "model_path": str(model_path),
            "train_seconds": t.elapsed, "peak_memory_mb": mem.peak_mb,
            "model_size_mb": size_mb,
            "class_balancing": balancing}


def _load_existing_result(name: str, spec, n_train_rows: int,
                          n_validation_rows: int) -> Optional[Dict[str, Any]]:
    """Reassemble a previously-completed model's summary from disk, so a run
    interrupted after some models finished (e.g. an external session/process
    teardown, not a code failure) can be resumed without re-fitting models
    that already trained successfully. Only reused if it was produced against
    the SAME row counts as the current run (never mixes a --limit smoke-test
    artifact into a full-scale comparison, or vice versa)."""
    fit_info_path = config.REPORTS_DIR / f"fit_info_{name}.json"
    val_path = config.REPORTS_DIR / f"validation_{name}.json"
    if not (fit_info_path.exists() and val_path.exists()):
        return None

    fit_info = json.loads(fit_info_path.read_text(encoding="utf-8"))
    if (fit_info.get("n_train_rows") != n_train_rows
            or fit_info.get("n_validation_rows") != n_validation_rows):
        return None

    val_report = json.loads(val_path.read_text(encoding="utf-8"))
    cm_plot_path = config.REPORTS_DIR / f"confusion_matrix_{name}.png"
    return {
        "model": name, "display_name": spec.display_name,
        "handles_native_nan": spec.handles_native_nan,
        "imputation_note": spec.imputation_note,
        "hyperparameters": config.MODEL_DEFAULTS.get(name, {}),
        "model_path": fit_info["model_path"],
        "train_seconds": fit_info["train_seconds"],
        "peak_memory_mb": fit_info["peak_memory_mb"],
        "model_size_mb": fit_info["model_size_mb"],
        "validation_full": val_report["metrics_full_split"],
        "validation_leakage_free": val_report["metrics_leakage_free_subset"],
        "confusion_matrix_plot": str(cm_plot_path) if cm_plot_path.exists() else None,
        "resumed_from_disk": True,
    }


def run(model_names: Optional[List[str]] = None,
       limit: Optional[int] = None) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train", config.LOGS_DIR / f"train_{timestamp}.log")

    set_global_seed(config.SEED)
    logger.info("Healix Disease Prediction — Phase 6.1 baseline training")
    logger.info("python=%s platform=%s", sys.version.split()[0], platform.platform())

    dataset_dir = config.resolve_dataset_dir()
    logger.info("dataset_dir=%s", dataset_dir)
    feature_order = load_feature_order(dataset_dir)
    verify_no_label_leak_into_features(feature_order)
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)
    logger.info("n_features=%d n_classes=%d", len(feature_order), encoder.n_classes)

    with Timer() as t_load:
        train_bundle = load_bundle(dataset_dir, "train", encoder, feature_order,
                                   limit=limit)
        val_bundle = load_bundle(dataset_dir, "validation", encoder, feature_order,
                                 limit=limit)
    logger.info("loaded train=%d validation=%d rows in %.1fs",
               train_bundle.n_rows, val_bundle.n_rows, t_load.elapsed)

    registry = model_registry()
    selected = model_names or list(config.MODEL_FAMILIES)
    unknown = set(selected) - set(registry)
    if unknown:
        raise ValueError(f"Unknown model family: {sorted(unknown)}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for name in selected:
        spec = registry[name]
        fit_info = _fit_one_model(name, spec, train_bundle.X, train_bundle.y, logger)
        model = fit_info.pop("model")

        # Persist fit_info on its own (separate from the evaluation report) so
        # a later invocation can resume without re-fitting this model — see
        # _load_existing_result.
        write_json(config.REPORTS_DIR / f"fit_info_{name}.json", {
            **fit_info, "n_train_rows": train_bundle.n_rows,
            "n_validation_rows": val_bundle.n_rows})

        val_report = evaluate_split(model, val_bundle, encoder.classes_, "validation")
        write_json(config.REPORTS_DIR / f"validation_{name}.json", val_report)

        cm = np.array(val_report["confusion_matrix_full_split"])
        cm_plot = _save_confusion_matrix_plot(
            cm, encoder.classes_, config.REPORTS_DIR / f"confusion_matrix_{name}.png")

        clean = val_report["metrics_leakage_free_subset"]
        full = val_report["metrics_full_split"]
        summary = {
            "model": name,
            "display_name": spec.display_name,
            "handles_native_nan": spec.handles_native_nan,
            "imputation_note": spec.imputation_note,
            "hyperparameters": config.MODEL_DEFAULTS.get(name, {}),
            "class_balancing": fit_info.get("class_balancing"),
            **fit_info,
            "validation_full": full,
            "validation_leakage_free": clean,
            "confusion_matrix_plot": cm_plot,
        }
        results.append(summary)
        logger.info(
            "%s -> val(full) acc=%.4f f1_macro=%.4f | val(clean) acc=%.4f f1_macro=%.4f",
            name, full["accuracy"], full["f1_macro"],
            (clean or full)["accuracy"], (clean or full)["f1_macro"])

    # Merge in any OTHER model family already completed on disk (from an
    # earlier, interrupted invocation of this same script) against the exact
    # same row counts, so the final comparison covers every model ever
    # successfully trained, not just the ones fitted in this call.
    already_have = {r["model"] for r in results}
    for name in config.MODEL_FAMILIES:
        if name in already_have:
            continue
        spec = registry[name]
        prior = _load_existing_result(name, spec, train_bundle.n_rows, val_bundle.n_rows)
        if prior is not None:
            logger.info("resumed %s from a previous run (not re-fitted)", name)
            results.append(prior)

    return _finalize_comparison(results, dataset_dir, len(feature_order),
                                encoder, train_bundle.n_rows, val_bundle.n_rows,
                                started, logger)


def _finalize_comparison(results: List[Dict[str, Any]], dataset_dir,
                         n_features: int, encoder: DiseaseLabelEncoder,
                         n_train_rows: int, n_validation_rows: int,
                         started: datetime, logger) -> Dict[str, Any]:
    """Rank the collected per-model results and write every comparison
    artifact. Shared by both a normal fit-then-compare run and
    finalize_only() (which never fits anything, only merges disk results)."""
    if not results:
        raise RuntimeError(
            "No model results to finalize — nothing has been trained yet. "
            "Run without --finalize-only first.")

    primary_key = config.PRIMARY_SELECTION_METRIC
    ranked = sorted(
        results,
        key=lambda r: (r["validation_leakage_free"] or r["validation_full"])[primary_key],
        reverse=True,
    )
    best = ranked[0]

    comparison = {
        "generated_utc": started.isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "n_train_rows": n_train_rows,
        "n_validation_rows": n_validation_rows,
        "n_features": n_features,
        "n_classes": encoder.n_classes,
        "n_models_reported": len(results),
        "primary_selection_metric": primary_key,
        "primary_selection_split": config.PRIMARY_SELECTION_SPLIT,
        "seed": config.SEED,
        "models": results,
        "ranking": [
            {"rank": i + 1, "model": r["model"],
             "primary_metric_value": (r["validation_leakage_free"] or
                                      r["validation_full"])[primary_key]}
            for i, r in enumerate(ranked)
        ],
        "best_model": best["model"],
    }
    write_json(config.REPORTS_DIR / "model_comparison.json", comparison)
    _write_comparison_csv(results, config.REPORTS_DIR / "model_comparison.csv")

    write_json(config.MODELS_DIR / "label_encoder.json",
              {"classes_": encoder.classes_, "ontology_version":
               config.EXPECTED_ONTOLOGY_VERSION})
    write_json(config.MODELS_DIR / "best_model.json", {
        "model": best["model"], "model_path": best["model_path"],
        "primary_selection_metric": primary_key,
        "primary_metric_value": (best["validation_leakage_free"] or
                                 best["validation_full"])[primary_key],
        "feature_order_path": str(dataset_dir / "metadata" / "feature_dictionary.json"),
    })

    logger.info("BEST MODEL: %s (%s=%.4f on %s) | models_reported=%d/%d",
               best["model"], primary_key,
               (best["validation_leakage_free"] or best["validation_full"])[primary_key],
               config.PRIMARY_SELECTION_SPLIT, len(results), len(config.MODEL_FAMILIES))
    return comparison


def finalize_only() -> Dict[str, Any]:
    """Merge whatever model families are already complete on disk into the
    final comparison, without fitting anything. Used to recover a run that
    was interrupted (e.g. by an external session/process teardown) after
    some models finished but not all — never re-fits an already-successful
    model."""
    started = datetime.now(timezone.utc)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train", config.LOGS_DIR / f"finalize_{timestamp}.log")
    logger.info("Healix Disease Prediction — Phase 6.1 finalize-only (no fitting)")

    dataset_dir = config.resolve_dataset_dir()
    feature_order = load_feature_order(dataset_dir)
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)

    n_train_rows = pq.ParquetFile(
        dataset_dir / "datasets" / SPLIT_FILES["train"]).metadata.num_rows
    n_validation_rows = pq.ParquetFile(
        dataset_dir / "datasets" / SPLIT_FILES["validation"]).metadata.num_rows

    registry = model_registry()
    results: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name in config.MODEL_FAMILIES:
        prior = _load_existing_result(name, registry[name], n_train_rows, n_validation_rows)
        if prior is not None:
            results.append(prior)
            logger.info("found completed result for %s on disk", name)
        else:
            missing.append(name)
    if missing:
        logger.warning("no completed result found for: %s (excluded from the "
                       "comparison, NOT treated as a failure)", missing)

    return _finalize_comparison(results, dataset_dir, len(feature_order), encoder,
                                n_train_rows, n_validation_rows, started, logger)


def _write_comparison_csv(results: List[Dict[str, Any]], path) -> None:
    fields = ["model", "display_name", "train_seconds", "peak_memory_mb",
             "model_size_mb", "val_full_accuracy", "val_full_f1_macro",
             "val_full_f1_weighted", "val_clean_accuracy", "val_clean_f1_macro",
             "val_clean_f1_weighted", "val_clean_top3_accuracy",
             "val_clean_top5_accuracy", "predict_seconds_per_1000_rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            full = r["validation_full"]
            clean = r["validation_leakage_free"] or full
            writer.writerow({
                "model": r["model"], "display_name": r["display_name"],
                "train_seconds": r["train_seconds"],
                "peak_memory_mb": r["peak_memory_mb"],
                "model_size_mb": r["model_size_mb"],
                "val_full_accuracy": full["accuracy"],
                "val_full_f1_macro": full["f1_macro"],
                "val_full_f1_weighted": full["f1_weighted"],
                "val_clean_accuracy": clean["accuracy"],
                "val_clean_f1_macro": clean["f1_macro"],
                "val_clean_f1_weighted": clean["f1_weighted"],
                "val_clean_top3_accuracy": clean.get("top_3_accuracy"),
                "val_clean_top5_accuracy": clean.get("top_5_accuracy"),
                "predict_seconds_per_1000_rows":
                    full["timing"]["predict_seconds_per_1000_rows"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Healix Disease Prediction — baseline training")
    parser.add_argument("--models", nargs="*", default=None,
                        choices=list(config.MODEL_FAMILIES))
    parser.add_argument("--limit", type=int, default=None,
                        help="rows per split (smoke test)")
    parser.add_argument("--finalize-only", action="store_true",
                        help="merge already-completed model results on disk into the "
                             "final comparison; fits nothing")
    args = parser.parse_args()
    if args.finalize_only:
        result = finalize_only()
    else:
        result = run(model_names=args.models, limit=args.limit)
    print(json.dumps({"best_model": result["best_model"],
                      "n_models_reported": result["n_models_reported"],
                      "ranking": result["ranking"]}, indent=2))


if __name__ == "__main__":
    main()
