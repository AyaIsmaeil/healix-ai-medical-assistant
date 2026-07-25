"""
Healix - Phase 6.2 hyperparameter optimization (OFFLINE ONLY).

Stage 1 of the two-stage protocol documented in ``config_optimization.py``:
a randomized search per model family, on a fixed stratified subsample, scored
by macro-F1 under an inner StratifiedKFold. Randomized (not exhaustive grid)
search is used deliberately — with 4 hyperparameters per family, a full grid
would be 100-300 fits per family at ~1-5 min each, which does not fit the
compute budget; randomized search over the same space gives a comparable
best-found configuration for a fraction of the cost, and the sampled
configurations are logged so the search is fully auditable.

Run:
    python -m training.disease_prediction.optimize --families random_forest
    python -m training.disease_prediction.optimize            # all families
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from training.disease_prediction import config, config_optimization as optcfg
from training.disease_prediction.cross_validation import stratified_subsample
from training.disease_prediction.dataset import (DiseaseLabelEncoder,
                                                 load_bundle, load_feature_order)
from training.disease_prediction.utils import (PeakMemorySampler, Timer,
                                               set_global_seed, setup_logger,
                                               write_json)


# ----------------------------------------------------------------------
# Estimator construction from a sampled parameter dict
# ----------------------------------------------------------------------
def build_estimator(family: str, params: Dict[str, Any]):
    fixed = dict(optcfg.FIXED_PARAMS[family])
    merged = {**fixed, **params}

    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**merged)

    if family == "logistic_regression":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("classifier", LogisticRegression(**merged)),
        ])

    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**merged)

    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**merged)

    if family == "catboost":
        from catboost import CatBoostClassifier
        merged.setdefault("train_dir", str(config.LOGS_DIR / "catboost_info"))
        return CatBoostClassifier(**merged)

    raise ValueError(f"unknown family {family!r}")


def sample_configurations(space: Dict[str, List[Any]], n_iter: int,
                          seed: int) -> List[Dict[str, Any]]:
    """Deterministic random sample of distinct configurations."""
    rng = np.random.default_rng(seed)
    keys = list(space)
    seen, out = set(), []
    max_attempts = n_iter * 50
    for _ in range(max_attempts):
        if len(out) >= n_iter:
            break
        cfg = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        sig = json.dumps(cfg, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def evaluate_configuration(family: str, params: Dict[str, Any],
                           X: np.ndarray, y: np.ndarray, n_splits: int,
                           seed: int) -> Dict[str, Any]:
    """Inner cross-validated score for one configuration."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores, fit_times = [], []
    peak_mb = 0.0

    for train_idx, test_idx in skf.split(X, y):
        est = build_estimator(family, params)
        try:
            with PeakMemorySampler() as mem, Timer() as t:
                est.fit(X[train_idx], y[train_idx])
            pred = est.predict(X[test_idx])
        except Exception as exc:  # noqa: BLE001 - a bad config must not kill the search
            return {"params": params, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}
        scores.append(float(f1_score(y[test_idx], pred, average="macro",
                                     zero_division=0)))
        fit_times.append(t.elapsed)
        peak_mb = max(peak_mb, mem.peak_mb)
        del est

    arr = np.array(scores)
    return {"params": params, "status": "ok",
            "f1_macro_mean": round(float(arr.mean()), 6),
            "f1_macro_std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else 0.0,
            "fold_scores": [round(s, 6) for s in scores],
            "mean_fit_seconds": round(float(np.mean(fit_times)), 2),
            "peak_memory_mb": peak_mb}


def search_family(family: str, X: np.ndarray, y: np.ndarray,
                  logger) -> Dict[str, Any]:
    space = optcfg.SEARCH_SPACES[family]
    n_iter = optcfg.SEARCH_N_ITER[family]
    configs = sample_configurations(space, n_iter, config.SEED)
    logger.info("[%s] evaluating %d sampled configurations (inner %d-fold CV)",
                family, len(configs), optcfg.SEARCH_CV_FOLDS)

    trials: List[Dict[str, Any]] = []
    for i, params in enumerate(configs, start=1):
        rec = evaluate_configuration(family, params, X, y,
                                     optcfg.SEARCH_CV_FOLDS, config.SEED)
        trials.append(rec)
        if rec["status"] == "ok":
            logger.info("  [%s] %d/%d f1_macro=%.5f (+/-%.5f) fit=%.1fs %s",
                        family, i, len(configs), rec["f1_macro_mean"],
                        rec["f1_macro_std"], rec["mean_fit_seconds"], params)
        else:
            logger.warning("  [%s] %d/%d FAILED %s -> %s", family, i,
                           len(configs), params, rec["error"])

    ok = [t for t in trials if t["status"] == "ok"]
    if not ok:
        return {"family": family, "status": "all_failed", "trials": trials}

    best = max(ok, key=lambda t: t["f1_macro_mean"])
    baseline = config.MODEL_DEFAULTS.get(family, {})
    return {
        "family": family, "status": "ok",
        "n_configurations_evaluated": len(configs),
        "n_failed": len(trials) - len(ok),
        "search_space": {k: [str(v) for v in vals] for k, vals in space.items()},
        "phase61_baseline_params": baseline,
        "best_params": best["params"],
        "best_f1_macro_mean": best["f1_macro_mean"],
        "best_f1_macro_std": best["f1_macro_std"],
        "best_mean_fit_seconds": best["mean_fit_seconds"],
        "trials": sorted(ok, key=lambda t: -t["f1_macro_mean"]),
        "failed_trials": [t for t in trials if t["status"] != "ok"],
    }


def run(families: Optional[List[str]] = None) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    ts = started.strftime("%Y%m%dT%H%M%SZ")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("optimize", config.LOGS_DIR / f"optimize_{ts}.log")
    set_global_seed(config.SEED)

    dataset_dir = config.resolve_dataset_dir()
    feature_order = load_feature_order(dataset_dir)
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)
    logger.info("Phase 6.2 hyperparameter search | dataset=%s", dataset_dir)

    full = load_bundle(dataset_dir, "train", encoder, feature_order)
    idx = stratified_subsample(full.y, optcfg.SEARCH_SUBSAMPLE_ROWS, config.SEED)
    X, y = full.X[idx], full.y[idx]
    del full
    logger.info("search subsample: %d rows, %d classes present",
                len(y), len(np.unique(y)))

    selected = families or list(config.MODEL_FAMILIES)
    results: Dict[str, Any] = {}
    for family in selected:
        with Timer() as t:
            results[family] = search_family(family, X, y, logger)
        results[family]["search_wall_seconds"] = t.elapsed
        write_json(config.REPORTS_DIR / f"search_{family}.json", results[family])
        if results[family]["status"] == "ok":
            logger.info("[%s] BEST f1_macro=%.5f params=%s", family,
                        results[family]["best_f1_macro_mean"],
                        results[family]["best_params"])

    payload = {
        "generated_utc": started.isoformat(timespec="seconds"),
        "protocol": {
            "stage": "1 of 2 (randomized search on a stratified subsample)",
            "subsample_rows": optcfg.SEARCH_SUBSAMPLE_ROWS,
            "inner_cv_folds": optcfg.SEARCH_CV_FOLDS,
            "scoring": optcfg.SCORING,
            "seed": config.SEED,
            "note": ("Subsampled by design (see config_optimization.py). Stage 2 "
                     "re-verifies the winning config per family with 5-fold CV "
                     "on a larger sample."),
        },
        "families": results,
    }
    write_json(config.REPORTS_DIR / "hyperparameter_search.json", payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6.2 hyperparameter search")
    p.add_argument("--families", nargs="*", default=None,
                   choices=list(config.MODEL_FAMILIES))
    args = p.parse_args()
    out = run(families=args.families)
    summary = {f: {"best_f1_macro": r.get("best_f1_macro_mean"),
                   "best_params": r.get("best_params")}
               for f, r in out["families"].items()}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
