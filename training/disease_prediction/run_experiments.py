"""
Healix - Phase 6.2 experiment runner (OFFLINE ONLY).

Executes Stage 2 of the Phase-6.2 protocol and every downstream scientific
experiment, against models refitted from the winning hyperparameters found
by ``optimize.py`` (Stage 1).

Experiments, in order:
  1. Stage-2 5-fold Stratified CV per family (mean +/- std)  -> settles the
     Phase-6.1 "statistical tie" with measured variance.
  2. Calibration (ECE / MCE / Brier + reliability diagram).
  3. Explainability (impurity + permutation + SHAP, ontology-mapped).
  4. Robustness under 10/20/30/40% hidden symptoms, two masking semantics.
  5. Error analysis (confusion matrix, per-class, ontology-explained pairs).
  6. Clinical validation (Top-k, MRR, differential coverage).

Run:
    python -m training.disease_prediction.run_experiments --stage cv
    python -m training.disease_prediction.run_experiments --stage all
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq

from training.disease_prediction import config, config_optimization as optcfg
from training.disease_prediction import calibration as calib
from training.disease_prediction import clinical_validation as clinval
from training.disease_prediction import error_analysis as erra
from training.disease_prediction import explainability as expl
from training.disease_prediction import robustness as robust
from training.disease_prediction import statistical_tests as stats_tests
from training.disease_prediction.cross_validation import (cross_validate_model,
                                                          paired_fold_comparison,
                                                          stratified_subsample)
from training.disease_prediction.dataset import (DiseaseLabelEncoder,
                                                 SPLIT_FILES,
                                                 load_bundle, load_feature_order)
from training.disease_prediction.metrics import align_proba
from training.disease_prediction.optimize import build_estimator
from training.disease_prediction.utils import (Timer, set_global_seed,
                                               setup_logger, write_json)

DDXPLUS_DIR = config.REPO_ROOT / "app" / "data" / "raw" / "ddxplus"


# ----------------------------------------------------------------------
# Winning configuration resolution
# ----------------------------------------------------------------------
def load_best_params(family: str) -> Dict[str, Any]:
    """Winning params from Stage 1; falls back to the Phase-6.1 baseline if a
    family was never searched (recorded explicitly in the output)."""
    path = config.REPORTS_DIR / f"search_{family}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "ok" and data.get("best_params"):
            return dict(data["best_params"])
    return {}


def tuned_builder(family: str, params: Dict[str, Any]):
    def _build():
        return build_estimator(family, params)
    return _build


# ----------------------------------------------------------------------
# 1. Stage-2 cross validation
# ----------------------------------------------------------------------
def run_cross_validation(families: List[str], logger) -> Dict[str, Any]:
    dataset_dir = config.resolve_dataset_dir()
    feature_order = load_feature_order(dataset_dir)
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)

    full = load_bundle(dataset_dir, "train", encoder, feature_order)
    idx = stratified_subsample(full.y, optcfg.VERIFY_SUBSAMPLE_ROWS, config.SEED)
    X, y = full.X[idx], full.y[idx]
    del full
    logger.info("CV sample: %d rows, %d classes", len(y), len(np.unique(y)))

    results: Dict[str, Any] = {}
    for family in families:
        params = load_best_params(family)
        logger.info("[%s] 5-fold CV with params=%s", family, params or "(baseline)")
        try:
            cv = cross_validate_model(tuned_builder(family, params), X, y,
                                      n_splits=optcfg.CV_FOLDS,
                                      seed=config.SEED, logger=logger)
            cv["params_used"] = params
            cv["params_source"] = ("stage1_search" if params else
                                   "phase61_baseline_fallback")
            results[family] = cv
            logger.info("[%s] CV f1_macro=%.5f +/- %.5f | acc=%.5f +/- %.5f",
                        family, cv["f1_macro"]["mean"], cv["f1_macro"]["std"],
                        cv["accuracy"]["mean"], cv["accuracy"]["std"])
        except Exception as exc:  # noqa: BLE001 - one family must not kill the run
            logger.error("[%s] CV FAILED: %s", family, exc)
            results[family] = {"status": "failed", "error": str(exc)}

    ok = {f: r for f, r in results.items() if "f1_macro" in r}
    ranked = sorted(ok.items(), key=lambda kv: -kv[1]["f1_macro"]["mean"])

    comparisons = []
    if len(ranked) >= 2:
        for i in range(len(ranked) - 1):
            a_name, a_cv = ranked[i]
            b_name, b_cv = ranked[i + 1]
            comparisons.append(paired_fold_comparison(a_cv, b_cv, a_name, b_name))

    # Formal hypothesis testing across every metric, not just mean F1.
    statistics: Dict[str, Any] = {}
    for metric in ("f1_macro", "accuracy", "f1_weighted"):
        try:
            statistics[metric] = stats_tests.compare_all(ok, metric=metric,
                                                         top_n=3)
        except Exception as exc:  # noqa: BLE001 - stats must not kill the run
            statistics[metric] = {"error": f"{type(exc).__name__}: {exc}"}
            logger.warning("statistical comparison failed for %s: %s", metric, exc)

    primary = statistics.get("f1_macro", {})
    for cmp_ in primary.get("pairwise_comparisons_top_n", []):
        logger.info("STAT %s vs %s: diff=%+.6f  p_corrected=%.4f  %s",
                    cmp_["model_a"], cmp_["model_b"], cmp_["mean_difference"],
                    cmp_["nadeau_bengio_corrected_t_test"]["p_value"],
                    cmp_["verdict"])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "stage": "2 of 2 (verification)",
            "n_splits": optcfg.CV_FOLDS,
            "subsample_rows": optcfg.VERIFY_SUBSAMPLE_ROWS,
            "seed": config.SEED,
            "note": ("Folds are identical across families (same seed), so the "
                     "paired comparisons below are on matched splits."),
        },
        "results": results,
        "ranking": [{"rank": i + 1, "family": f,
                     "f1_macro_mean": r["f1_macro"]["mean"],
                     "f1_macro_std": r["f1_macro"]["std"]}
                    for i, (f, r) in enumerate(ranked)],
        "paired_comparisons": comparisons,
        "statistical_tests": statistics,
    }
    write_json(config.REPORTS_DIR / "cross_validation.json", payload)
    write_json(config.REPORTS_DIR / "statistical_comparison.json", statistics)
    return payload


# ----------------------------------------------------------------------
# helpers for the model-level experiments
# ----------------------------------------------------------------------
def _fit_final_candidate(family: str, dataset_dir: Path, feature_order,
                         encoder, logger, n_rows: Optional[int] = None):
    params = load_best_params(family)
    bundle = load_bundle(dataset_dir, "train", encoder, feature_order)
    if n_rows:
        idx = stratified_subsample(bundle.y, n_rows, config.SEED)
        X, y = bundle.X[idx], bundle.y[idx]
    else:
        X, y = bundle.X, bundle.y
    logger.info("[%s] fitting on %d rows params=%s", family, len(y), params)
    model = build_estimator(family, params)
    with Timer() as t:
        model.fit(X, y)
    logger.info("[%s] fitted in %.1fs", family, t.elapsed)
    return model, params, t.elapsed


def _proba(model, X, n_classes):
    p = model.predict_proba(X)
    if hasattr(model, "classes_"):
        p = align_proba(p, np.asarray(model.classes_), n_classes)
    return p


def _load_differential_ids(dataset_dir: Path, split: str,
                           clean_mask: np.ndarray) -> np.ndarray:
    """Read the list-typed ``y_differential_ids`` column for ``split``, in
    the same row order ``load_split``/``load_bundle`` use (a full,
    unfiltered read of the same parquet file), then apply the same leakage
    ``clean_mask`` so the result stays index-aligned with ``Xv``/``yv``.

    Loaded separately from ``load_bundle`` because ``dataset.py`` deliberately
    excludes this list-typed column from the default feature+target read
    (see ``dataset.py`` ``load_split`` docstring) — that exclusion is correct
    for training/CV, but clinical validation needs it to compare against
    DDXPlus's own reference differential.
    """
    path = dataset_dir / "datasets" / SPLIT_FILES[split]
    table = pq.read_table(path, columns=["y_differential_ids"])
    diff_all = table.to_pandas()["y_differential_ids"].to_numpy()
    return diff_all[clean_mask]


# ----------------------------------------------------------------------
# 2-6. Model-level experiments
# ----------------------------------------------------------------------
def run_model_experiments(families: List[str], logger,
                          fit_rows: Optional[int] = None) -> Dict[str, Any]:
    dataset_dir = config.resolve_dataset_dir()
    feature_order = load_feature_order(dataset_dir)
    encoder = DiseaseLabelEncoder.from_label_dictionary(dataset_dir)
    n_classes = encoder.n_classes
    resolver = expl.FeatureOntologyResolver(dataset_dir, DDXPLUS_DIR)
    explainer = erra.ConfusionExplainer(dataset_dir, DDXPLUS_DIR)

    val = load_bundle(dataset_dir, "validation", encoder, feature_order)
    clean = val.clean_mask
    Xv, yv = val.X[clean], val.y[clean]
    diff_v = _load_differential_ids(dataset_dir, "validation", clean)
    logger.info("validation (leakage-free): %d rows", len(yv))

    rng = np.random.default_rng(config.SEED)

    def sample(n):
        n = min(n, len(yv))
        sel = rng.choice(len(yv), size=n, replace=False)
        sel.sort()
        return Xv[sel], yv[sel]

    def sample_with_index(n):
        n = min(n, len(yv))
        sel = rng.choice(len(yv), size=n, replace=False)
        sel.sort()
        return Xv[sel], yv[sel], sel

    out: Dict[str, Any] = {}
    for family in families:
        logger.info("=== experiments for %s ===", family)
        model, params, fit_s = _fit_final_candidate(
            family, dataset_dir, feature_order, encoder, logger, fit_rows)
        fam: Dict[str, Any] = {"params_used": params, "fit_seconds": fit_s,
                               "fit_rows": fit_rows or "full_train"}

        # --- calibration ---
        Xc, yc, sel_c = sample_with_index(optcfg.CALIBRATION_SAMPLE_ROWS)
        pc = _proba(model, Xc, n_classes)
        cal = calib.calibration_report(pc, yc, n_classes,
                                       n_bins=optcfg.CALIBRATION_N_BINS)
        cal["plot"] = calib.plot_reliability_diagram(
            cal, f"Reliability — {family}",
            config.REPORTS_DIR / f"reliability_{family}.png")
        fam["calibration"] = cal
        logger.info("[%s] ECE=%.5f MCE=%.5f Brier=%.5f (%s)", family,
                    cal["ece"], cal["mce"], cal["brier_multiclass"],
                    cal["direction"])

        # --- clinical validation ---
        clin = clinval.clinical_metrics(pc, yc)
        clin["rank_plot"] = clinval.plot_rank_distribution(
            pc, yc, f"Rank of true disease — {family}",
            config.REPORTS_DIR / f"rank_distribution_{family}.png")
        clin["differential_agreement"] = clinval.differential_agreement(
            pc, yc, list(diff_v[sel_c]), encoder.classes_, k=5)
        fam["clinical_validation"] = clin
        logger.info("[%s] top1=%.5f top3=%.5f top5=%.5f MRR=%.5f "
                    "mean_differential_coverage_at_5=%s", family,
                    clin["top_1_accuracy"], clin["top_3_accuracy"],
                    clin["top_5_accuracy"], clin["mrr"],
                    clin["differential_agreement"]["mean_differential_coverage_at_k"])

        # --- error analysis ---
        pred_full = model.predict(Xv)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        np.add.at(cm, (yv, pred_full), 1)
        per_class = erra.per_class_metrics(yv, pred_full, encoder.classes_)
        sym_pairs = erra.symmetric_confusion_pairs(cm, encoder.classes_, top_n=12)
        for pair in sym_pairs:
            pair["ontology_explanation"] = explainer.explain_pair(
                pair["disease_a"], pair["disease_b"])
        fam["error_analysis"] = {
            "top_confusions_directional": erra.top_confusions(
                cm, encoder.classes_, top_n=20),
            "top_confusion_pairs_symmetric": sym_pairs,
            "per_class": per_class,
            "confusion_matrix_plot": erra.plot_confusion_matrix(
                cm, f"Confusion (validation, row-normalized) — {family}",
                config.REPORTS_DIR / f"cm_normalized_{family}.png"),
        }
        write_json(config.REPORTS_DIR / f"confusion_matrix_{family}_counts.json",
                   {"class_names": encoder.classes_, "matrix": cm.tolist()})

        # --- robustness ---
        Xr, yr = sample(optcfg.ROBUSTNESS_SAMPLE_ROWS)
        rob = robust.robustness_curve(model, Xr, yr, feature_order,
                                      rates=optcfg.ROBUSTNESS_MISSING_RATES,
                                      seed=config.SEED, logger=logger)
        rob["plot"] = robust.plot_robustness(
            rob, f"Robustness to hidden symptoms — {family}",
            config.REPORTS_DIR / f"robustness_{family}.png")
        fam["robustness"] = rob

        # --- explainability ---
        imp = expl.impurity_importance(model, feature_order,
                                       top_n=optcfg.TOP_FEATURES_TO_REPORT)
        Xp, yp = sample(optcfg.PERMUTATION_IMPORTANCE_ROWS)
        try:
            perm = expl.compute_permutation_importance(
                model, Xp, yp, feature_order,
                n_repeats=optcfg.PERMUTATION_REPEATS,
                top_n=optcfg.TOP_FEATURES_TO_REPORT, seed=config.SEED)
        except Exception as exc:  # noqa: BLE001
            perm = []
            logger.warning("[%s] permutation importance failed: %s", family, exc)

        Xb, _ = sample(optcfg.SHAP_BACKGROUND_ROWS)
        Xe, _ = sample(optcfg.SHAP_EXPLAIN_ROWS)
        shap_res = expl.compute_shap_importance(
            model, Xb, Xe, feature_order, top_n=optcfg.TOP_FEATURES_TO_REPORT)

        for entry in imp:
            entry.update(resolver.describe(entry["column"]))
        for entry in perm:
            entry.update(resolver.describe(entry["column"]))
        if shap_res.get("available"):
            for entry in shap_res["top_features"]:
                entry.update(resolver.describe(entry["column"]))

        fam["explainability"] = {
            "impurity_importance_top": imp,
            "permutation_importance_top": perm,
            "shap": shap_res,
            "plots": {
                "impurity": expl.plot_top_features(
                    imp, "importance", f"Impurity importance — {family}",
                    config.REPORTS_DIR / f"importance_impurity_{family}.png"),
                "permutation": expl.plot_top_features(
                    perm, "importance_mean",
                    f"Permutation importance (macro-F1 drop) — {family}",
                    config.REPORTS_DIR / f"importance_permutation_{family}.png")
                if perm else None,
            },
        }

        out[family] = fam
        write_json(config.REPORTS_DIR / f"experiments_{family}.json", fam)
        del model

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": out,
        "sampling": {
            "calibration_rows": optcfg.CALIBRATION_SAMPLE_ROWS,
            "robustness_rows": optcfg.ROBUSTNESS_SAMPLE_ROWS,
            "permutation_rows": optcfg.PERMUTATION_IMPORTANCE_ROWS,
            "shap_background": optcfg.SHAP_BACKGROUND_ROWS,
            "shap_explained": optcfg.SHAP_EXPLAIN_ROWS,
        },
    }
    write_json(config.REPORTS_DIR / "model_experiments.json", payload)
    return payload


def select_top_families(top_n: int, logger,
                        metric: str = "f1_macro") -> List[str]:
    """Rank families by cross-validated ``metric`` and keep the best ``top_n``.

    Screening step: the expensive per-model analyses (SHAP, permutation
    importance, robustness sweeps, calibration) cost far more than the CV that
    ranks them, so running them on families already shown to be inferior wastes
    compute without changing the decision. Every family still receives the full
    hyperparameter search AND full cross-validation, so the elimination is made
    on measured evidence, never on assumption -- and the eliminated families'
    CV numbers remain in the final comparison table.
    """
    path = config.REPORTS_DIR / "cross_validation.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run --stage cv before selecting top families.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    ranked = [r["family"] for r in payload.get("ranking", [])]
    if not ranked:
        raise RuntimeError("cross_validation.json contains no usable ranking")

    selected = ranked[:top_n]
    dropped = ranked[top_n:]
    logger.info("SCREENING by cross-validated %s: keeping %s", metric, selected)
    if dropped:
        logger.info("  eliminated from deep analysis (CV results retained): %s",
                    dropped)
    write_json(config.REPORTS_DIR / "screening_decision.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterion": f"cross-validated {metric} (5-fold, identical folds)",
        "full_ranking": payload.get("ranking", []),
        "advanced_to_deep_analysis": selected,
        "eliminated": dropped,
        "rationale": ("Deep analyses (calibration, SHAP, permutation, "
                      "robustness, clinical validation, error analysis) are "
                      "run only on the top candidates. Eliminated families "
                      "keep their hyperparameter-search and cross-validation "
                      "results in the final comparison."),
    })
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6.2 experiments")
    p.add_argument("--stage", default="all",
                   choices=["cv", "models", "all"])
    p.add_argument("--families", nargs="*", default=None,
                   choices=list(config.MODEL_FAMILIES))
    p.add_argument("--fit-rows", type=int, default=None,
                   help="fit candidates on a stratified subsample (default full train)")
    p.add_argument("--top-n", type=int, default=None,
                   help="after CV, run the deep analyses only on the best N families")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("experiments", config.LOGS_DIR / f"experiments_{ts}.log")
    set_global_seed(config.SEED)

    families = args.families or list(config.MODEL_FAMILIES)
    if args.stage in ("cv", "all"):
        run_cross_validation(families, logger)

    if args.stage in ("models", "all"):
        deep = families
        if args.top_n:
            deep = select_top_families(args.top_n, logger)
        run_model_experiments(deep, logger, fit_rows=args.fit_rows)
    logger.info("experiments complete")


if __name__ == "__main__":
    main()
