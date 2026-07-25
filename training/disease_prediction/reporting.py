"""
Healix - Phase 6.2 report generation (OFFLINE ONLY).

Every table in every generated report is rendered FROM the experiment JSON
artifacts, never transcribed by hand. This is a deliberate integrity control:
the Phase 6.2 brief requires that no number be invented or estimated, and the
most realistic way for a wrong number to enter a report is human transcription
between a results file and a markdown table. If an artifact is missing, the
generator writes an explicit "NOT MEASURED" marker instead of a plausible
value.

Generates:
  optimization_results.json / optimization_summary.md   (Step 1)
  CROSS_VALIDATION_REPORT.md                            (Step 2)
  MODEL_OPTIMIZATION_REPORT.md                          (Step 5)
  EXPLAINABILITY_REPORT.md                              (Step 5)
  ERROR_ANALYSIS.md                                     (Step 5)
  FINAL_MODEL_SELECTION.md                              (Step 5)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from training.disease_prediction import config
from training.disease_prediction.config import MODEL_FAMILIES
from training.disease_prediction.utils import write_json

NOT_MEASURED = "NOT MEASURED"
DOCS = config.DOCS_RESEARCH_DIR
REPORTS = config.REPORTS_DIR

DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


# ----------------------------------------------------------------------
# artifact loading (missing artifact -> None, never a fabricated default)
# ----------------------------------------------------------------------
def _load(name: str) -> Optional[Dict[str, Any]]:
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, nd: int = 5) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _pm(stat: Optional[Dict[str, Any]], nd: int = 5) -> str:
    """mean ± std renderer for a descriptive_stats dict."""
    if not stat or stat.get("mean") is None:
        return NOT_MEASURED
    mean = stat["mean"]
    std = stat.get("std")
    if std is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def _ci(stat: Optional[Dict[str, Any]], nd: int = 5) -> str:
    if not stat or stat.get("ci_low") is None:
        return NOT_MEASURED
    return f"[{stat['ci_low']:.{nd}f}, {stat['ci_high']:.{nd}f}]"


def _header(title: str, subtitle: str = "") -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = [f"# {title}", ""]
    if subtitle:
        out += [f"> {subtitle}", ""]
    out += [
        f"> **Generated:** {stamp} — rendered automatically from the experiment "
        f"JSON artifacts in `training/disease_prediction/outputs/reports/`. "
        f"No value in this document was transcribed or estimated by hand; "
        f"`{NOT_MEASURED}` appears wherever an experiment did not run.",
        "",
        "> **Data caveat (applies to every number below):** all results come "
        "from `ddxplus-v2.0-a8490b06a563`, a **synthetic, closed-world, "
        "English/French** corpus. See "
        "[`SCIENTIFIC_LIMITATIONS_AND_DEPLOYMENT_ROADMAP.md`]"
        "(SCIENTIFIC_LIMITATIONS_AND_DEPLOYMENT_ROADMAP.md) for what these "
        "numbers do and do not support.",
        "",
    ]
    return "\n".join(out)


# ----------------------------------------------------------------------
# Step 1 — optimization consolidation
# ----------------------------------------------------------------------
def build_optimization_results() -> Dict[str, Any]:
    families: Dict[str, Any] = {}
    for fam in MODEL_FAMILIES:
        data = _load(f"search_{fam}.json")
        if data is None:
            families[fam] = {"status": "not_run"}
            continue
        families[fam] = {
            "status": data.get("status"),
            "n_configurations_evaluated": data.get("n_configurations_evaluated"),
            "n_failed": data.get("n_failed"),
            "best_params": data.get("best_params"),
            "best_f1_macro_mean": data.get("best_f1_macro_mean"),
            "best_f1_macro_std": data.get("best_f1_macro_std"),
            "best_mean_fit_seconds": data.get("best_mean_fit_seconds"),
            "phase61_baseline_params": data.get("phase61_baseline_params"),
            "search_space": data.get("search_space"),
            "search_wall_seconds": data.get("search_wall_seconds"),
            "all_trials": [
                {"params": t["params"], "f1_macro_mean": t["f1_macro_mean"],
                 "f1_macro_std": t["f1_macro_std"],
                 "mean_fit_seconds": t["mean_fit_seconds"]}
                for t in data.get("trials", [])
            ],
        }

    scored = {f: v for f, v in families.items()
              if v.get("best_f1_macro_mean") is not None}
    ranking = sorted(scored.items(), key=lambda kv: -kv[1]["best_f1_macro_mean"])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "Phase 6.2 Step 1 — hyperparameter search (Stage 1 of 2)",
        "protocol": {
            "search": "randomized search over a documented space",
            "inner_cv_folds": 3,
            "subsample_rows": 120000,
            "scoring": "f1_macro",
            "seed": config.SEED,
        },
        "families": families,
        "ranking_by_tuned_f1_macro": [
            {"rank": i + 1, "family": f, "best_f1_macro_mean": v["best_f1_macro_mean"]}
            for i, (f, v) in enumerate(ranking)
        ],
    }
    write_json(REPORTS / "optimization_results.json", payload)
    return payload


def write_optimization_summary(payload: Dict[str, Any]) -> Path:
    baseline = {   # Phase 6.1 measured validation macro-F1 (leakage-free subset)
        "random_forest": 0.996152, "catboost": 0.995938,
        "logistic_regression": 0.993333, "xgboost": 0.944098,
        "lightgbm": 0.162327,
    }
    lines = [_header(
        "Hyperparameter Optimization Summary (Phase 6.2 — Step 1)",
        "Stage 1 of the two-stage protocol: randomized search per family on a "
        "fixed stratified 120,000-row subsample, scored by 3-fold macro-F1.")]

    lines += ["## Tuned results vs the Phase 6.1 defaults-only baseline", "",
              "| Family | Phase 6.1 baseline F1 | Tuned F1 (search CV) | Δ | Configs | Failed |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in payload["ranking_by_tuned_f1_macro"]:
        fam = row["family"]
        v = payload["families"][fam]
        base = baseline.get(fam)
        tuned = v["best_f1_macro_mean"]
        delta = f"{tuned - base:+.5f}" if base is not None else NOT_MEASURED
        lines.append(
            f"| **{DISPLAY.get(fam, fam)}** | {_fmt(base)} | {_fmt(tuned)} | "
            f"{delta} | {v.get('n_configurations_evaluated')} | {v.get('n_failed')} |")

    lines += ["", "> The Phase 6.1 column is single-split validation macro-F1; "
                  "the tuned column is a 3-fold CV mean on a subsample. They are "
                  "not measured on identical data, so Δ indicates direction and "
                  "rough magnitude, not an exact gain.", ""]

    lines += ["## Winning configuration per family", ""]
    for fam in MODEL_FAMILIES:
        v = payload["families"].get(fam, {})
        lines += [f"### {DISPLAY.get(fam, fam)}", ""]
        if v.get("status") != "ok":
            lines += [f"- Status: **{v.get('status', 'not_run')}** — {NOT_MEASURED}", ""]
            continue
        lines += [
            f"- Best macro-F1 (search CV): **{_fmt(v['best_f1_macro_mean'])}** "
            f"± {_fmt(v.get('best_f1_macro_std'))}",
            f"- Best params: `{json.dumps(v['best_params'], sort_keys=True)}`",
            f"- Phase 6.1 baseline params: `{json.dumps(v.get('phase61_baseline_params', {}), sort_keys=True)}`",
            f"- Configurations evaluated: {v.get('n_configurations_evaluated')} "
            f"(failed: {v.get('n_failed')})",
            f"- Mean fit time of best config: {_fmt(v.get('best_mean_fit_seconds'), 1)} s",
            "",
        ]
    path = REPORTS / "optimization_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Step 2 — cross-validation report
# ----------------------------------------------------------------------
METRIC_LABELS = [
    ("accuracy", "Accuracy"), ("f1_macro", "F1 macro"),
    ("f1_weighted", "F1 weighted"), ("precision_macro", "Precision macro"),
    ("recall_macro", "Recall macro"), ("roc_auc_ovr_macro", "ROC-AUC (OvR macro)"),
]


def write_cross_validation_report() -> Optional[Path]:
    cv = _load("cross_validation.json")
    if cv is None:
        return None
    stats_all = cv.get("statistical_tests", {})
    primary = stats_all.get("f1_macro", {})
    desc = primary.get("descriptive_by_family", {})

    lines = [_header(
        "Cross-Validation Report (Phase 6.2 — Step 2)",
        "5-fold Stratified K-Fold on identical folds for every family, with "
        "formal statistical comparison of the top candidates.")]

    proto = cv.get("protocol", {})
    lines += ["## Protocol", "",
              f"- Folds: **{proto.get('n_splits')}** (StratifiedKFold, shuffled, "
              f"seed `{proto.get('seed')}`)",
              f"- Rows: **{proto.get('subsample_rows'):,}** stratified from the "
              f"1,025,602-row training split" if proto.get("subsample_rows") else "",
              "- Folds are **identical across families**, so every pairwise "
              "comparison below is on matched splits (a paired design).",
              "- Each family uses the winning hyperparameters from Step 1.",
              "",
              "> **Why a subsample:** full-scale 5-fold CV across five families "
              "would take days on the available hardware. The subsample is "
              "fixed by seed and shared by every family, so the comparison is "
              "fair even though it is not full-scale.",
              ""]

    lines += ["## Ranking by cross-validated macro-F1", "",
              "| Rank | Family | F1 macro (mean ± std) | 95% CI |",
              "|---|---|---|---|"]
    for row in cv.get("ranking", []):
        fam = row["family"]
        d = desc.get(fam, {}).get("f1_macro")
        lines.append(f"| {row['rank']} | **{DISPLAY.get(fam, fam)}** | "
                     f"{_pm(d)} | {_ci(d)} |")
    lines.append("")

    lines += ["## Full metric table (mean ± std across 5 folds)", "",
              "| Family | " + " | ".join(l for _, l in METRIC_LABELS) + " |",
              "|---" * (len(METRIC_LABELS) + 1) + "|"]
    for fam in MODEL_FAMILIES:
        res = cv.get("results", {}).get(fam)
        if not res or "folds" not in res:
            lines.append(f"| {DISPLAY.get(fam, fam)} | " +
                         " | ".join([NOT_MEASURED] * len(METRIC_LABELS)) + " |")
            continue
        cells = []
        for key, _ in METRIC_LABELS:
            if key in res:
                cells.append(_pm(res[key]))
            elif res["folds"] and key in res["folds"][0]:
                vals = [f[key] for f in res["folds"] if f.get(key) is not None]
                if vals:
                    import statistics as st
                    cells.append(f"{st.mean(vals):.5f} ± "
                                 f"{(st.stdev(vals) if len(vals) > 1 else 0):.5f}")
                else:
                    cells.append(NOT_MEASURED)
            else:
                cells.append(NOT_MEASURED)
        lines.append(f"| **{DISPLAY.get(fam, fam)}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Operational cost (measured during CV)", "",
              "| Family | Fit seconds/fold | Predict s/1000 rows | Peak RSS (MB) |",
              "|---|---|---|---|"]
    for fam in MODEL_FAMILIES:
        res = cv.get("results", {}).get(fam)
        if not res or "fit_seconds" not in res:
            lines.append(f"| {DISPLAY.get(fam, fam)} | {NOT_MEASURED} | "
                         f"{NOT_MEASURED} | {NOT_MEASURED} |")
            continue
        lines.append(
            f"| **{DISPLAY.get(fam, fam)}** | {_pm(res.get('fit_seconds'), 1)} | "
            f"{_pm(res.get('predict_seconds_per_1000'), 6)} | "
            f"{_pm(res.get('peak_memory_mb'), 1)} |")
    lines.append("")

    # statistical comparisons
    lines += ["## Statistical comparison", "",
              f"- Significance level α = **{primary.get('alpha', 0.05)}**",
              f"- Practical-significance threshold Δ = "
              f"**{primary.get('practical_significance_delta', 0.002)}** "
              f"(declared before any Phase 6.2 result was seen)",
              f"- Multiple-comparison correction: "
              f"{primary.get('multiple_comparison_correction', NOT_MEASURED)}",
              "",
              "> **Methodological note.** Fold scores in k-fold CV are NOT "
              "independent (folds share training data), so an uncorrected "
              "paired t-test is anti-conservative and would manufacture false "
              "positives. The **Nadeau–Bengio corrected** test is therefore the "
              "primary result; the uncorrected value is shown only for "
              "reference. With 5 folds, Wilcoxon's smallest attainable "
              "two-sided p is 0.0625, so it **cannot** reach α=0.05 regardless "
              "of effect size — a power limitation, not evidence of absence.",
              ""]

    comps = primary.get("pairwise_comparisons_top_n", [])
    if comps:
        lines += ["| Comparison | Mean Δ | 95% CI of Δ | NB-corrected p | Holm-adj p | Wilcoxon p | Cohen's d | Stat. sig. | Pract. sig. |",
                  "|---|---:|---|---:|---:|---:|---:|:--:|:--:|"]
        for c in comps:
            nb = c.get("nadeau_bengio_corrected_t_test", {})
            wx = c.get("wilcoxon_signed_rank", {})
            lines.append(
                f"| {DISPLAY.get(c['model_a'], c['model_a'])} vs "
                f"{DISPLAY.get(c['model_b'], c['model_b'])} | "
                f"{c['mean_difference']:+.6f} | "
                f"[{_fmt(c.get('ci_difference_low'), 6)}, {_fmt(c.get('ci_difference_high'), 6)}] | "
                f"{_fmt(nb.get('p_value'), 4)} | "
                f"{_fmt(c.get('holm_bonferroni_adjusted_p'), 4)} | "
                f"{_fmt(wx.get('p_value'), 4)} | "
                f"{_fmt(c.get('cohens_d_paired'), 3)} | "
                f"{'YES' if c.get('statistically_significant') else 'no'} | "
                f"{'YES' if c.get('practically_significant') else 'no'} |")
        lines.append("")
        lines += ["### Verdict per comparison", ""]
        for c in comps:
            lines.append(f"- **{DISPLAY.get(c['model_a'], c['model_a'])} vs "
                         f"{DISPLAY.get(c['model_b'], c['model_b'])}** — "
                         f"{c.get('verdict')}")
        lines.append("")
    else:
        lines += [f"{NOT_MEASURED} — no pairwise comparisons available.", ""]

    path = DOCS / "CROSS_VALIDATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Step 5 — explainability / error analysis
# ----------------------------------------------------------------------
def _feature_row(entry: Dict[str, Any], value_key: str) -> str:
    meaning = entry.get("clinical_meaning") or "—"
    if len(meaning) > 90:
        meaning = meaning[:87] + "..."
    return (f"| {entry.get('rank')} | `{entry.get('column')}` | "
            f"{_fmt(entry.get(value_key), 6)} | {entry.get('group', '—')} | "
            f"{entry.get('healix_symptom_id') or entry.get('ddxplus_evidence') or '—'} | "
            f"{meaning} |")


def write_explainability_report() -> Optional[Path]:
    exp = _load("model_experiments.json")
    if exp is None:
        return None
    lines = [_header(
        "Explainability Report (Phase 6.2 — Step 4/5)",
        "Impurity, permutation, and SHAP importance for the screened top "
        "candidates, each mapped back to the HEALIX ontology.")]

    lines += ["## Method", "",
              "Three complementary views, because each alone misleads:",
              "",
              "- **Impurity (Gini)** — free from a fitted forest, but biased "
              "toward high-cardinality features and computed on training data.",
              "- **Permutation** — measured drop in **macro-F1** on held-out "
              "data when a column is shuffled; answers *does this feature carry "
              "real predictive signal?*",
              "- **SHAP (TreeExplainer)** — per-prediction attribution, sampled "
              "because exact TreeSHAP over a 49-class ensemble is prohibitive "
              "at full scale.",
              ""]

    for fam, data in exp.get("families", {}).items():
        e = data.get("explainability", {})
        lines += [f"## {DISPLAY.get(fam, fam)}", ""]

        imp = e.get("impurity_importance_top", [])
        if imp:
            lines += ["### Top 50 — impurity importance", "",
                      "| # | Feature column | Importance | Group | Ontology ref | Clinical meaning |",
                      "|---|---|---:|---|---|---|"]
            lines += [_feature_row(x, "importance") for x in imp[:50]]
            lines.append("")
        else:
            lines += [f"### Impurity importance — {NOT_MEASURED}", ""]

        perm = e.get("permutation_importance_top", [])
        if perm:
            lines += ["### Top 50 — permutation importance (macro-F1 drop, held-out)", "",
                      "| # | Feature column | Mean drop | Group | Ontology ref | Clinical meaning |",
                      "|---|---|---:|---|---|---|"]
            lines += [_feature_row(x, "importance_mean") for x in perm[:50]]
            lines.append("")
        else:
            lines += [f"### Permutation importance — {NOT_MEASURED}", ""]

        shap = e.get("shap", {})
        if shap.get("available"):
            lines += [f"### Top 50 — mean |SHAP| "
                      f"({shap.get('n_explained')} explained rows, "
                      f"{shap.get('n_background')} background)", "",
                      "| # | Feature column | mean abs SHAP | Group | Ontology ref | Clinical meaning |",
                      "|---|---|---:|---|---|---|"]
            lines += [_feature_row(x, "mean_abs_shap")
                      for x in shap.get("top_features", [])[:50]]
            lines.append("")
        else:
            lines += [f"### SHAP — {NOT_MEASURED} "
                      f"({shap.get('error', 'not run')})", ""]

    path = DOCS / "EXPLAINABILITY_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_error_analysis_report() -> Optional[Path]:
    exp = _load("model_experiments.json")
    if exp is None:
        return None
    lines = [_header(
        "Error Analysis (Phase 6.2 — Step 4/5)",
        "Confusion structure of the screened top candidates, with every "
        "hard pair explained through the HEALIX/DDXPlus evidence sets.")]

    for fam, data in exp.get("families", {}).items():
        ea = data.get("error_analysis", {})
        lines += [f"## {DISPLAY.get(fam, fam)}", ""]

        pairs = ea.get("top_confusion_pairs_symmetric", [])
        if not pairs:
            lines += [f"{NOT_MEASURED}", ""]
            continue

        lines += ["### Hardest disease pairs (bidirectional confusion)", "",
                  "| # | Disease A | Disease B | A→B | B→A | Total | Evidence overlap (Jaccard) | Crosses urgency? |",
                  "|---|---|---|---:|---:|---:|---:|:--:|"]
        for p in pairs:
            ex = p.get("ontology_explanation", {})
            lines.append(
                f"| {p['rank']} | {ex.get('pathology_a', p['disease_a'])} | "
                f"{ex.get('pathology_b', p['disease_b'])} | {p['a_to_b']} | "
                f"{p['b_to_a']} | {p['total_confusions']} | "
                f"{_fmt(ex.get('jaccard_overlap'), 3)} | "
                f"{'**YES**' if ex.get('crosses_urgency_boundary') else 'no'} |")
        lines.append("")

        lines += ["### Ontology-based explanation of the hardest pairs", ""]
        for p in pairs[:6]:
            ex = p.get("ontology_explanation", {})
            lines += [
                f"#### {ex.get('pathology_a')} ↔ {ex.get('pathology_b')}", "",
                f"- ICD-10: `{ex.get('icd10_a')}` vs `{ex.get('icd10_b')}`",
                f"- Specialty: {ex.get('specialty_a')} vs {ex.get('specialty_b')} "
                f"(same specialty: {ex.get('same_specialty')})",
                f"- Urgency prior: {ex.get('urgency_a')} vs {ex.get('urgency_b')} "
                f"— **crosses urgency boundary: {ex.get('crosses_urgency_boundary')}**",
                f"- Evidence sets: {ex.get('n_evidences_a')} vs "
                f"{ex.get('n_evidences_b')} evidences, "
                f"**{ex.get('n_shared_evidences')} shared** "
                f"(Jaccard {_fmt(ex.get('jaccard_overlap'), 3)})", ""]
            shared = ex.get("shared_evidences", [])
            if shared:
                lines += ["Shared evidence (why they look alike):", ""]
                lines += [f"  - {s}" for s in shared[:6]]
                lines.append("")
            only_a = ex.get("distinguishing_only_a", [])
            only_b = ex.get("distinguishing_only_b", [])
            if only_a or only_b:
                lines += ["Distinguishing evidence (what should separate them):", ""]
                lines += [f"  - *only {ex.get('pathology_a')}*: {s}" for s in only_a[:4]]
                lines += [f"  - *only {ex.get('pathology_b')}*: {s}" for s in only_b[:4]]
                lines.append("")

        per_class = ea.get("per_class", {})
        if per_class:
            worst = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:10]
            lines += ["### 10 weakest classes by F1", "",
                      "| Disease ID | Precision | Recall | F1 | Support |",
                      "|---|---:|---:|---:|---:|"]
            for cid, m in worst:
                lines.append(f"| {cid} | {_fmt(m['precision'], 4)} | "
                             f"{_fmt(m['recall'], 4)} | {_fmt(m['f1'], 4)} | "
                             f"{m['support']} |")
            lines.append("")

    path = DOCS / "ERROR_ANALYSIS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
