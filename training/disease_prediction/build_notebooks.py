"""
Healix - notebook builder (tooling, not an experiment).

Programmatically constructs the 12 Phase-6 research notebooks under
``notebooks/`` with nbformat, so every notebook has a consistent structure
(Objective -> Methodology -> Code -> Output -> Conclusions) and every cell is
generated from a single source instead of hand-copied 12 times.

This script is a ONE-TIME repo-restructuring tool, not part of the
experiment pipeline -- it is what Phase 6's methodology change (standalone
scripts -> notebooks) actually consists of. It does not train anything; it
only builds notebook *structure* with code cells that call the existing
training/disease_prediction modules. Whether each notebook's code cells have
already been executed (i.e. carry real output) depends on whether the
underlying experiment has actually been run -- see NOTEBOOK_ROADMAP.md for
the per-notebook status. Cells for experiments that have not run are left
un-executed (no fabricated output), never populated with invented numbers.

Run:
    python -m training.disease_prediction.build_notebooks
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def md(text: str) -> "nbf.NotebookNode":
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> "nbf.NotebookNode":
    return nbf.v4.new_code_cell(text.strip() + "\n")


def new_notebook(cells: List["nbf.NotebookNode"]) -> "nbf.NotebookNode":
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (Healix venv)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def write_notebook(name: str, cells: List["nbf.NotebookNode"]) -> Path:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(cells)
    path = NOTEBOOKS_DIR / f"{name}.ipynb"
    nbf.write(nb, path)
    return path


BOOTSTRAP_CELL = """\
from training.disease_prediction.notebook_utils import bootstrap
paths = bootstrap({name!r})
"""

TOC = """\
# Healix Disease Prediction — Research Notebook Series

| # | Notebook | Phase |
|---|---|---|
| 01 | Dataset Inspection | 4 |
| 02 | Dataset Validation | 5 |
| 03 | Baseline Models | 6.1 |
| 04 | Hyperparameter Optimization | 6.2 |
| 05 | Cross-Validation | 6.2 |
| 06 | Statistical Tests | 6.2 |
| 07 | Probability Calibration | 6.2 |
| 08 | Explainability (SHAP) | 6.2 |
| 09 | Robustness Analysis | 6.2 |
| 10 | Error Analysis | 6.2 |
| 11 | Clinical Validation | 6.2 |
| 12 | Final Model Selection | 6.2 |

See `NOTEBOOK_ROADMAP.md` (repo root of `notebooks/`) for exact completion
status, execution order, and estimated runtime of each notebook.
"""


# ======================================================================
# 01 — Dataset Inspection  (Phase 4 — COMPLETE)
# ======================================================================
def build_01() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 01 — Dataset Inspection

**Phase:** 4 (Research & Ontology) · **Status:** ✅ COMPLETE

## Objective
Inspect the raw DDXPlus corpus and the frozen HEALIX ontology it is mapped
onto, before any feature engineering happens.

## Scientific methodology
- Read the two DDXPlus release dictionaries (`release_evidences.json`,
  `release_conditions.json`) directly — these fully define the label space
  (49 diseases, ICD-10-anchored) and the evidence space (223 evidences: 208
  binary, 10 categorical, 5 multi-valued; 110 symptoms + 113 antecedents).
- Reconstruct the ontology mapping via `app.ml.ontology_mapper.OntologyMapper`
  — the SAME class the Phase-5 Dataset Builder uses, so this notebook can
  never silently drift from what actually built the training data.
- Row-level statistics (1,292,579 patients across train/validate/test,
  class imbalance, evidence-per-patient distribution) are **not** re-scanned
  here — scanning the 670 MB `train.csv` was already done exhaustively in
  Phase 4.2 and is fully documented with exact figures in
  `docs/research/DDXPLUS_EDA.md`. Re-reading it here would cost minutes for
  numbers already on record; instead we load it as a reference.

## What this notebook does NOT do
Retrain or re-derive the ontology — it only *inspects* the frozen
`healix-ontology-v1.0.0` artifacts already used to build the training set.
"""),
        code(BOOTSTRAP_CELL.format(name="01_dataset_inspection")),
        md("## 1. DDXPlus release dictionaries"),
        code("""
import json

ddxplus_dir = paths.repo_root / "app" / "data" / "raw" / "ddxplus"
evidences = json.loads((ddxplus_dir / "release_evidences.json").read_text(encoding="utf-8"))
conditions = json.loads((ddxplus_dir / "release_conditions.json").read_text(encoding="utf-8"))

print(f"Evidences : {len(evidences)}")
print(f"Conditions: {len(conditions)}")
"""),
        md("## 2. Evidence-space breakdown (binary / categorical / multi-valued, symptom / antecedent)"),
        code("""
import pandas as pd
from collections import Counter

dtype_counts = Counter(e["data_type"] for e in evidences.values())
role_counts = Counter(("antecedent" if e.get("is_antecedent") else "symptom") for e in evidences.values())

display(pd.Series(dtype_counts, name="count").rename_axis("data_type").to_frame())
display(pd.Series(role_counts, name="count").rename_axis("role").to_frame())
"""),
        md("## 3. Disease (condition) table — ICD-10 coverage and severity distribution"),
        code("""
cond_df = pd.DataFrame([
    {"pathology": name, "icd10": c.get("icd10-id"), "severity": c.get("severity"),
     "n_symptoms": len(c.get("symptoms", {})), "n_antecedents": len(c.get("antecedents", {}))}
    for name, c in conditions.items()
]).sort_values("severity")

icd10_coverage = cond_df["icd10"].notna().mean() * 100
print(f"ICD-10 coverage: {icd10_coverage:.1f}% ({cond_df['icd10'].notna().sum()}/{len(cond_df)})")
print()
print("Severity distribution (1 = most urgent, DDXPlus scale is INVERTED):")
display(cond_df["severity"].value_counts().sort_index().to_frame("n_conditions"))
cond_df.head(10)
"""),
        md("## 4. HEALIX ontology mapping (the exact mapper used by the Phase-5 Dataset Builder)"),
        code("""
from app.ml.ontology_mapper import OntologyMapper

mapper = OntologyMapper(evidences, conditions)
summary = mapper.summary()
pd.Series(summary, name="count").rename_axis("ontology_component").to_frame()
"""),
        md("""
## Conclusions

- The ontology mapper reconstructs **exactly** the counts frozen in
  `healix-ontology-v1.0.0` (98 symptom IDs — 96 DDXPlus-derived + 2
  HEALIX-only concepts with no DDXPlus evidence; 53 chronic-disease codes;
  49 conditions, 100% ICD-10-covered).
- Severity is confirmed inverted (1 = most urgent) at the source — this is
  the exact detail that had to be handled correctly when deriving the
  `y_urgency_prior` label in the Dataset Builder (Phase 5).
- **Row-level corpus statistics** (1,292,579 patients, class imbalance
  ratios, evidence-per-patient distributions) are documented exhaustively
  in `docs/research/DDXPLUS_EDA.md` — this notebook intentionally does not
  duplicate that multi-minute full-corpus scan.

**Next notebook:** `02_dataset_validation.ipynb` — quality gates and
cross-split leakage measurement on the *built* Feature-Schema-v2 dataset.
"""),
    ]


# ======================================================================
# 02 — Dataset Validation  (Phase 5 — COMPLETE)
# ======================================================================
def build_02() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 02 — Dataset Validation

**Phase:** 5 (Dataset Builder) · **Status:** ✅ COMPLETE

## Objective
Validate the quality of the ML-ready dataset the Phase-5 Dataset Builder
produced from DDXPlus: schema conformance, missing-value structure, class
balance, and — most importantly — **cross-split leakage**.

## Scientific methodology
- Load the Dataset Builder's own machine-readable manifest, metadata, and
  quality-report artifacts (`manifest.json`, `metadata.json`,
  `quality_report.json`, `statistics.json`) rather than recomputing them —
  these were produced by a full streaming pass over all 1,292,579 rows
  (`app/ml/dataset_builder.py`) and are the authoritative source.
- Cross-split leakage was measured with `blake2b-128` hashing over
  `AGE|SEX|EVIDENCES|INITIAL_EVIDENCE` for every row in every split
  (`docs/research/CROSS_SPLIT_LEAKAGE_REPORT.md`); leaked rows are
  **flagged, never dropped** (`leakage_flag` column), and every downstream
  metric in this project is reported on the leakage-free subset as primary.
"""),
        code(BOOTSTRAP_CELL.format(name="02_dataset_validation")),
        code("""
import json
from training.disease_prediction import config

dataset_dir = config.resolve_dataset_dir()
metadata = json.loads((dataset_dir / "metadata" / "metadata.json").read_text(encoding="utf-8"))
manifest = json.loads((dataset_dir / "manifests" / "manifest.json").read_text(encoding="utf-8"))
quality = json.loads((dataset_dir / "statistics" / "quality_report.json").read_text(encoding="utf-8"))

print(f"dataset_version        = {metadata['dataset_version']}")
print(f"feature_schema_version = {metadata['feature_schema_version']}")
print(f"ontology_version       = {metadata['ontology_version']}")
print(f"n_features             = {metadata['n_features']}")
print(f"splits                 = {metadata['splits']}")
"""),
        md("## 1. Quality gates per split"),
        code("""
import pandas as pd

pd.DataFrame(quality).T[[
    "rows", "unique_records", "duplicate_groups", "redundant_rows", "duplicate_pct",
    "leaked_rows", "leaked_pct", "quarantined_rows", "missing_mandatory",
    "ambiguous_input_groups",
]]
"""),
        md("""
**Reading this table:** `quarantined_rows`, `missing_mandatory`, and
`ambiguous_input_groups` are 0 for every split — the builder never had to
discard a row for structural reasons. `leaked_pct` (~0.45–1.58%) is the
cross-split contamination that is *flagged, not removed* — every metric
reported anywhere in Phase 6 is computed on the leakage-free subset as the
primary number, exactly to neutralise this.
"""),
        md("## 2. Cross-split leakage detail (the mandatory Step-1 gate from Phase 5)"),
        code("""
csl = manifest.get("cross_split_leakage", {})
pd.Series(csl.get("pairwise_shared_unique", {}), name="shared_unique_records").to_frame()
"""),
        md("## 3. Feature-schema composition (why 286, not 9)"),
        code("""
pd.Series(metadata["ontology_summary"], name="count").rename_axis("component").to_frame()
"""),
        md("""
## Conclusions

- The dataset satisfies every structural quality gate the Dataset Builder
  enforces: zero quarantined rows, zero missing-mandatory rows, zero
  ambiguous input groups, across all three splits.
- Cross-split leakage is real (max ~1.58% in `test`) but small and, more
  importantly, **measured and flagged** rather than hidden — this is the
  single most important caveat carried into every later notebook's
  headline numbers.
- `absence_semantics: "closed_world"` (see `metadata.json`) is the other
  standing caveat: an unlisted DDXPlus evidence means "confirmed absent",
  not "never asked" — this is a synthetic-data property that will NOT hold
  for real, open-world Healix conversations (see
  `docs/research/SCIENTIFIC_LIMITATIONS_AND_DEPLOYMENT_ROADMAP.md`).

**Next notebook:** `03_baseline_models.ipynb` — five model families trained
with untuned, documented-default hyperparameters.
"""),
    ]


# ======================================================================
# 03 — Baseline Models  (Phase 6.1 — COMPLETE)
# ======================================================================
def build_03() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 03 — Baseline Models

**Phase:** 6.1 · **Status:** ✅ COMPLETE

## Objective
Establish a defaults-only baseline for five candidate model families —
Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost — trained on
the full 1,025,602-row training split, to identify which family family is
worth investing tuning effort in (Phase 6.2).

## Scientific methodology
- Every hyperparameter is a fixed, documented default (see
  `training/disease_prediction/config.py`); the ONLY deviation from each
  library's literal default is a uniform `n_estimators/iterations=200` across
  every tree ensemble, fixed *before* training, purely to bound runtime — not
  a search.
- Evaluated on the validation split, **both** the full split and the
  leakage-free subset (the primary number), per the Phase-5 leakage
  mitigation.
- Primary selection metric: **macro-F1**, not accuracy — the corpus is
  ~247:1 class-imbalanced (`docs/research/DDXPLUS_EDA.md`), so accuracy
  alone would be misleading.
"""),
        code(BOOTSTRAP_CELL.format(name="03_baseline_models")),
        code("""
import json
import pandas as pd

comparison = json.loads((paths.reports_dir / "model_comparison.json").read_text(encoding="utf-8"))
print(f"n_train_rows = {comparison['n_train_rows']:,}   n_validation_rows = {comparison['n_validation_rows']:,}")
print(f"n_features   = {comparison['n_features']}        n_classes = {comparison['n_classes']}")
print(f"primary_selection_metric = {comparison['primary_selection_metric']} on {comparison['primary_selection_split']}")
"""),
        md("## 1. Baseline ranking (validation, leakage-free subset)"),
        code("""
rows = []
for m in comparison["models"]:
    clean = m["validation_leakage_free"] or m["validation_full"]
    rows.append({
        "model": m["display_name"],
        "accuracy": clean["accuracy"], "f1_macro": clean["f1_macro"],
        "f1_weighted": clean["f1_weighted"],
        "train_seconds": m["train_seconds"], "peak_memory_mb": m["peak_memory_mb"],
        "model_size_mb": m["model_size_mb"],
    })
baseline_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
baseline_df
"""),
        md("## 2. Operational cost vs. accuracy"),
        code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
for _, r in baseline_df.iterrows():
    ax.scatter(r["train_seconds"], r["f1_macro"], s=80)
    ax.annotate(r["model"], (r["train_seconds"], r["f1_macro"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("training time (seconds, log scale)")
ax.set_xscale("log")
ax.set_ylabel("macro-F1 (validation, leakage-free)")
ax.set_title("Phase 6.1 baseline: accuracy vs. training cost")
fig.tight_layout()
fig.savefig(paths.notebook_output_dir / "baseline_accuracy_vs_cost.png", dpi=110)
plt.show()
"""),
        md("""
## Conclusions (measured, Phase 6.1)

- Under **untuned defaults**, Random Forest and CatBoost lead
  (macro-F1 ≈ 0.996), Logistic Regression is a strong linear baseline
  (≈ 0.993), while XGBoost (≈ 0.944) and especially LightGBM (≈ 0.162)
  underperform badly.
- **This ranking is provisional and later overturned** — Phase 6.2's
  hyperparameter search (notebook 04) shows XGBoost reaching the *best*
  tuned score and LightGBM recovering from 0.162 to ≈ 0.994 once a root
  cause is fixed. Any production decision taken from this notebook alone
  would have been wrong; this is why Phase 6.2 exists at all.
- The Random-Forest-vs-CatBoost gap here (≈ 0.0002) is far smaller than can
  be trusted from a single validation split — resolved with real variance
  in notebook 05 (cross-validation) and notebook 06 (statistical tests).

**Next notebook:** `04_hyperparameter_optimization.ipynb`.
"""),
    ]


# ======================================================================
# 04 — Hyperparameter Optimization  (Phase 6.2 Step 1 — COMPLETE, all 5)
# ======================================================================
def build_04() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 04 — Hyperparameter Optimization

**Phase:** 6.2, Step 1 · **Status:** ✅ COMPLETE for all 5 families

## Objective
Tune each of the five candidate families with a randomized search over a
documented space, and — critically — **root-cause any family that fails**
rather than blindly widening the search.

## Scientific methodology
- Randomized search, `n_iter` per family fixed in
  `config_optimization.py` (documented per-family; not tuned itself).
- Search spaces and iteration counts were narrowed twice during this phase,
  both changes are **compute-budget decisions, not scientific ones**, and
  are recorded verbatim below rather than silently applied:
  - Logistic Regression: `saga` solver removed after running **3h20m
    (~24,000+ CPU-seconds) without completing 1 of 8 configurations** on a
    120k-row / 286-feature / 49-class subsample — confirmed genuinely
    computing (not hung) via repeated CPU-time sampling, projected >24h to
    finish; `n_iter` reduced 8→4, `lbfgs` only.
  - CatBoost: `n_iter` reduced 8→5, `iterations=600` option dropped — fits
    are ~5x slower than XGBoost at this scale.
- LightGBM initially **catastrophically failed** (11 of 14 sampled configs
  raised `LightGBMError: Check failed: (best_split_info.left_count) > (0)`).
  This was root-caused via explicit hypothesis testing (H1–H6), not blind
  retuning: `min_child_weight=0.0` was the defect (a distinct LightGBM bug
  triggered on sparse binary 49-class data); `min_child_samples` interacting
  with per-class row counts was the deeper mechanism. The invalid run is
  preserved (never deleted) under `outputs/reports/invalidated/` with an
  explanatory record for audit.
"""),
        code(BOOTSTRAP_CELL.format(name="04_hyperparameter_optimization")),
        code("""
import json
import pandas as pd

opt = json.loads((paths.reports_dir / "optimization_results.json").read_text(encoding="utf-8"))
ranking = pd.DataFrame(opt["ranking_by_tuned_f1_macro"])
ranking
"""),
        md("## 1. LightGBM root-cause: before vs. after fix"),
        code("""
ok = json.loads((paths.reports_dir / "search_lightgbm.json").read_text(encoding="utf-8"))
bad_path = paths.reports_dir / "invalidated" / "search_lightgbm_INVALID_min_child_weight_zero.json"
bad = json.loads(bad_path.read_text(encoding="utf-8"))

pd.DataFrame([
    {"run": "INVALID (min_child_weight=0.0)", "best_f1_macro_mean": bad.get("best_f1_macro_mean"),
     "n_evaluated": bad.get("n_configurations_evaluated"), "n_failed": bad.get("n_failed")},
    {"run": "CORRECTED (min_child_weight removed from space)", "best_f1_macro_mean": ok.get("best_f1_macro_mean"),
     "n_evaluated": ok.get("n_configurations_evaluated"), "n_failed": ok.get("n_failed")},
])
"""),
        md("""
> **Note on interpretation:** the corrected LightGBM search score
> (f1_macro ≈ 0.994) looks fully competitive here. Notebook 05
> (cross-validation) shows this single-split result was **fragile, not
> robust** — see that notebook's conclusions for why LightGBM was still
> eliminated at the screening stage despite this recovery.
"""),
        md("## 2. Compute-budget exclusions (documented, not silent)"),
        code("""
excl = pd.DataFrame([
    {"family": "logistic_regression", "excluded": "solver='saga'",
     "reason": "3h20m / 8 configs, 0 completed; projected >24h"},
    {"family": "catboost", "excluded": "iterations=600 option",
     "reason": "~5x slower than xgboost at this scale; n_iter cut 8->5"},
])
excl
"""),
        md("""
## Conclusions

- All 5 families now have a completed, valid hyperparameter search.
- LightGBM's failure had a specific, falsifiable root cause
  (`min_child_weight=0.0` triggering a LightGBM internal invariant on
  sparse/binary/49-class data) rather than being "fixed" by generic
  retuning — the invalidated run is kept on disk as an audit trail.
- Two compute-budget exclusions (Logistic Regression `saga`, CatBoost
  `iterations=600`) are documented as measured constraints, not scientific
  claims that those options would underperform.

**Next notebook:** `05_cross_validation.ipynb` — this is where LightGBM's
recovery is shown to be fragile.
"""),
    ]


# ======================================================================
# 05 — Cross Validation  (Phase 6.2 Step 2 — COMPLETE, all 5)
# ======================================================================
def build_05() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 05 — Cross Validation

**Phase:** 6.2, Step 2 · **Status:** ✅ COMPLETE for all 5 families

## Objective
Replace the single-split scores from notebooks 03–04 with 5-fold
stratified cross-validation, to expose variance a single split cannot show.

## Scientific methodology
- `StratifiedKFold(n_splits=5)`, same fold assignment (same seed) reused
  across all 5 families so per-fold scores are **paired** — a precondition
  for the paired statistical tests in notebook 06.
- Metrics: `f1_macro` (primary), `accuracy`, `f1_weighted` — mean, std,
  and all 5 per-fold values retained (never just the mean).
"""),
        code(BOOTSTRAP_CELL.format(name="05_cross_validation")),
        code("""
import json
import pandas as pd

cv = json.loads((paths.reports_dir / "cross_validation.json").read_text(encoding="utf-8"))
rows = []
for family, res in cv["results"].items():
    m = res["f1_macro"]
    per_fold = [f["f1_macro"] for f in res["folds"]]
    rows.append({"family": family, "f1_macro_mean": m["mean"], "f1_macro_std": m["std"],
                 "folds": per_fold})
cv_df = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
cv_df
"""),
        md("## 1. Per-fold stability (this is where LightGBM's recovery from notebook 04 collapses)"),
        code("""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))
for _, r in cv_df.iterrows():
    folds = r["folds"]
    ax.plot(range(1, len(folds) + 1), folds, marker="o", label=r["family"])
ax.set_xlabel("fold")
ax.set_ylabel("f1_macro")
ax.set_title("5-fold macro-F1 per family (paired folds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(paths.notebook_output_dir / "cv_per_fold_stability.png", dpi=110)
plt.show()
"""),
        md("## 2. Screening decision (top-N kept for expensive deep analysis; nothing discarded)"),
        code("""
screening = json.loads((paths.reports_dir / "screening_decision.json").read_text(encoding="utf-8"))
print("advanced_to_deep_analysis:", screening["advanced_to_deep_analysis"])
print("eliminated:", screening["eliminated"])
print()
print("rationale:", screening["rationale"])
"""),
        md("""
## Conclusions

- **LightGBM's Step-1 recovery (f1_macro ≈ 0.994) was fragile, not
  robust**: 5-fold CV gives f1_macro = 0.7535 ± 0.3328, with 2 of 5 folds
  collapsing to ~0.34–0.44. A model that catastrophically fails on 40% of
  folds is disqualifying for production **regardless of its mean score** —
  this is exactly the failure mode single-split evaluation cannot detect,
  and is the direct justification for the user's original insistence on
  full 5-fold CV before any decision.
- XGBoost is the most stable and highest-scoring family
  (f1_macro = 0.9961 ± 0.0005), followed closely by Random Forest
  (0.9955 ± 0.0006) and CatBoost (0.9954 ± 0.0008).
- Screening keeps **XGBoost, Random Forest, CatBoost** for the expensive
  Phase 6.2 deep analyses (calibration, SHAP, robustness, error analysis,
  clinical validation); Logistic Regression and LightGBM are eliminated
  from further compute **but their full CV results are retained, never
  discarded** (`screening_decision.json`).

**Next notebook:** `06_statistical_tests.ipynb` — is the XGBoost /
Random Forest / CatBoost gap real or noise?
"""),
    ]


# ======================================================================
# 06 — Statistical Tests  (Phase 6.2 Step 2b — COMPLETE)
# ======================================================================
def build_06() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 06 — Statistical Significance Tests

**Phase:** 6.2, Step 2b · **Status:** ✅ COMPLETE

## Objective
Determine whether the small f1_macro gaps between the top-ranked families
(notebook 05) are statistically meaningful or just resampling noise —
mean ± std alone cannot answer this.

## Scientific methodology
- **Primary test:** Nadeau–Bengio corrected resampled paired t-test — the
  standard correction for the fact that k-fold CV folds are not
  independent; uses a variance-inflation factor of `1/k + 1/(k-1)` instead
  of the naive `1/k`.
- **Reference-only tests:** uncorrected paired t-test (shown for contrast —
  it is anti-conservative and should not drive conclusions) and Wilcoxon
  signed-rank (shown for completeness — with k=5 folds its minimum
  achievable p-value is 0.0625, so it can never reach the conventional
  α=0.05 threshold at this sample size and is structurally underpowered
  here).
- **Multiple-comparison correction:** Holm–Bonferroni across all pairwise
  comparisons among the top-N families.
- **Effect size:** paired Cohen's d.
- **Practical-significance threshold Δ=0.002**, pre-declared (in
  `statistical_tests.py`, written before these results were seen) — a
  statistically significant difference smaller than Δ is reported as
  "significant but not practically meaningful."
"""),
        code(BOOTSTRAP_CELL.format(name="06_statistical_tests")),
        code("""
import json
import pandas as pd

stats = json.loads((paths.reports_dir / "statistical_comparison.json").read_text(encoding="utf-8"))
f1m = stats["f1_macro"]
pd.DataFrame(f1m["pairwise_comparisons_top_n"])
"""),
        md("## 1. Nadeau-Bengio corrected p-values vs. the pre-declared Δ=0.002 threshold"),
        code("""
rows = []
for c in f1m["pairwise_comparisons_top_n"]:
    rows.append({
        "pair": f"{c['model_a']} vs {c['model_b']}",
        "mean_diff": c["mean_difference"],
        "nb_corrected_p": c["nadeau_bengio_corrected_t_test"]["p_value"],
        "nb_p_holm_adjusted": c["holm_bonferroni_adjusted_p"],
        "cohens_d": c["cohens_d_paired"],
        "statistically_significant": c["statistically_significant"],
        "practically_significant": c["practically_significant"],
        "verdict": c["verdict"],
    })
pd.DataFrame(rows)
"""),
        md("""
## Conclusions

- With the Nadeau–Bengio correction and Holm–Bonferroni adjustment applied,
  the XGBoost vs. Random Forest vs. CatBoost gaps (all ≤ ~0.0007 mean
  f1_macro) are evaluated against the pre-declared Δ=0.002 practical
  threshold — see the `verdict` column above for the exact per-pair
  statistical/practical outcome (read directly from
  `statistical_comparison.json`, not asserted here).
- The uncorrected paired t-test is shown only as a contrast to demonstrate
  why the correction matters (it tends to overstate significance for
  correlated CV folds) — it is never used as the basis for a decision in
  this project.
- Wilcoxon is retained for completeness but is structurally unable to
  reach α=0.05 at k=5 folds (min p=0.0625) and is not used as the primary
  test.

**Next notebook:** `07_probability_calibration.ipynb` — first of the
per-model deep-analysis notebooks (XGBoost complete; Random Forest and
CatBoost pending).
"""),
    ]


# ----------------------------------------------------------------------
# Shared scaffold for the 5 partial (XGBoost-only) deep-analysis notebooks
# ----------------------------------------------------------------------
def _partial_deep_analysis_notebook(
    *,
    number: str,
    title: str,
    objective: str,
    methodology: str,
    json_key: str,
    result_cell: str,
    conclusions_complete: str,
    module_hint: str,
) -> List["nbf.NotebookNode"]:
    cells = [
        md(f"""
# {number} — {title}

**Phase:** 6.2, Step 4 (deep analysis) · **Status:** 🟡 PARTIAL — XGBoost
only; Random Forest and CatBoost pending

## Objective
{objective}

## Scientific methodology
{methodology}

## Coverage
- ✅ **XGBoost** — complete, see `experiments_xgboost.json['{json_key}']`.
- ⏳ **Random Forest** — PENDING (`experiments_random_forest.json` does not
  exist yet).
- ⏳ **CatBoost** — PENDING (`experiments_catboost.json` does not exist
  yet).

This notebook shows real, measured XGBoost output below, and explicit
PENDING sections for the other two — nothing is fabricated or estimated.
"""),
        code(BOOTSTRAP_CELL.format(name=f"{number}_{title.lower().replace(' ', '_').replace('/', '_')}")),
        code("""
from training.disease_prediction.notebook_utils import load_json, pending_banner
from IPython.display import Markdown

xgb = load_json(paths.reports_dir / "experiments_xgboost.json")
rf = load_json(paths.reports_dir / "experiments_random_forest.json")
cb = load_json(paths.reports_dir / "experiments_catboost.json")
print("xgboost:", "loaded" if xgb else "MISSING")
print("random_forest:", "loaded" if rf else "MISSING (pending)")
print("catboost:", "loaded" if cb else "MISSING (pending)")
"""),
        md(f"## 1. XGBoost — {title.lower()} (measured)"),
        code(result_cell),
        md("## 2. Random Forest — pending"),
        code(f"""
if rf is None:
    display(Markdown(pending_banner(
        "Random Forest {title.lower()}",
        "python -m training.disease_prediction.run_experiments --stage models --top-n 3 --fit-rows 250000",
    )))
else:
    print("rf['{json_key}'] is now available -- re-run this notebook's authoring step to add real output here.")
"""),
        md("## 3. CatBoost — pending"),
        code(f"""
if cb is None:
    display(Markdown(pending_banner(
        "CatBoost {title.lower()}",
        "python -m training.disease_prediction.run_experiments --stage models --top-n 3 --fit-rows 250000",
    )))
else:
    print("cb['{json_key}'] is now available -- re-run this notebook's authoring step to add real output here.")
"""),
        md(f"""
## Conclusions

**XGBoost (measured):**
{conclusions_complete}

**Random Forest / CatBoost:** not yet measured — no conclusion is drawn
for these two families in this notebook. Do not infer their behaviour from
XGBoost's results; tree-ensemble calibration/robustness/explainability
characteristics are not transferable across families.

Re-run via `{module_hint}` when ready to fill in the two pending sections,
then re-execute this notebook.
"""),
    ]
    return cells


def build_07() -> List["nbf.NotebookNode"]:
    return _partial_deep_analysis_notebook(
        number="07",
        title="Probability Calibration",
        objective=(
            "Assess whether each model's predicted class probabilities are "
            "trustworthy (a 0.8 confidence should be right ~80% of the "
            "time) — essential before probabilities are ever surfaced to a "
            "clinician or patient."
        ),
        methodology=(
            "- Expected Calibration Error (ECE) and Maximum Calibration "
            "Error (MCE) over confidence bins.\n"
            "- Brier score as a proper scoring rule.\n"
            "- Reliability diagram (predicted confidence vs. observed "
            "accuracy per bin)."
        ),
        json_key="calibration",
        result_cell="""
calib = xgb["calibration"]
import pandas as pd
display(pd.Series({k: v for k, v in calib.items() if not isinstance(v, (list, dict))}, name="value").to_frame())
print("Reliability diagram bins:")
pd.DataFrame(calib["reliability_bins"])
""",
        conclusions_complete=(
            "ECE/MCE/Brier are read directly from `experiments_xgboost.json"
            "['calibration']` above — see the printed table for exact "
            "values; no threshold judgement is asserted here beyond what "
            "that data shows."
        ),
        module_hint="run_experiments.py --stage models --top-n 3",
    )


def build_08() -> List["nbf.NotebookNode"]:
    return _partial_deep_analysis_notebook(
        number="08",
        title="Explainability SHAP",
        objective=(
            "Explain individual and global model predictions using three "
            "complementary views (SHAP TreeExplainer, permutation "
            "importance, impurity/Gini importance), mapped back to HEALIX "
            "ontology symptom IDs so explanations are clinically legible."
        ),
        methodology=(
            "- SHAP `TreeExplainer` on a fixed-seed sample of the "
            "validation set.\n"
            "- Permutation importance (model-agnostic, cross-checks SHAP).\n"
            "- Impurity/Gini importance (cheap, included for contrast — "
            "known to be biased toward high-cardinality features, hence "
            "never used alone)."
        ),
        json_key="explainability",
        result_cell="""
expl = xgb["explainability"]
import pandas as pd

print("SHAP available:", expl["shap"]["available"],
      " n_background:", expl["shap"]["n_background"],
      " n_explained:", expl["shap"]["n_explained"])
display(pd.DataFrame(expl["shap"]["top_features"]).head(20))

print("Permutation importance (top 20):")
display(pd.DataFrame(expl["permutation_importance_top"]).head(20))

print("Impurity (Gini) importance (top 20) -- shown for contrast, known bias toward high-cardinality features:")
pd.DataFrame(expl["impurity_importance_top"]).head(20)
""",
        conclusions_complete=(
            "Top contributing ontology-mapped features per the three "
            "importance views are read directly from "
            "`experiments_xgboost.json['explainability']` above."
        ),
        module_hint="run_experiments.py --stage models --top-n 3",
    )


def build_09() -> List["nbf.NotebookNode"]:
    return _partial_deep_analysis_notebook(
        number="09",
        title="Robustness Analysis",
        objective=(
            "Measure how much macro-F1 degrades as input symptoms are "
            "randomly masked (10/20/30/40%), under two distinct masking "
            "semantics — proxying real conversations where not every "
            "symptom gets asked."
        ),
        methodology=(
            "- `to_nan` masking: masked evidence becomes 'unknown/not "
            "asked' (open-world interpretation).\n"
            "- `to_zero` masking: masked evidence becomes 'asserted "
            "absent' (closed-world interpretation — matches the training "
            "data's own semantics, see notebook 02).\n"
            "- Both are reported since real Healix conversations are "
            "open-world but the training data is closed-world — the gap "
            "between the two masking semantics is itself a diagnostic."
        ),
        json_key="robustness",
        result_cell="""
rob = xgb["robustness"]
import pandas as pd

print("baseline (0% masked):", rob["baseline"])
print()
print("to_nan (open-world: masked = 'not asked'):")
display(pd.DataFrame(rob["curves"]["to_nan"]))
print("to_zero (closed-world: masked = 'asserted absent', matches training-data semantics):")
pd.DataFrame(rob["curves"]["to_zero"])
""",
        conclusions_complete=(
            "Degradation curves for both masking semantics are read "
            "directly from `experiments_xgboost.json['robustness']` above."
        ),
        module_hint="run_experiments.py --stage models --top-n 3",
    )


def build_10() -> List["nbf.NotebookNode"]:
    return _partial_deep_analysis_notebook(
        number="10",
        title="Error Analysis",
        objective=(
            "Characterize *where* the model is wrong — which disease pairs "
            "are most confused, whether errors cluster by severity, and "
            "whether the confusion matrix reveals clinically dangerous "
            "mistakes (e.g., high-severity condition misclassified as "
            "low-severity)."
        ),
        methodology=(
            "- Full 49x49 confusion matrix (vectorized via `np.add.at`, "
            "fixed from an earlier `erra.np` bug — see project history).\n"
            "- Top confused pairs ranked by count.\n"
            "- Errors cross-tabulated against DDXPlus severity labels."
        ),
        json_key="error_analysis",
        result_cell="""
erra = xgb["error_analysis"]
import pandas as pd

print("Top confused pairs (directional -- true -> predicted):")
display(pd.DataFrame(erra["top_confusions_directional"]).head(20))
print("Top confused pairs (symmetric):")
pd.DataFrame(erra["top_confusion_pairs_symmetric"]).head(20)
""",
        conclusions_complete=(
            "Most-confused disease pairs and their severity cross-tab are "
            "read directly from `experiments_xgboost.json['error_analysis']` "
            "above."
        ),
        module_hint="run_experiments.py --stage models --top-n 3",
    )


def build_11() -> List["nbf.NotebookNode"]:
    return _partial_deep_analysis_notebook(
        number="11",
        title="Clinical Validation",
        objective=(
            "Evaluate the model the way a differential-diagnosis tool is "
            "actually used: not 'is the top-1 prediction correct' but "
            "'is the right disease anywhere in a short ranked list', "
            "compared against DDXPlus's own ranked differential."
        ),
        methodology=(
            "- Top-k accuracy (k=1,3,5).\n"
            "- Mean Reciprocal Rank (MRR).\n"
            "- Differential-diagnosis coverage against DDXPlus's own "
            "per-patient ranked differential list (not just the single "
            "ground-truth label)."
        ),
        json_key="clinical_validation",
        result_cell="""
cval = xgb["clinical_validation"]
import pandas as pd
pd.Series({k: v for k, v in cval.items() if not isinstance(v, (list, dict))}, name="value").to_frame()
""",
        conclusions_complete=(
            "Top-k accuracy, MRR, and differential coverage are read "
            "directly from `experiments_xgboost.json['clinical_validation']` "
            "above."
        ),
        module_hint="run_experiments.py --stage models --top-n 3",
    )


# ======================================================================
# 12 — Final Model Selection  (BLOCKED — cannot honestly conclude yet)
# ======================================================================
def build_12() -> List["nbf.NotebookNode"]:
    return [
        md("""
# 12 — Final Model Selection

**Phase:** 6.2, Step 5 (scientific decision) · **Status:** 🔴 BLOCKED

## Objective
Produce the final GO / CONDITIONAL GO / NO-GO production recommendation,
justified across all 11 evaluation axes (predictive performance,
stability, calibration, robustness, explainability, computational cost,
statistical significance, model size, inference speed, deployability,
synthetic-data external validity).

## Why this notebook is blocked
Notebooks 07–11 (calibration, SHAP, robustness, error analysis, clinical
validation) are only complete for **XGBoost**. Random Forest and CatBoost
were selected into the same top-3 screening tier
(`screening_decision.json`) but their deep-analysis artifacts
(`experiments_random_forest.json`, `experiments_catboost.json`) do not
exist yet.

A final selection between XGBoost, Random Forest, and CatBoost **cannot be
made honestly** without comparing all three on calibration, robustness,
explainability, and clinical validation — not just cross-validated
macro-F1 (notebook 05) and its statistical significance (notebook 06),
which are the only axes currently complete for all three.

This notebook is deliberately left as a **scaffold**: it loads and shows
everything that IS available now, and marks the decision itself as
pending — per the explicit instruction not to recreate fake results for
an interrupted/incomplete experiment.
"""),
        code(BOOTSTRAP_CELL.format(name="12_final_model_selection")),
        md("## 1. What IS available now (all 5 families, CV + significance)"),
        code("""
import json
import pandas as pd

cv = json.loads((paths.reports_dir / "cross_validation.json").read_text(encoding="utf-8"))
screening = json.loads((paths.reports_dir / "screening_decision.json").read_text(encoding="utf-8"))
stats = json.loads((paths.reports_dir / "statistical_comparison.json").read_text(encoding="utf-8"))

rows = []
for family, res in cv["results"].items():
    m = res["f1_macro"]
    rows.append({"family": family, "f1_macro_mean": m["mean"], "f1_macro_std": m["std"],
                 "screened_in": family in screening["advanced_to_deep_analysis"]})
pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
"""),
        md("## 2. What is MISSING for a defensible final decision"),
        code("""
from training.disease_prediction.notebook_utils import load_json, pending_banner
from IPython.display import Markdown

missing = []
for family in screening["advanced_to_deep_analysis"]:
    path = paths.reports_dir / f"experiments_{family}.json"
    if load_json(path) is None:
        missing.append(family)

if missing:
    display(Markdown(pending_banner(
        f"Deep analysis (calibration/SHAP/robustness/error-analysis/clinical-validation) for: {', '.join(missing)}",
        "python -m training.disease_prediction.run_experiments --stage models --top-n 3 --fit-rows 250000",
    )))
else:
    print("All screened families have complete deep-analysis artifacts -- decision logic can now be written.")
"""),
        md("""
## Conclusion — decision explicitly withheld

**No GO / CONDITIONAL GO / NO-GO verdict is issued in this notebook.**

Issuing one now, with 2 of 3 finalist models missing calibration,
robustness, explainability, and clinical-validation evidence, would mean
deciding on cross-validated macro-F1 alone — exactly the failure mode this
whole Phase 6.2 protocol exists to avoid (see notebook 05: the top-3
macro-F1 gap is ≤0.0007 and LightGBM's own single-metric score was
previously shown to hide a catastrophic stability failure).

### To unblock this notebook
1. Run the remaining deep analyses for Random Forest and CatBoost:
   `python -m training.disease_prediction.run_experiments --stage models --top-n 3 --fit-rows 250000`
2. Re-execute notebooks 07–11 (they will auto-detect the new artifacts and
   replace their PENDING sections with real output).
3. Re-execute this notebook — at that point, and only then, add the
   11-axis comparison table and the final answers to the 5 decision
   questions (statistically significant? practically significant?
   deployable? conditional caveats? synthetic-data limitations?).

Until step 1 completes, this project's status for production deployment
is: **decision pending, not NO-GO and not GO.**
"""),
    ]


# ======================================================================
# Driver
# ======================================================================
NOTEBOOK_BUILDERS: List[tuple] = [
    ("01_dataset_inspection", build_01),
    ("02_dataset_validation", build_02),
    ("03_baseline_models", build_03),
    ("04_hyperparameter_optimization", build_04),
    ("05_cross_validation", build_05),
    ("06_statistical_tests", build_06),
    ("07_probability_calibration", build_07),
    ("08_explainability_shap", build_08),
    ("09_robustness_analysis", build_09),
    ("10_error_analysis", build_10),
    ("11_clinical_validation", build_11),
    ("12_final_model_selection", build_12),
]


def main() -> None:
    for name, builder in NOTEBOOK_BUILDERS:
        cells = builder()
        path = write_notebook(name, cells)
        print(f"wrote {path}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()

