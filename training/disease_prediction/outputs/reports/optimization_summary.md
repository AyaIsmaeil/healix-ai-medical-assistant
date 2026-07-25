# Hyperparameter Optimization Summary (Phase 6.2 — Step 1)

> Stage 1 of the two-stage protocol: randomized search per family on a fixed stratified 120,000-row subsample, scored by 3-fold macro-F1.

> **Generated:** 2026-07-22T22:19:21+00:00 — rendered automatically from the experiment JSON artifacts in `training/disease_prediction/outputs/reports/`. No value in this document was transcribed or estimated by hand; `NOT MEASURED` appears wherever an experiment did not run.

> **Data caveat (applies to every number below):** all results come from `ddxplus-v2.0-a8490b06a563`, a **synthetic, closed-world, English/French** corpus. See [`SCIENTIFIC_LIMITATIONS_AND_DEPLOYMENT_ROADMAP.md`](SCIENTIFIC_LIMITATIONS_AND_DEPLOYMENT_ROADMAP.md) for what these numbers do and do not support.

## Tuned results vs the Phase 6.1 defaults-only baseline

| Family | Phase 6.1 baseline F1 | Tuned F1 (search CV) | Δ | Configs | Failed |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | 0.94410 | 0.99576 | +0.05166 | 12 | 0 |
| **CatBoost** | 0.99594 | 0.99476 | -0.00118 | 5 | 0 |
| **Random Forest** | 0.99615 | 0.99455 | -0.00160 | 12 | 0 |
| **LightGBM** | 0.16233 | 0.99374 | +0.83142 | 14 | 0 |
| **Logistic Regression** | 0.99333 | 0.99325 | -0.00009 | 4 | 0 |

> The Phase 6.1 column is single-split validation macro-F1; the tuned column is a 3-fold CV mean on a subsample. They are not measured on identical data, so Δ indicates direction and rough magnitude, not an exact gain.

## Winning configuration per family

### Logistic Regression

- Best macro-F1 (search CV): **0.99325** ± 0.00065
- Best params: `{"C": 1.0, "solver": "lbfgs"}`
- Phase 6.1 baseline params: `{"max_iter": 1000, "n_jobs": null, "solver": "lbfgs"}`
- Configurations evaluated: 4 (failed: 0)
- Mean fit time of best config: 254.8 s

### Random Forest

- Best macro-F1 (search CV): **0.99455** ± 0.00059
- Best params: `{"max_depth": 30, "max_features": "log2", "min_samples_leaf": 1, "n_estimators": 300}`
- Phase 6.1 baseline params: `{"class_weight": null, "max_depth": null, "min_samples_leaf": 1, "n_estimators": 200, "n_jobs": -1}`
- Configurations evaluated: 12 (failed: 0)
- Mean fit time of best config: 13.3 s

### XGBoost

- Best macro-F1 (search CV): **0.99576** ± 0.00027
- Best params: `{"colsample_bytree": 0.7, "learning_rate": 0.05, "max_depth": 10, "subsample": 0.7}`
- Phase 6.1 baseline params: `{"eval_metric": "mlogloss", "n_estimators": 200, "n_jobs": -1, "tree_method": "hist"}`
- Configurations evaluated: 12 (failed: 0)
- Mean fit time of best config: 108.1 s

### LightGBM

- Best macro-F1 (search CV): **0.99374** ± 0.00018
- Best params: `{"learning_rate": 0.05, "min_child_samples": 10, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 400, "num_leaves": 255}`
- Phase 6.1 baseline params: `{"n_estimators": 200, "n_jobs": -1, "verbose": -1}`
- Configurations evaluated: 14 (failed: 0)
- Mean fit time of best config: 163.5 s

### CatBoost

- Best macro-F1 (search CV): **0.99476** ± 0.00054
- Best params: `{"depth": 6, "iterations": 200, "l2_leaf_reg": 10.0, "learning_rate": 0.3}`
- Phase 6.1 baseline params: `{"iterations": 200, "thread_count": -1, "verbose": false}`
- Configurations evaluated: 5 (failed: 0)
- Mean fit time of best config: 320.6 s
