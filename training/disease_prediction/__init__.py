"""
Healix - Disease Prediction training package (Phase 6.1).

Offline-only. Trains and compares baseline model families (Logistic
Regression, Random Forest, XGBoost, LightGBM, CatBoost) on the Phase-5
DDXPlus Parquet dataset (Feature Schema v2 / healix-ontology-v1.0.0).

This package does NOT:
  * modify app/domain, app/routes, app/services, or any runtime file,
  * replace RuleBasedDiseasePredictor,
  * get imported by app.main or any FastAPI route,
  * perform hyperparameter search, cross-validation, or threshold calibration
    (that is Phase 6.2).

Modules
-------
config.py    - paths, seed, per-model default hyperparameters (fixed, not tuned)
utils.py     - logging, timing, memory sampling, reproducibility helpers
dataset.py   - loads the Phase-5 Parquet dataset; builds X/y; label encoding
models.py    - unfitted estimator factory for each model family
metrics.py   - the full evaluation metric suite
train.py     - CLI: trains every model family, evaluates, saves everything
evaluate.py  - re-load a saved model and evaluate it against any split
predict.py   - offline demonstration inference (NOT wired into the runtime)
"""
