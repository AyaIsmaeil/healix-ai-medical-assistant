# Training notebook — reference only, not run by this service

`disease_symptom_checklist_training.ipynb` is the notebook that trained the
XGBoost bundle vendored at `ml/models/xgboost-symptom-checklist-v1.0-aca19e97690a/`
(`nodes/ml_corroborate.py`, CLAUDE.md's XGBoost corroboration-signal section).
Copied here from its source project, `healix-ai-medical-assistant`
(`training/notebooks/disease_symptom_checklist_training.ipynb`), for
reference and reproducibility — not imported, executed, or otherwise
depended on by any code in this service.

`data/dataset.csv` is the raw dataset the notebook trains on (Kaggle,
itachi9604-style symptom-checklist dataset — see the vendored bundle's own
`metadata.json` for the full provenance/limitations already documented in
CLAUDE.md's Known limitations: 41 diseases, small/synthetic, 5–10 unique
examples per class after deduplication).

Re-running the notebook requires the source project's own environment
(`scikit-learn`, `xgboost`, `pandas`, etc. — see that project's own
`requirements.txt`; the exact versions this service itself depends on for
*inference* are pinned separately in `requirements.txt` at this repo's
root). Re-training and re-vendoring an updated bundle is a deliberate,
reviewed action — see `ml/disease_crosswalk.py` and `ml/density_floor.py`'s
own module docstrings for what would need re-verifying against a new bundle
(class labels, feature order, per-disease density floors) before it could
safely replace the current one.
