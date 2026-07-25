#!/usr/bin/env bash
# Healix - Phase 6.2 completion chain (OFFLINE ONLY).
#
# Resumes from the current checkpoint and runs every remaining step in order,
# stopping immediately if any step fails (set -e) so a failure is investigated
# rather than silently skipped.
#
# Steps:
#   0. wait for the in-flight CatBoost hyperparameter search to finish
#   1. consolidate Step-1 artifacts (optimization_results.json / _summary.md)
#   2. 5-fold cross-validation for ALL five families + statistical tests
#   3. screening -> top 3 (screening_decision.json)
#   4. deep analyses for the selected models only
#   5. generate all markdown reports
set -euo pipefail

cd /c/healix-ai-medical-assistant
export PYTHONPATH=/c/healix-ai-medical-assistant
REPORTS=training/disease_prediction/outputs/reports
LOG=/tmp/phase62_chain.out

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== STEP 0: waiting for CatBoost hyperparameter search ==="
while [ ! -f "$REPORTS/search_catboost.json" ]; do
  if ! tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi python; then
    say "FATAL: search process died before writing search_catboost.json"
    exit 1
  fi
  sleep 60
done
say "CatBoost search complete."

say "=== STEP 1: consolidate optimization artifacts ==="
python -c "
from training.disease_prediction import reporting as r
p = r.build_optimization_results()
r.write_optimization_summary(p)
print('ranking:', [(x['family'], x['best_f1_macro_mean']) for x in p['ranking_by_tuned_f1_macro']])
" 2>&1 | tee -a "$LOG"

say "=== STEP 2: 5-fold cross-validation (all 5 families) + statistics ==="
python -m training.disease_prediction.run_experiments --stage cv 2>&1 | tee -a "$LOG"

say "=== STEP 2b: CROSS_VALIDATION_REPORT.md ==="
python -c "
from training.disease_prediction import reporting as r
print('wrote', r.write_cross_validation_report())
" 2>&1 | tee -a "$LOG"

say "=== STEPS 3+4: screening to top 3, then deep analyses on those only ==="
python -m training.disease_prediction.run_experiments \
    --stage models --top-n 3 --fit-rows 250000 2>&1 | tee -a "$LOG"

say "=== STEP 5: generate explainability + error-analysis reports ==="
python -c "
from training.disease_prediction import reporting as r
print('wrote', r.write_explainability_report())
print('wrote', r.write_error_analysis_report())
" 2>&1 | tee -a "$LOG"

say "=== PHASE 6.2 EXPERIMENT CHAIN COMPLETE ==="
