"""ml_corroborate: an optional, doctor-report-only annotation on
candidates rag_retrieve already found, using the vendored XGBoost model
(ml/) as a second, independent signal.

Two gates, no probability threshold anywhere:
1. Density gate (ml.density_floor): skip a candidate unless the feature
   vector has at least MIN_REQUIRED_MATCHED of that disease's own
   real training-observed features.
2. Rank-agreement gate: only when the density gate clears, check whether
   the candidate's crosswalk-mapped class is the model's own argmax. If
   yes, candidate["ml_corroboration"] = "model_signal_present"; if no,
   the key is left off entirely (absent, never a lower-confidence label).

The feature vector is built once per turn (not per-candidate), so
predict_proba() is called exactly once and argmax reused for every
candidate.

Never introduces a new candidate — only ever adds a key to a dict
rag_retrieve already produced. No raw score ever leaves this function;
the only value attached to state is the fixed string above.

Fails open: any error in the model-dependent path is caught, logged to
"healix.ml" (not the patient-data audit logger — this log line never
carries patient-authored text), and the turn proceeds with
candidate_diseases completely unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from ml.density_floor import clears_density_floor
from ml.disease_crosswalk import xgboost_label_for
from ml.feature_mapper import build_feature_vector
from ml.model_loader import load_bundle
from state import CandidateDisease, HealixState

_logger = logging.getLogger("healix.ml")

_SIGNAL_VALUE = "model_signal_present"


def ml_corroborate(state: HealixState) -> dict[str, Any]:
    candidates: list[CandidateDisease] = state.get("candidate_diseases", [])
    if not any(xgboost_label_for(candidate["name"]) for candidate in candidates):
        return {}

    try:
        bundle = load_bundle()
        vector = build_feature_vector(state, bundle.feature_order)
        active_features = frozenset(
            feature for feature, value in zip(bundle.feature_order, vector) if value == 1.0
        )

        proba = bundle.model.predict_proba([vector])[0]
        argmax_index = max(range(len(proba)), key=lambda i: proba[i])
        argmax_label = bundle.label_encoder.inverse_transform([argmax_index])[0]
    except Exception:
        _logger.exception("ml_corroborate: model unavailable this turn, proceeding without it")
        return {}

    updated: list[CandidateDisease] = []
    for candidate in candidates:
        xgb_label = xgboost_label_for(candidate["name"])
        if (
            xgb_label is not None
            and xgb_label == argmax_label
            and clears_density_floor(candidate["name"], active_features)
        ):
            updated.append({**candidate, "ml_corroboration": _SIGNAL_VALUE})
        else:
            updated.append(candidate)

    return {"candidate_diseases": updated}
