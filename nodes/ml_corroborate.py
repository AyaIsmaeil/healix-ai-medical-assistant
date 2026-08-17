"""ml_corroborate: an optional, doctor-report-only annotation on
candidates nodes.rag_retrieve already found, using the vendored XGBoost
model (ml/) as a second, independent signal. Runs after rag_retrieve,
before diagnose (CLAUDE.md > Graph flow: not yet wired into graph.py in
this change — CLAUDE.md > Working style: one node at a time).

See CLAUDE.md's XGBoost corroboration-signal section for the full audit
this design rests on (checksum-verified bundle, a 13-disease crosswalk
verified against the real label encoder, and the sparse-input finding
that ruled out probability-magnitude thresholds entirely). Summary of
what that audit forced this node to be:

--- Two gates, no probability threshold anywhere ---

1. Density gate (ml.density_floor): skip a candidate entirely unless the
   built feature vector contains at least MIN_REQUIRED_MATCHED of that
   specific disease's own real training-observed, Arabic-mappable
   features. Confirmed empirically necessary — a two-feature vector
   (rag/knowledge_base/hypertension.json's own bare symptom list) is
   already sparser than what this model's training data typically uses.
2. Rank-agreement gate: only when the density gate clears, check whether
   this candidate's crosswalk-mapped XGBoost class is the model's own
   argmax (top pick) across all 41 classes for this turn's feature
   vector. If yes: candidate["ml_corroboration"] = "model_signal_present".
   If no — or the density gate never cleared, or the candidate has no
   crosswalk entry at all — the key is left OFF that candidate's dict
   entirely (absent, never None, never a lower-confidence label).

The feature vector is the same for every candidate in a given turn (it
is built once from state["symptoms"]/state["negated_symptoms"], not
per-candidate), so predict_proba() is called exactly once per turn, and
argmax is computed once and reused for every candidate's rank-agreement
check — not once per candidate.

--- Safety rule 6: never a new candidate ---

This node only ever ADDS a key to a dict already present in
state["candidate_diseases"] — nodes.rag_retrieve's own output. It cannot
introduce a disease RAG did not already retrieve; there is no code path
here that appends to candidate_diseases at all.

--- Safety rule 7 / the user's own explicit requirement: no raw score,
anywhere, patient or doctor ---

predict_proba()'s actual numbers never leave this function. The only
value ever attached to state is the fixed string "model_signal_present" —
a technical, binary label, not a clinical confidence level (CLAUDE.md
is explicit that this is not to be read as one).

--- Fails open: XGBoost is not on the critical path ---

Any error anywhere in the model-dependent part of this node (bundle load
failure, a feature/label lookup problem, predict_proba raising) is
caught, logged (to "healix.ml", not the patient-data audit logger — see
below), and the turn proceeds with candidate_diseases COMPLETELY
UNCHANGED, as if this node had not run at all. A broken or missing model
must never be the reason a triage turn fails.

Logged to a plain, non-audit logger deliberately: unlike an LLM call
(audit.logger.log_llm_call, which records the patient's own raw text),
an ml_corroborate failure's log line carries only an exception
type/message and, at most, a disease name already public in
rag/knowledge_base/ — no patient-authored free text. It does not meet
CLAUDE.md > Audit logs and patient data's "assume patient data unless
demonstrably not" bar the way an LLM call's raw response does, so it is
not routed through that channel.
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
    """Annotate any state["candidate_diseases"] entries that clear both
    gates with ml_corroboration="model_signal_present". Never touches
    candidate_diseases at all (returns {}) if nothing here has a
    crosswalk entry, or if anything about the model fails."""
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
