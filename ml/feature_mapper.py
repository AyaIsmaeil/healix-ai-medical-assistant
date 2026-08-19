"""feature_mapper: translates Healix_chatbot's confirmed Arabic symptoms
into the binary feature vector ml/model_loader's XGBoost bundle expects.

--- Direct identity match, not a translation table ---

ml/models/xgboost-healix-arabic-v1.0/ was trained directly on
vocabulary/symptoms.py's own canonical Arabic terms —
ml/training/disease_symptom_checklist_training.ipynb's
filter_to_canonical() step built every feature column FROM
vocabulary/symptoms.py's CANONICAL_SYMPTOMS, normalized the same way
(rules.crisis.normalize) everywhere else in this project compares symptom
names. Every feature_schema.json column is therefore exactly
normalize(some canonical term) — confirmed for all of the current
bundle's columns against the live vocabulary by
tests/unit/test_ml_feature_mapper.py.

This replaces the earlier bundle's hand-authored SYMPTOM_TO_FEATURE
translation dictionary (52/131 columns mapped, against an
English-vocabulary Kaggle dataset): there is nothing left to translate,
only to normalize and match directly. mapped_features()/
unmapped_features() are kept for ml.density_floor's own consistency
checks, now defined against the full canonical vocabulary rather than a
hand-typed dict's value set.
"""

from __future__ import annotations

from rules.crisis import normalize
from state import HealixState, Symptom
from vocabulary.symptoms import CANONICAL_SYMPTOMS


def _normalized_names(symptoms: list[Symptom]) -> set[str]:
    # Mirrors nodes.rag_retrieve._normalized_names exactly — same
    # normalize-before-compare discipline, applied at this layer too.
    return {normalize(symptom["name"]) for symptom in symptoms if symptom.get("name")}


def build_feature_vector(state: HealixState, feature_order: tuple[str, ...]) -> list[float]:
    """A 0/1 vector in `feature_order`'s exact column order, built from
    state["symptoms"] minus state["negated_symptoms"] (negation
    suppresses a feature the same way it carries real evidentiary weight
    everywhere else in this project — CLAUDE.md > State).

    Each column is a normalized canonical Arabic term; a symptom sets its
    column directly, no intermediate translation step.
    """
    confirmed = _normalized_names(state.get("symptoms", []))
    negated = _normalized_names(state.get("negated_symptoms", []))
    effective = confirmed - negated

    return [1.0 if column in effective else 0.0 for column in feature_order]


def mapped_features() -> frozenset[str]:
    """Every normalized canonical symptom term this module can ever set a
    feature column to 1 for — vocabulary/symptoms.py's full
    CANONICAL_SYMPTOMS set (already normalized at import there). Used by
    ml.density_floor to confirm its own expected features are all
    representable from Arabic input."""
    return CANONICAL_SYMPTOMS


def unmapped_features(feature_order: tuple[str, ...]) -> frozenset[str]:
    """Computed from the live feature order, not a second hand-maintained
    list — see module docstring."""
    return frozenset(feature_order) - mapped_features()
