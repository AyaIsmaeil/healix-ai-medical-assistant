"""density_floor: per-disease minimum feature-density gate for
nodes.ml_corroborate — the fix for the sparse-input problem found while
auditing this integration (see CLAUDE.md's XGBoost corroboration-signal
section for the full writeup).

--- Why this exists ---

The XGBoost model needs a feature vector reasonably close to its own
training-time pattern to say anything reliable. Confirmed empirically
during the audit: feeding the real model only Hypertension's two
rag/knowledge_base/hypertension.json symptoms (headache + dizziness)
still landed Hypertension as the model's own top pick, but a full,
training-pattern-matching row is what the model actually learned from —
an arbitrary probability-magnitude threshold calibrated against dense
rows would misrepresent what the model does on the sparse input this
integration realistically produces (CLAUDE.md, and the user's own
decision: no probability-magnitude thresholds anywhere in this feature).

This module answers a narrower, structural question instead: for THIS
disease, does the built feature vector contain enough of the columns
that disease's real training examples actually used, for
nodes.ml_corroborate to even bother asking the model?

--- Provenance — how EXPECTED_FEATURES_BY_DISEASE was derived ---

For each of ml.disease_crosswalk.DISEASE_CROSSWALK's 13 pairs, computed
directly against the real training + holdout data for this exact model
version (`app/data/processed/disease_symptom_v1/disease-symptom-v1-7414d5132a47/`
in the source project — train.parquet + test.parquet, 304 rows total,
the same dataset_version metadata.json's own `dataset_version` field
names):

    EXPECTED_FEATURES_BY_DISEASE[rag_name] = (
        {every feature_schema.json column that was nonzero in AT LEAST
         ONE real training/holdout row for that XGBoost class}
        INTERSECTED WITH
        ml.feature_mapper.mapped_features()
    )

A stricter "present in EVERY row for this disease" (set intersection
across rows, rather than union) was tried first and rejected: with only
5-10 examples per class, it came back EMPTY for 8 of the 13 diseases —
this dataset's rows are evidently distinct symptom-subset "cases" per
disease, not variations on one fixed core pattern, so no single feature
is common to all of them. Union-then-intersect-with-mappable is the
looser, still-defensible standard: "has this disease's real data ever
actually used this column, and can Healix_chatbot's Arabic vocabulary
even represent it" — not merely mappable in the abstract, but mappable
AND actually part of this specific disease's real signal.

This is a hand-reviewed, frozen constant, not re-derived from the raw
parquet files at runtime — same "explicit, reviewed, no live
re-derivation" discipline as ml.disease_crosswalk.DISEASE_CROSSWALK. The
derivation script is not part of this package; re-run it against the
same dataset_version if the vendored model bundle or feature_mapper's
coverage ever changes, and update the sets below by hand, reviewed, same
as any other change to a reviewed constant.

--- The gate itself ---

MIN_REQUIRED_MATCHED = 2 — the same absolute-count floor and the same
reasoning as nodes.rag_retrieve.MIN_MATCHED_SYMPTOMS: a single matched
feature, even a disease-relevant one, was already shown elsewhere in
this project (rag_retrieve's own hypertension counterexample) to be too
weak a basis on its own. Applying the identical, already-reviewed
threshold here — rather than inventing a new number — is deliberate.

One disease structurally can never clear this floor with the CURRENT
feature_mapper: Urinary Tract Infection's real training data only ever
uses ONE mappable feature (burning_micturition — bladder_discomfort,
continuous_feel_of_urine, and foul_smell_of_urine all have no Arabic
vocabulary equivalent yet). This is flagged here rather than silently
discovered later: nodes.ml_corroborate will never emit a signal for a
Urinary Tract Infection candidate until vocabulary/symptoms.py grows a
second UTI-relevant term feature_mapper.py can pick up. That is a
correct, expected "no signal" outcome (CLAUDE.md > Known limitations),
not a bug to work around here.
"""

from __future__ import annotations

MIN_REQUIRED_MATCHED = 2

# RAG disease name (ml.disease_crosswalk.DISEASE_CROSSWALK's keys) -> the
# set of XGBoost feature_schema.json columns this disease's real training
# data actually used AND ml.feature_mapper can represent. See module
# docstring for exact derivation.
EXPECTED_FEATURES_BY_DISEASE: dict[str, frozenset[str]] = {
    "Asthma": frozenset({"breathlessness", "cough", "fatigue", "high_fever"}),
    "Chickenpox": frozenset(
        {
            "fatigue",
            "headache",
            "high_fever",
            "itching",
            "loss_of_appetite",
            "mild_fever",
            "skin_rash",
            "swelled_lymph_nodes",
        }
    ),
    "Community-Acquired Pneumonia": frozenset(
        {"breathlessness", "chest_pain", "cough", "fatigue", "high_fever", "phlegm", "sweating"}
    ),
    "Gastroesophageal Reflux Disease": frozenset(
        {"acidity", "chest_pain", "cough", "ulcers_on_tongue", "vomiting"}
    ),
    "Hepatitis A": frozenset(
        {
            "abdominal_pain",
            "dark_urine",
            "diarrhoea",
            "joint_pain",
            "loss_of_appetite",
            "mild_fever",
            "muscle_pain",
            "nausea",
            "vomiting",
        }
    ),
    "Hypertension": frozenset({"chest_pain", "dizziness", "headache", "loss_of_balance"}),
    "Impetigo": frozenset({"high_fever", "skin_rash", "yellow_crust_ooze"}),
    "Migraine": frozenset(
        {"acidity", "blurred_and_distorted_vision", "headache", "irritability", "stiff_neck"}
    ),
    "Peptic Ulcer Disease": frozenset({"abdominal_pain", "loss_of_appetite", "vomiting"}),
    "Typhoid Fever": frozenset(
        {
            "abdominal_pain",
            "constipation",
            "diarrhoea",
            "fatigue",
            "headache",
            "high_fever",
            "nausea",
            "vomiting",
        }
    ),
    # Structurally can never clear MIN_REQUIRED_MATCHED=2 today — see
    # module docstring. Kept as a real (not padded) 1-element set rather
    # than removing the disease from this dict entirely, so a future
    # vocabulary addition that adds a second UTI-mappable feature is a
    # one-line change here, not a rediscovery of this whole analysis.
    "Urinary Tract Infection": frozenset({"burning_micturition"}),
    "Benign Paroxysmal Positional Vertigo": frozenset(
        {"headache", "loss_of_balance", "nausea", "spinning_movements", "vomiting"}
    ),
    "Acute Gastroenteritis": frozenset({"diarrhoea", "vomiting"}),
}


def clears_density_floor(rag_disease_name: str, active_features: frozenset[str]) -> bool:
    """True if `active_features` (the built feature vector's nonzero
    columns) contains at least MIN_REQUIRED_MATCHED of the features this
    disease's real training data is known to use. False for a disease
    with no entry here at all (not in the crosswalk, or the crosswalk
    entry has since been removed) — never a KeyError."""
    expected = EXPECTED_FEATURES_BY_DISEASE.get(rag_disease_name)
    if not expected:
        return False
    return len(active_features & expected) >= MIN_REQUIRED_MATCHED
