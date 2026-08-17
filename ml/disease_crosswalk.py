"""disease_crosswalk: the ONLY place a rag/knowledge_base/ disease name is
paired with an XGBoost class label.

Hand-authored, no fuzzy/semantic matching — same "explicit list, never
guessed" discipline as vocabulary/symptoms.py. Every pair below was
verified directly against the real vendored bundle during this feature's
audit (loaded label_encoder.classes_ directly, not inferred from training
code), and against the real rag/knowledge_base/*.json `name` fields — see
tests/unit/test_ml_disease_crosswalk.py, which re-verifies both sides
against the live files rather than trusting this dict blindly.

Only 13 of the RAG knowledge base's 49 diseases have a class in the
XGBoost model's 41-class space that is unambiguously the same condition.
Three further candidate pairs were investigated and deliberately
EXCLUDED as too ambiguous to trust for a doctor-facing signal:

  - Type 2 Diabetes <-> "Diabetes": the source dataset never
    distinguishes Type 1 from Type 2 (confirmed: no "Type 2 Diabetes"
    class exists, only the generic "Diabetes").
  - Allergic Rhinitis <-> "Allergy": too broad — the model's "Allergy"
    class is a generic allergic-reaction bucket, not specifically
    allergic rhinitis.
  - Rheumatoid Arthritis <-> "Arthritis": the model has a SEPARATE
    "Osteoarthristis" class, confirming "Arthritis" means something
    else in this dataset, not rheumatoid arthritis specifically.

A wrong pairing here is worse than a missing one: it would make
nodes.ml_corroborate silently corroborate (or fail to corroborate) the
wrong disease. See CLAUDE.md's XGBoost corroboration-signal section for
the full audit.

Several XGBoost-side labels carry real spelling quirks from the source
dataset (a typo, a lowercase leading letter, a double space) — these are
reproduced VERBATIM below because they are the actual values
label_encoder.classes_ returns; "fixing" the spelling here would just
make the lookup silently fail.
"""

from __future__ import annotations

# RAG knowledge base `name` (rag.schema.KnowledgeBaseEntry.name, exactly as
# it appears in rag/knowledge_base/*.json) -> XGBoost class label (exactly
# as ml.model_loader.load_bundle().label_encoder.classes_ returns it).
DISEASE_CROSSWALK: dict[str, str] = {
    "Asthma": "Bronchial Asthma",
    "Chickenpox": "Chicken pox",
    "Community-Acquired Pneumonia": "Pneumonia",
    "Gastroesophageal Reflux Disease": "GERD",
    "Hepatitis A": "hepatitis A",
    "Hypertension": "Hypertension",
    "Impetigo": "Impetigo",
    "Migraine": "Migraine",
    "Peptic Ulcer Disease": "Peptic ulcer diseae",  # sic — typo in the source dataset
    "Typhoid Fever": "Typhoid",
    "Urinary Tract Infection": "Urinary tract infection",
    "Benign Paroxysmal Positional Vertigo": "(vertigo) Paroymsal  Positional Vertigo",  # sic
    "Acute Gastroenteritis": "Gastroenteritis",
}


def xgboost_label_for(rag_disease_name: str) -> str | None:
    """The XGBoost class label for a RAG disease name, or None if this
    disease has no crosswalk entry (the common case — 36 of 49 RAG
    diseases have none, by design, not by omission)."""
    return DISEASE_CROSSWALK.get(rag_disease_name)
