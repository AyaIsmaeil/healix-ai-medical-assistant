"""disease_crosswalk: the ONLY place a rag/knowledge_base/ disease name is
confirmed as having a class in the vendored XGBoost bundle.

--- Identity, not translation ---

ml/models/xgboost-healix-arabic-v1.0/ was trained directly on
rag/knowledge_base/*.json's own `name` fields as class labels (see
ml/training/disease_symptom_checklist_training.ipynb) — there is no
separate, differently-labeled dataset to cross-reference anymore. All 49
RAG diseases have a class in this bundle; label_encoder.classes_ is
verified identical to {entry.name for entry in rag.schema.load_all()} by
tests/unit/test_ml_disease_crosswalk.py, which loads the real bundle and
the real knowledge base rather than trusting this list blindly.

XGBOOST_COVERED_DISEASES is still a hand-typed, frozen constant — not a
live label_encoder.classes_ read on the hot path — same "explicit,
reviewed, no live re-derivation" discipline as
ml.density_floor.EXPECTED_FEATURES_BY_DISEASE. Keeping it frozen also
keeps nodes.ml_corroborate's early exit ("no candidate has ML coverage,
never load the 6MB bundle") cheap.

The earlier bundle covered only 13 of 49 diseases and needed three
ambiguous pairs (Type 2 Diabetes/"Diabetes", Allergic
Rhinitis/"Allergy", Rheumatoid Arthritis/"Arthritis") deliberately
excluded as unsafe guesses. That problem doesn't exist here: every class
label IS a RAG disease name, verbatim, so there is no pairing left to get
wrong.
"""

from __future__ import annotations

# rag/knowledge_base/*.json `name` values this XGBoost bundle has a class
# for — currently all 49 (100% coverage; see module docstring).
XGBOOST_COVERED_DISEASES: frozenset[str] = frozenset(
    {
        "Acute Bronchitis",
        "Acute Gastroenteritis",
        "Acute Musculoskeletal Strain",
        "Acute Otitis Media",
        "Acute Sinusitis",
        "Allergic Rhinitis",
        "Asthma",
        "Atopic Dermatitis",
        "Bacterial Vaginosis",
        "Benign Paroxysmal Positional Vertigo",
        "Chickenpox",
        "Community-Acquired Pneumonia",
        "Conjunctivitis",
        "Cutaneous Leishmaniasis",
        "Dysmenorrhea",
        "Gastroesophageal Reflux Disease",
        "Gout",
        "Hand, Foot, and Mouth Disease",
        "Hepatitis A",
        "Herpes Zoster",
        "Hypertension",
        "Impetigo",
        "Infectious Mononucleosis",
        "Influenza",
        "Iron Deficiency Anaemia",
        "Irritable Bowel Syndrome",
        "Kidney Stones",
        "Measles",
        "Migraine",
        "Mumps",
        "Otitis Externa",
        "Pediculosis Capitis",
        "Peptic Ulcer Disease",
        "Pinworm Infection",
        "Polycystic Ovary Syndrome",
        "Rheumatoid Arthritis",
        "Roseola",
        "Rubella",
        "Scabies",
        "Streptococcal Pharyngitis",
        "Tendinitis",
        "Tension-Type Headache",
        "Tonsillitis",
        "Type 2 Diabetes",
        "Typhoid Fever",
        "Urinary Tract Infection",
        "Urticaria",
        "Vaginal Candidiasis",
        "Viral Pharyngitis",
    }
)


def xgboost_label_for(rag_disease_name: str) -> str | None:
    """The XGBoost class label for a RAG disease name — identical to
    `rag_disease_name` itself, since this bundle was trained on
    rag/knowledge_base/'s own names — or None if this disease has no
    class in the model."""
    return rag_disease_name if rag_disease_name in XGBOOST_COVERED_DISEASES else None
