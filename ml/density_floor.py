"""density_floor: per-disease minimum feature-density gate for
nodes.ml_corroborate — the fix for the sparse-input problem found while
auditing this integration (see CLAUDE.md's XGBoost corroboration-signal
section for the full writeup).

--- Why this exists ---

The XGBoost model needs a feature vector reasonably close to its own
training-time pattern to say anything reliable. An arbitrary
probability-magnitude threshold calibrated against dense rows would
misrepresent what the model does on the sparse input this integration
realistically produces (CLAUDE.md, and the user's own decision: no
probability-magnitude thresholds anywhere in this feature).

This module answers a narrower, structural question instead: for THIS
disease, does the built feature vector contain enough of the columns
that disease's real training examples actually used, for
nodes.ml_corroborate to even bother asking the model?

--- Provenance — how EXPECTED_FEATURES_BY_DISEASE was derived ---

Exported directly by ml/training/disease_symptom_checklist_training.ipynb
at training time, from the actual generated feature matrix for
ml/models/xgboost-healix-arabic-v1.0/ (density_floor_export.json in that
bundle directory):

    EXPECTED_FEATURES_BY_DISEASE[rag_name] = every feature_schema.json
        column that was nonzero in AT LEAST ONE generated training/test
        row for that disease's class.

Every column here is, by construction, one of vocabulary/symptoms.py's
own canonical (normalized) terms — the notebook's filter_to_canonical()
step guarantees this, unlike the previous bundle where only 52/131
columns were ever Arabic-mappable. There is no "structurally can never
clear the floor" disease in this bundle: every one of the 49 diseases has
between 2 and 8 expected features (all >= MIN_REQUIRED_MATCHED).

This is copied from that export as a frozen constant, not re-read from
the JSON file at runtime — same "explicit, reviewed, no live
re-derivation on the hot path" discipline as ml.disease_crosswalk. Re-run
the notebook and refresh this dict by hand, reviewed, if the vendored
bundle is ever retrained.

--- The gate itself ---

MIN_REQUIRED_MATCHED = 2 — the same absolute-count floor and the same
reasoning as nodes.rag_retrieve.MIN_MATCHED_SYMPTOMS: a single matched
feature, even a disease-relevant one, was already shown elsewhere in this
project (rag_retrieve's own hypertension counterexample) to be too weak a
basis on its own. Applying the identical, already-reviewed threshold here
— rather than inventing a new number — is deliberate.
"""

from __future__ import annotations

MIN_REQUIRED_MATCHED = 2

# RAG disease name (ml.disease_crosswalk.XGBOOST_COVERED_DISEASES's
# members) -> the set of normalized, feature_schema.json columns this
# disease's real (synthetic, canonical-vocabulary) training data actually
# used. See module docstring for exact derivation.
EXPECTED_FEATURES_BY_DISEASE: dict[str, frozenset[str]] = {
    "Acute Bronchitis": frozenset({"بلغم", "تعب", "حمى خفيفه", "سعال", "ضيق بالصدر"}),
    "Acute Gastroenteritis": frozenset({"اسهال", "الم بطن", "تقيؤ", "حمى", "غثيان"}),
    "Acute Musculoskeletal Strain": frozenset({"الم موضعي عضلي", "تيبس", "صعوبه حركه"}),
    "Acute Otitis Media": frozenset({"الم اذن", "حمى", "فقدان سمع"}),
    "Acute Sinusitis": frozenset(
        {"احتقان انف", "افرازات انفيه", "الم بالوجه", "صداع", "غثيان", "فقدان حاسه الشم"}
    ),
    "Allergic Rhinitis": frozenset(
        {"احتقان انف", "حكه بالعين", "دموع زائده", "سيلان انف", "عطس متكرر"}
    ),
    "Asthma": frozenset({"ازيز صدر", "سعال", "ضيق بالصدر", "ضيق تنفس"}),
    "Atopic Dermatitis": frozenset({"احمرار", "جفاف الجلد", "حكه", "طفح جلدي"}),
    "Bacterial Vaginosis": frozenset({"افرازات مهبليه", "حكه مهبليه", "رائحه مهبليه كريهه"}),
    "Benign Paroxysmal Positional Vertigo": frozenset({"دوار", "دوخه", "غثيان"}),
    "Chickenpox": frozenset({"تعب", "حكه", "حمى", "طفح جلدي"}),
    "Community-Acquired Pneumonia": frozenset(
        {"الم في الصدر", "تعب", "حمى", "سعال", "ضيق تنفس"}
    ),
    "Conjunctivitis": frozenset(
        {"احمرار", "احمرار العين", "افرازات عينيه", "حكه", "حكه بالعين", "دموع زائده"}
    ),
    "Cutaneous Leishmaniasis": frozenset({"تقرح جلدي", "طفح جلدي"}),
    "Dysmenorrhea": frozenset({"الم اسفل البطن", "الم اسفل الظهر", "غثيان"}),
    "Gastroesophageal Reflux Disease": frozenset(
        {"ارتجاع حمضي", "حرقه في الصدر", "صعوبه بلع", "طعم حامض او مر بالفم"}
    ),
    "Gout": frozenset({"احمرار", "الم مفاصل شديد ومفاجئ", "تورم مفاصل"}),
    "Hand, Foot, and Mouth Disease": frozenset({"التهاب حلق", "تقرحات الفم", "حمى", "طفح جلدي"}),
    "Hepatitis A": frozenset({"الم اعلى البطن", "بول داكن", "تعب", "غثيان", "يرقان"}),
    "Herpes Zoster": frozenset({"حرقان او وخز بالجلد", "حساسيه للضوء", "حمى", "صداع", "طفح جلدي"}),
    "Hypertension": frozenset({"دوخه", "صداع"}),
    "Impetigo": frozenset({"احمرار", "حكه", "حمى", "طفح جلدي", "قشور عسليه اللون على الجلد"}),
    "Infectious Mononucleosis": frozenset({"الم حلق شديد", "تعب شديد", "تورم غدد الرقبه", "حمى"}),
    "Influenza": frozenset(
        {"التهاب حلق", "الم عضلي", "تعب", "حمى", "سعال", "سيلان انف", "صداع", "فقدان الوعي"}
    ),
    "Iron Deficiency Anaemia": frozenset(
        {"الم في الصدر", "تعب", "خفقان القلب", "دوار", "دوخه", "شحوب", "ضيق تنفس"}
    ),
    "Irritable Bowel Syndrome": frozenset({"اسهال", "الم بطن معمم", "امساك", "انتفاخ"}),
    "Kidney Stones": frozenset({"الم شديد بالخاصره", "تقيؤ", "دم في البول", "غثيان"}),
    "Measles": frozenset({"احمرار العين", "بقع بيضاء بالفم", "حمى", "سعال", "طفح جلدي"}),
    "Migraine": frozenset({"حساسيه للصوت", "حساسيه للضوء", "صداع نابض من جهه واحده", "غثيان"}),
    "Mumps": frozenset({"الم عضلي", "تورم الغدد اللعابيه", "حمى", "صداع"}),
    "Otitis Externa": frozenset({"احمرار", "افرازات من الاذن", "الم اذن", "حكه بالاذن"}),
    "Pediculosis Capitis": frozenset({"احساس بحركه بفروه الراس", "حكه فروه الراس"}),
    "Peptic Ulcer Disease": frozenset({"الم اعلى البطن", "انتفاخ", "غثيان", "فقدان شهيه"}),
    "Pinworm Infection": frozenset({"اضطراب نوم", "تهيج", "حكه شرجيه"}),
    "Polycystic Ovary Syndrome": frozenset(
        {"اضطراب الدوره الشهريه", "حب الشباب", "زياده وزن", "نمو شعر زائد"}
    ),
    "Rheumatoid Arthritis": frozenset({"الم مفاصل", "تعب", "تورم مفاصل", "تيبس صباحي"}),
    "Roseola": frozenset({"حمى مرتفعه مفاجئه", "طفح جلدي"}),
    "Rubella": frozenset({"الم مفاصل", "تورم غدد خلف الاذن", "حمى خفيفه", "طفح جلدي"}),
    "Scabies": frozenset({"حكه شديده ليليه", "طفح جلدي"}),
    "Streptococcal Pharyngitis": frozenset(
        {"الم حلق شديد ومفاجئ", "تورم غدد الرقبه", "حمى", "صعوبه بلع"}
    ),
    "Tendinitis": frozenset({"الم موضعي عند الحركه", "تورم موضعي", "تيبس"}),
    "Tension-Type Headache": frozenset({"تعب", "تيبس", "صداع ضاغط من الجهتين"}),
    "Tonsillitis": frozenset({"الم حلق شديد", "تورم اللوزتين", "حمى", "صعوبه بلع"}),
    "Type 2 Diabetes": frozenset(
        {"تبول متكرر", "تشوش رؤيه", "تعب", "عطش شديد", "نقص وزن غير مبرر"}
    ),
    "Typhoid Fever": frozenset({"الم بطن", "تعب", "حمى", "صداع"}),
    "Urinary Tract Infection": frozenset({"الم اسفل البطن", "تبول متكرر", "حرقه عند التبول", "حمى"}),
    "Urticaria": frozenset({"تورم موضعي", "حكه", "طفح جلدي"}),
    "Vaginal Candidiasis": frozenset({"افرازات مهبليه", "حرقه عند التبول", "حكه مهبليه"}),
    "Viral Pharyngitis": frozenset({"التهاب حلق", "حمى خفيفه", "سعال", "سيلان انف"}),
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
