"""feature_mapper: translates Healix_chatbot's confirmed Arabic symptoms
into the 131-column binary feature vector ml/model_loader's XGBoost
bundle expects.

Hand-authored, one Arabic canonical term (vocabulary/symptoms.py) at a
time — never a bulk/automated translation pass. This is real bilingual
clinical work, not mechanical wiring, and is flagged here (and in
CLAUDE.md) as needing a further human bilingual/clinical review pass
before this signal is trusted in anything beyond the narrow, doctor-only,
non-decision-making role nodes.ml_corroborate gives it.

--- Coverage: 52 of 131 columns, not all of them ---

Roughly 60% of the model's columns have NO defensible Arabic equivalent
and are deliberately left unmapped, never guessed — same "an absent entry
fails loudly, a wrong one fails silently" principle CLAUDE.md >
Vocabulary provenance and growth already applies to
vocabulary/symptoms.py itself. Three distinct reasons a column stays
unmapped, all real, not oversights:

  1. Risk-factor / history fields that are not symptoms at all —
     extra_marital_contacts, family_history, history_of_alcohol_consumption,
     receiving_blood_transfusion, receiving_unsterile_injections. Healix_chatbot
     extracts symptoms from patient-reported complaints, not exposure history.
  2. Hyper-specific dermatologic/nail findings with no vocabulary
     counterpart — silver_like_dusting, small_dents_in_nails,
     pus_filled_pimples, blackheads, brittle_nails, inflammatory_nails,
     nodal_skin_eruptions, dischromic_patches, red_sore_around_nose,
     toxic_look_typhos, and others in the same family.
  3. A genuine concept gap even for an ordinary symptom — e.g. no
     Arabic vocabulary entry exists yet for "ear pain", "wheezing",
     "vaginal discharge", or "blood in urine", even though these are
     everyday complaints. Closing this requires adding the term to
     vocabulary/symptoms.py through its own reviewed process (CLAUDE.md >
     Vocabulary provenance and growth), not inventing a mapping here.

UNMAPPED_FEATURES (below) is computed from feature_schema.json's real
column list minus what SYMPTOM_TO_FEATURE actually covers — not a
separately hand-maintained list that could drift out of sync.

--- Convergence: several Arabic terms can point at the same feature ---

The model's own vocabulary is coarser than Healix_chatbot's in places —
e.g. it has exactly one "headache" column, while vocabulary/symptoms.py
has four headache variants (plain, migraine-type, meningitis-severity,
tension-type) because those distinctions matter to rag/knowledge_base/'s
own disease matching. All four converge on the single "headache" column
here — losing a qualifier (severity, laterality, timing) the model has
no column for anyway is not a mapping error, it is the model's own
resolution limit. The one deliberate exception is fever: the model
itself encodes two distinct severity columns (high_fever / mild_fever),
so those two vocabulary entries are kept separate rather than merged —
mirrored on purpose. A handful of judgment calls in the table below are
commented individually where the reasoning is not obvious from the pair
alone.

--- What is deliberately NOT here ---

Several Arabic terms were investigated and deliberately left unmapped
because forcing a mapping would assert something the term does not
actually say:

  - Generic "يرقان" (jaundice) is not mapped to either
    yellowing_of_eyes or yellowish_skin — the model has no single
    generic jaundice column, and picking one of the two anatomical
    variants without the patient having specified which would be a
    guess, not a translation.
  - "حب الشباب" (acne) is not mapped to any of the model's several
    acne-adjacent columns (pus_filled_pimples, blackheads, scurring,
    skin_peeling) for the same reason — the model wants a specific
    sign, the vocabulary term is a general label.
  - "فقدان الوعي" (loss of consciousness) is not mapped to "coma" —
    coma is a specific, sustained state; conflating a transient loss of
    consciousness with it would overstate severity to the model.
  - Region-qualified pain terms (e.g. "ألم أسفل البطن", "ألم أعلى
    البطن", "ألم بطن معمم") all converge on the single generic
    "abdominal_pain" column, never on "belly_pain" or "stomach_pain" —
    the model provides three near-synonymous abdominal-pain columns
    with no documented semantic distinction between them, and guessing
    which of the other two a given regional phrase was "meant" for
    would be exactly the kind of invented mapping this module avoids.
    Those two columns simply stay unmapped.

--- rules.crisis.normalize, same as everywhere else ---

Keys are written in natural Arabic spelling for readability (matching
vocabulary/symptoms.py's own convention) and normalized at import into
_SYMPTOM_TO_FEATURE_N — CLAUDE.md > Symptom vocabulary > "Compare against
the normalized form, never the authored spelling" applies here exactly
as it does in nodes/rag_retrieve.py.
"""

from __future__ import annotations

from rules.crisis import normalize
from state import HealixState, Symptom
from vocabulary.symptoms import is_canonical

# Arabic canonical term (vocabulary/symptoms.py spelling) -> XGBoost
# feature_schema.json column name. See module docstring for the
# convergence/exclusion reasoning behind the choices below.
SYMPTOM_TO_FEATURE: dict[str, str] = {
    # --- acs_chest_pain / stroke_fast / bacterial_meningitis / gi_bleed /
    # pulmonary_embolism / dka red-flag groups (vocabulary/symptoms.py) ---
    "ألم في الصدر": "chest_pain",
    "تعرق غزير": "sweating",
    "تقيؤ": "vomiting",
    "ضيق تنفس": "breathlessness",
    "غثيان": "nausea",
    "تلعثم مفاجئ في الكلام": "slurred_speech",
    "ضعف مفاجئ في نصف الجسم": "weakness_of_one_body_side",
    "فقدان التوازن المفاجئ": "loss_of_balance",
    # High-fever variant only — see module docstring's fever paragraph;
    # generic "حمى" itself and "حمى خفيفة" (below) stay distinct.
    "حمى مرتفعة مفاجئة": "high_fever",
    "تغير مفاجئ في مستوى الوعي": "altered_sensorium",
    "تيبس الرقبة": "stiff_neck",
    "دم في البراز": "bloody_stool",
    # Unilateral-leg qualifier lost — the model has no laterality column
    # for leg swelling either way.
    "تورم في ساق واحدة": "swollen_legs",
    "خفقان القلب": "palpitations",
    "تبول متكرر": "polyuria",

    # --- rag/knowledge_base/ groups ---
    "ألم عضلي": "muscle_pain",
    "التهاب حلق": "throat_irritation",
    "تعب": "fatigue",
    "سعال": "cough",
    "سيلان أنف": "runny_nose",
    "صداع": "headache",
    "دوخة": "dizziness",
    "حرقة عند التبول": "burning_micturition",
    "ألم أسفل البطن": "abdominal_pain",  # region qualifier lost, see docstring
    "ألم حلق شديد ومفاجئ": "throat_irritation",
    "تورم غدد الرقبة": "swelled_lymph_nodes",
    "إسهال": "diarrhoea",
    "ألم بطن": "abdominal_pain",
    "نقص وزن غير مبرر": "weight_loss",
    "تشوش رؤية": "blurred_and_distorted_vision",
    "صداع نابض من جهة واحدة": "headache",  # migraine-specific qualifier lost
    "ألم مفاصل": "joint_pain",
    "تورم مفاصل": "swelling_joints",
    "تيبس صباحي": "movement_stiffness",  # "morning" qualifier lost
    "احتقان أنف": "congestion",
    "إفرازات أنفية": "runny_nose",
    "فقدان حاسة الشم": "loss_of_smell",
    "انتفاخ": "distention_of_abdomen",
    "إمساك": "constipation",
    "ألم بطن معمم": "abdominal_pain",
    "طفح جلدي": "skin_rash",
    "حكة": "itching",
    "احمرار العين": "redness_of_eyes",
    "حكة بالعين": "itching",  # region qualifier lost — model has one itching column
    "دموع زائدة": "watering_from_eyes",
    "دوار": "spinning_movements",  # vertigo
    "ألم حلق شديد": "throat_irritation",
    "تيبس": "movement_stiffness",
    "تعب شديد": "fatigue",
    "حكة شرجية": "irritation_in_anus",
    "تهيج": "irritability",
    "حمى خفيفة": "mild_fever",
    "حكة شديدة ليلية": "itching",
    "حكة مهبلية": "itching",
    "ألم أسفل الظهر": "back_pain",
    "اضطراب الدورة الشهرية": "abnormal_menstruation",
    "زيادة وزن": "weight_gain",
    "بول داكن": "dark_urine",
    "ألم أعلى البطن": "abdominal_pain",
    "ألم مفاصل شديد ومفاجئ": "joint_pain",
    "ألم موضعي عضلي": "muscle_pain",
    "صعوبة حركة": "movement_stiffness",
    "حرقة في الصدر": "acidity",
    "ارتجاع حمضي": "acidity",
    "بلغم": "phlegm",
    "حكة بالأذن": "itching",
    "عطس متكرر": "continuous_sneezing",
    "قشور عسلية اللون على الجلد": "yellow_crust_ooze",
    "حكة فروة الرأس": "itching",
    "تورم غدد خلف الأذن": "swelled_lymph_nodes",
    "فقدان شهية": "loss_of_appetite",
    "صداع ضاغط من الجهتين": "headache",  # tension-headache qualifier lost
    "تقرحات الفم": "ulcers_on_tongue",  # closest available; mouth vs tongue-specific
}

# Validated at import — same discipline rules/red_flags.py already
# applies to its own symptom references (CLAUDE.md > Symptom vocabulary):
# a typo here would otherwise silently never match anything.
for _arabic_term in SYMPTOM_TO_FEATURE:
    if not is_canonical(_arabic_term):
        raise ValueError(
            f"ml.feature_mapper.SYMPTOM_TO_FEATURE references {_arabic_term!r}, "
            "which is not in vocabulary/symptoms.py's canonical set."
        )

_SYMPTOM_TO_FEATURE_N: dict[str, str] = {
    normalize(term): feature for term, feature in SYMPTOM_TO_FEATURE.items()
}


def _normalized_names(symptoms: list[Symptom]) -> set[str]:
    # Mirrors nodes.rag_retrieve._normalized_names exactly — same
    # normalize-before-compare discipline, applied at this layer too.
    return {normalize(symptom["name"]) for symptom in symptoms if symptom.get("name")}


def build_feature_vector(state: HealixState, feature_order: tuple[str, ...]) -> list[float]:
    """A 0/1 vector in `feature_order`'s exact column order, built from
    state["symptoms"] minus state["negated_symptoms"] (negation
    suppresses a feature the same way it carries real evidentiary weight
    everywhere else in this project — CLAUDE.md > State).

    Deliberately simpler than nodes.rag_retrieve's matching: no
    generic/specific-sibling (_SUBSUMES) resolution here. That machinery
    exists to let a KB entry's generic term match a patient's more
    specific one; this module already does its own, coarser many-to-one
    convergence (module docstring) and layering _SUBSUMES on top would
    be a second, harder-to-review translation step for a signal that is
    already the least-trusted layer in this pipeline.
    """
    confirmed = _normalized_names(state.get("symptoms", []))
    negated = _normalized_names(state.get("negated_symptoms", []))
    effective = confirmed - negated

    active_features = {
        _SYMPTOM_TO_FEATURE_N[term] for term in effective if term in _SYMPTOM_TO_FEATURE_N
    }
    return [1.0 if feature in active_features else 0.0 for feature in feature_order]


def mapped_features() -> frozenset[str]:
    """Every XGBoost feature this module can ever set to 1. Used by
    ml.density_floor to know which of a disease's real training features
    are even representable from Arabic input."""
    return frozenset(SYMPTOM_TO_FEATURE.values())


def unmapped_features(feature_order: tuple[str, ...]) -> frozenset[str]:
    """Computed from the live feature order, not a second hand-maintained
    list — see module docstring's "Coverage" section."""
    return frozenset(feature_order) - mapped_features()
