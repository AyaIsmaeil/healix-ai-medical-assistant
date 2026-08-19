from rules.crisis import normalize
from ml.feature_mapper import (
    build_feature_vector,
    mapped_features,
    unmapped_features,
)
from ml.model_loader import load_bundle
from vocabulary.symptoms import CANONICAL_SYMPTOMS


def _symptom(name):
    return {"name": name}


def _state(*, symptoms=(), negated_symptoms=()):
    return {
        "symptoms": [_symptom(n) for n in symptoms],
        "negated_symptoms": [_symptom(n) for n in negated_symptoms],
    }


def test_mapped_features_is_the_full_canonical_vocabulary():
    assert mapped_features() == CANONICAL_SYMPTOMS


def test_every_feature_schema_column_is_a_canonical_symptom():
    # The bundle was trained by filtering to canonical terms only — every
    # column must be representable, unlike the earlier English-Kaggle
    # bundle where only 52/131 columns ever were.
    feature_order = load_bundle().feature_order
    for column in feature_order:
        assert column in CANONICAL_SYMPTOMS, f"{column!r} is not a canonical vocabulary term"


def test_unmapped_features_is_empty_for_the_current_bundle():
    feature_order = load_bundle().feature_order
    assert unmapped_features(feature_order) == frozenset()


def test_build_feature_vector_sets_only_the_columns_present():
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(_state(symptoms=["صداع", "سعال"]), feature_order)

    active = {feature_order[i] for i, v in enumerate(vector) if v == 1.0}
    assert active == {normalize("صداع"), normalize("سعال")}
    assert len(vector) == len(feature_order)


def test_build_feature_vector_never_sets_a_column_outside_the_schema():
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(_state(symptoms=["صداع"]), feature_order)

    assert len(vector) == len(feature_order)
    assert all(v in (0.0, 1.0) for v in vector)


def test_build_feature_vector_ignores_a_symptom_with_no_matching_column():
    feature_order = load_bundle().feature_order
    # A canonical term this particular bundle never used as a training
    # column (present in the vocabulary, absent from feature_order).
    unused = next(
        term for term in CANONICAL_SYMPTOMS if term not in set(feature_order)
    )
    vector = build_feature_vector(_state(symptoms=[unused]), feature_order)

    assert sum(vector) == 0.0


def test_build_feature_vector_negation_suppresses_the_feature():
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(
        _state(symptoms=["صداع"], negated_symptoms=["صداع"]), feature_order
    )

    active = {feature_order[i] for i, v in enumerate(vector) if v == 1.0}
    assert normalize("صداع") not in active
