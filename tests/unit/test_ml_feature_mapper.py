from ml.feature_mapper import (
    SYMPTOM_TO_FEATURE,
    build_feature_vector,
    mapped_features,
    unmapped_features,
)
from ml.model_loader import load_bundle


def _symptom(name):
    return {"name": name}


def _state(*, symptoms=(), negated_symptoms=()):
    return {
        "symptoms": [_symptom(n) for n in symptoms],
        "negated_symptoms": [_symptom(n) for n in negated_symptoms],
    }


def test_every_mapped_feature_is_a_real_feature_schema_column():
    feature_order = load_bundle().feature_order
    for feature in SYMPTOM_TO_FEATURE.values():
        assert feature in feature_order, f"{feature!r} is not a real feature_schema.json column"


def test_no_two_distinct_arabic_terms_normalize_to_the_same_key_with_different_targets():
    from rules.crisis import normalize

    seen: dict[str, str] = {}
    for term, feature in SYMPTOM_TO_FEATURE.items():
        normalized = normalize(term)
        if normalized in seen:
            assert seen[normalized] == feature, (
                f"{term!r} normalizes to a key already claimed by a different "
                f"feature ({seen[normalized]!r} vs {feature!r})"
            )
        seen[normalized] = feature


def test_unmapped_features_is_computed_not_hand_maintained():
    feature_order = load_bundle().feature_order
    unmapped = unmapped_features(feature_order)

    assert unmapped == frozenset(feature_order) - mapped_features()
    # Known always-unmapped examples (module docstring) — pinned so a
    # future edit that accidentally maps one of these is caught.
    assert "extra_marital_contacts" in unmapped
    assert "silver_like_dusting" in unmapped
    assert "belly_pain" in unmapped
    assert "stomach_pain" in unmapped


def test_build_feature_vector_sets_only_the_mapped_columns_present():
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(_state(symptoms=["صداع", "سعال"]), feature_order)

    active = {feature_order[i] for i, v in enumerate(vector) if v == 1.0}
    assert active == {"headache", "cough"}
    assert len(vector) == len(feature_order)


def test_build_feature_vector_never_sets_a_column_outside_the_schema():
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(_state(symptoms=["صداع"]), feature_order)

    assert len(vector) == len(feature_order)
    assert all(v in (0.0, 1.0) for v in vector)


def test_build_feature_vector_ignores_a_symptom_with_no_mapping():
    feature_order = load_bundle().feature_order
    # "نمو شعر زائد" (excess hair growth) is deliberately unmapped.
    vector = build_feature_vector(_state(symptoms=["نمو شعر زائد"]), feature_order)

    assert sum(vector) == 0.0


def test_build_feature_vector_negation_suppresses_the_feature():
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(
        _state(symptoms=["صداع"], negated_symptoms=["صداع"]), feature_order
    )

    active = {feature_order[i] for i, v in enumerate(vector) if v == 1.0}
    assert "headache" not in active


def test_build_feature_vector_convergence_two_arabic_terms_set_the_same_column():
    # "صداع" (plain) and "صداع نابض من جهة واحدة" (migraine-type) both
    # converge on "headache" — see module docstring's convergence note.
    feature_order = load_bundle().feature_order
    vector = build_feature_vector(
        _state(symptoms=["صداع نابض من جهة واحدة"]), feature_order
    )

    active = {feature_order[i] for i, v in enumerate(vector) if v == 1.0}
    assert active == {"headache"}
