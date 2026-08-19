from ml.density_floor import (
    EXPECTED_FEATURES_BY_DISEASE,
    MIN_REQUIRED_MATCHED,
    clears_density_floor,
)
from ml.disease_crosswalk import XGBOOST_COVERED_DISEASES
from ml.feature_mapper import mapped_features
from ml.model_loader import load_bundle


def test_every_disease_here_is_a_real_covered_disease():
    for rag_name in EXPECTED_FEATURES_BY_DISEASE:
        assert rag_name in XGBOOST_COVERED_DISEASES


def test_every_covered_disease_has_a_density_floor_entry():
    # No silent gap — every covered disease was actually analyzed.
    for rag_name in XGBOOST_COVERED_DISEASES:
        assert rag_name in EXPECTED_FEATURES_BY_DISEASE


def test_every_expected_feature_is_a_real_feature_schema_column():
    feature_order = set(load_bundle().feature_order)
    for features in EXPECTED_FEATURES_BY_DISEASE.values():
        assert features <= feature_order


def test_every_expected_feature_is_actually_mappable_from_arabic():
    mapped = mapped_features()
    for disease, features in EXPECTED_FEATURES_BY_DISEASE.items():
        assert features <= mapped, f"{disease!r} references a feature feature_mapper can't set"


def test_min_required_matched_is_two_same_as_rag_retrieve_precedent():
    assert MIN_REQUIRED_MATCHED == 2


def test_no_disease_is_structurally_stuck_below_the_floor():
    # Unlike the earlier bundle (Urinary Tract Infection had exactly one
    # mappable feature and could never clear MIN_REQUIRED_MATCHED=2), this
    # bundle was trained entirely on canonical-vocabulary terms, so every
    # disease has at least MIN_REQUIRED_MATCHED expected features.
    for disease, features in EXPECTED_FEATURES_BY_DISEASE.items():
        assert len(features) >= MIN_REQUIRED_MATCHED, disease


def test_clears_density_floor_true_when_enough_overlap():
    active = frozenset({"دوخه", "صداع", "غثيان"})
    assert clears_density_floor("Hypertension", active) is True


def test_clears_density_floor_false_with_only_one_matching_feature():
    active = frozenset({"صداع", "سعال", "تقيؤ"})  # only "صداع" overlaps Hypertension
    assert clears_density_floor("Hypertension", active) is False


def test_clears_density_floor_false_for_a_disease_with_no_entry():
    assert clears_density_floor("Not A Real Disease", frozenset({"حكه"})) is False


def test_urinary_tract_infection_now_clears_the_floor():
    # Documented improvement over the earlier bundle (that module's old
    # docstring): this bundle's UTI has four mappable features, so a
    # realistic two-symptom overlap now clears the gate.
    assert clears_density_floor(
        "Urinary Tract Infection", frozenset({"تبول متكرر", "حرقه عند التبول"})
    ) is True
