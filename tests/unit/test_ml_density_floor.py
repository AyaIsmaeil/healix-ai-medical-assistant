from ml.density_floor import (
    EXPECTED_FEATURES_BY_DISEASE,
    MIN_REQUIRED_MATCHED,
    clears_density_floor,
)
from ml.disease_crosswalk import DISEASE_CROSSWALK
from ml.feature_mapper import mapped_features
from ml.model_loader import load_bundle


def test_every_disease_here_is_a_real_crosswalk_entry():
    for rag_name in EXPECTED_FEATURES_BY_DISEASE:
        assert rag_name in DISEASE_CROSSWALK


def test_every_crosswalk_disease_has_a_density_floor_entry():
    # No silent gap — every crosswalked disease was actually analyzed,
    # even if (like UTI) its resulting set is tiny.
    for rag_name in DISEASE_CROSSWALK:
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


def test_clears_density_floor_true_when_enough_overlap():
    active = frozenset({"chest_pain", "dizziness", "headache"})
    assert clears_density_floor("Hypertension", active) is True


def test_clears_density_floor_false_with_only_one_matching_feature():
    active = frozenset({"headache", "cough", "nausea"})  # only "headache" overlaps Hypertension
    assert clears_density_floor("Hypertension", active) is False


def test_clears_density_floor_false_for_a_disease_with_no_entry():
    assert clears_density_floor("Dysmenorrhea", frozenset({"itching"})) is False


def test_urinary_tract_infection_can_never_clear_the_floor_today():
    # Documented, expected limitation (module docstring) — pinned so a
    # silent change to feature_mapper coverage is caught, not missed.
    assert len(EXPECTED_FEATURES_BY_DISEASE["Urinary Tract Infection"]) == 1
    assert clears_density_floor(
        "Urinary Tract Infection", frozenset({"burning_micturition"})
    ) is False
