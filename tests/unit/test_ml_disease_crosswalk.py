from ml.disease_crosswalk import XGBOOST_COVERED_DISEASES, xgboost_label_for
from ml.model_loader import load_bundle
from rag.schema import load_all


def test_every_covered_disease_is_a_real_knowledge_base_disease_name():
    real_names = {entry.name for entry in load_all()}
    for rag_name in XGBOOST_COVERED_DISEASES:
        assert rag_name in real_names, f"{rag_name!r} is not a real rag/knowledge_base/ name"


def test_every_covered_disease_is_a_real_xgboost_class_label():
    real_labels = set(load_bundle().label_encoder.classes_)
    for rag_name in XGBOOST_COVERED_DISEASES:
        assert rag_name in real_labels, f"{rag_name!r} is not a real XGBoost class label"


def test_coverage_is_exactly_the_bundles_real_class_set():
    # This bundle was trained directly on rag/knowledge_base/ names as
    # class labels, so coverage should equal the label encoder exactly,
    # not merely be a subset of it.
    real_labels = set(load_bundle().label_encoder.classes_)
    assert XGBOOST_COVERED_DISEASES == real_labels


def test_covers_all_forty_nine_rag_diseases():
    real_names = {entry.name for entry in load_all()}
    assert XGBOOST_COVERED_DISEASES == real_names


def test_xgboost_label_for_returns_none_for_a_disease_with_no_coverage():
    assert xgboost_label_for("Not A Real Disease") is None


def test_xgboost_label_for_returns_the_same_name_identity():
    assert xgboost_label_for("Hypertension") == "Hypertension"
    assert xgboost_label_for("Peptic Ulcer Disease") == "Peptic Ulcer Disease"
