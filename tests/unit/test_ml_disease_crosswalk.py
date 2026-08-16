from ml.disease_crosswalk import DISEASE_CROSSWALK, xgboost_label_for
from ml.model_loader import load_bundle
from rag.schema import load_all


def test_every_rag_side_key_is_a_real_knowledge_base_disease_name():
    real_names = {entry.name for entry in load_all()}
    for rag_name in DISEASE_CROSSWALK:
        assert rag_name in real_names, f"{rag_name!r} is not a real rag/knowledge_base/ name"


def test_every_xgboost_side_value_is_a_real_model_class_label():
    real_labels = set(load_bundle().label_encoder.classes_)
    for xgb_label in DISEASE_CROSSWALK.values():
        assert xgb_label in real_labels, f"{xgb_label!r} is not a real XGBoost class label"


def test_no_duplicate_xgboost_labels_across_different_rag_diseases():
    # Two different RAG diseases silently pointing at the same XGBoost
    # class would make the corroboration signal ambiguous about which
    # RAG candidate it actually supports.
    labels = list(DISEASE_CROSSWALK.values())
    assert len(labels) == len(set(labels))


def test_the_three_investigated_and_excluded_pairs_stay_excluded():
    # Confirms the exclusion was a real decision, not silently reversed
    # by a later edit. See module docstring for why each is ambiguous.
    assert "Type 2 Diabetes" not in DISEASE_CROSSWALK
    assert "Allergic Rhinitis" not in DISEASE_CROSSWALK
    assert "Rheumatoid Arthritis" not in DISEASE_CROSSWALK


def test_crosswalk_has_exactly_thirteen_pairs():
    assert len(DISEASE_CROSSWALK) == 13


def test_xgboost_label_for_returns_none_for_a_disease_with_no_crosswalk_entry():
    assert xgboost_label_for("Dysmenorrhea") is None


def test_xgboost_label_for_returns_the_mapped_label():
    assert xgboost_label_for("Hypertension") == "Hypertension"
    assert xgboost_label_for("Peptic Ulcer Disease") == "Peptic ulcer diseae"
