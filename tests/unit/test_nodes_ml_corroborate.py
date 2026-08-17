import pytest

import nodes.ml_corroborate as ml_corroborate_module
from nodes.ml_corroborate import ml_corroborate


class _FakeModel:
    def __init__(self, proba_row):
        self.proba_row = proba_row
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(X)
        return [self.proba_row]


class _FakeLabelEncoder:
    def __init__(self, classes):
        self.classes_ = classes

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]


class _FakeBundle:
    def __init__(self, *, feature_order, proba_row, classes):
        self.feature_order = feature_order
        self.model = _FakeModel(proba_row)
        self.label_encoder = _FakeLabelEncoder(classes)


# Same order as ml.density_floor.EXPECTED_FEATURES_BY_DISEASE["Hypertension"]
# would expect an overlap against: chest_pain, dizziness, headache, loss_of_balance.
_CLASSES = ("Hypertension", "Migraine", "Bronchial Asthma")
_FEATURE_ORDER = ("chest_pain", "dizziness", "headache", "loss_of_balance", "cough", "nausea")


def _bundle_with_hypertension_as_argmax():
    # index 0 = Hypertension highest.
    return _FakeBundle(
        feature_order=_FEATURE_ORDER, proba_row=[0.7, 0.2, 0.1], classes=_CLASSES
    )


def _bundle_with_migraine_as_argmax():
    return _FakeBundle(
        feature_order=_FEATURE_ORDER, proba_row=[0.2, 0.7, 0.1], classes=_CLASSES
    )


def _symptom(name):
    return {"name": name}


def _state(*, symptoms=(), negated_symptoms=(), candidate_diseases=()):
    return {
        "symptoms": [_symptom(n) for n in symptoms],
        "negated_symptoms": [_symptom(n) for n in negated_symptoms],
        "candidate_diseases": list(candidate_diseases),
    }


def _candidate(name):
    return {
        "name": name,
        "name_ar": f"{name} (ar)",
        "match_score": 0.8,
        "matched_symptoms": [],
        "missing_symptoms": [],
        "negated_symptoms": [],
        "specialties": ["طب عام"],
    }


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Every test in this file stubs load_bundle explicitly via
    # monkeypatch.setattr(ml_corroborate_module, "load_bundle", ...);
    # this fixture just guarantees no test accidentally falls through to
    # the real 6MB bundle if it forgets to.
    yield


def test_mapped_candidate_clearing_both_gates_gets_the_signal(monkeypatch):
    monkeypatch.setattr(
        ml_corroborate_module, "load_bundle", _bundle_with_hypertension_as_argmax
    )
    state = _state(
        symptoms=["ألم في الصدر", "دوخة", "صداع"],  # chest_pain, dizziness, headache — 3 overlap
        candidate_diseases=[_candidate("Hypertension")],
    )

    result = ml_corroborate(state)

    assert result["candidate_diseases"][0]["ml_corroboration"] == "model_signal_present"


def test_candidate_with_no_crosswalk_entry_never_gets_a_signal(monkeypatch):
    load_bundle_calls = []
    monkeypatch.setattr(
        ml_corroborate_module,
        "load_bundle",
        lambda: load_bundle_calls.append(1) or _bundle_with_hypertension_as_argmax(),
    )
    state = _state(
        symptoms=["ألم أسفل الظهر"],
        candidate_diseases=[_candidate("Dysmenorrhea")],  # not in the crosswalk
    )

    result = ml_corroborate(state)

    assert result == {}
    assert not load_bundle_calls, "load_bundle must never be called with no crosswalk candidates"


def test_never_introduces_a_new_candidate(monkeypatch):
    monkeypatch.setattr(
        ml_corroborate_module, "load_bundle", _bundle_with_hypertension_as_argmax
    )
    state = _state(
        symptoms=["ألم في الصدر", "دوخة", "صداع"],
        candidate_diseases=[_candidate("Hypertension"), _candidate("Migraine")],
    )

    result = ml_corroborate(state)

    assert len(result["candidate_diseases"]) == 2
    assert {c["name"] for c in result["candidate_diseases"]} == {"Hypertension", "Migraine"}


def test_model_load_failure_fails_open(monkeypatch):
    def _raise():
        raise RuntimeError("bundle checksum mismatch")

    monkeypatch.setattr(ml_corroborate_module, "load_bundle", _raise)
    state = _state(
        symptoms=["ألم في الصدر", "دوخة", "صداع"],
        candidate_diseases=[_candidate("Hypertension")],
    )

    result = ml_corroborate(state)

    assert result == {}


def test_predict_proba_failure_fails_open(monkeypatch):
    class _ExplodingModel:
        def predict_proba(self, X):
            raise ValueError("shape mismatch")

    bundle = _bundle_with_hypertension_as_argmax()
    bundle.model = _ExplodingModel()
    monkeypatch.setattr(ml_corroborate_module, "load_bundle", lambda: bundle)

    state = _state(
        symptoms=["ألم في الصدر", "دوخة", "صداع"],
        candidate_diseases=[_candidate("Hypertension")],
    )

    result = ml_corroborate(state)

    assert result == {}


def test_density_floor_not_cleared_suppresses_the_signal_even_with_argmax_agreement(monkeypatch):
    monkeypatch.setattr(
        ml_corroborate_module, "load_bundle", _bundle_with_hypertension_as_argmax
    )
    # Only ONE Hypertension-relevant feature present (headache) — below
    # ml.density_floor.MIN_REQUIRED_MATCHED=2, even though the model's
    # own top pick still happens to be Hypertension.
    state = _state(
        symptoms=["صداع"],
        candidate_diseases=[_candidate("Hypertension")],
    )

    result = ml_corroborate(state)

    assert "ml_corroboration" not in result["candidate_diseases"][0]


def test_rank_disagreement_suppresses_the_signal_even_with_density_floor_cleared(monkeypatch):
    monkeypatch.setattr(ml_corroborate_module, "load_bundle", _bundle_with_migraine_as_argmax)
    state = _state(
        symptoms=["ألم في الصدر", "دوخة", "صداع"],  # clears Hypertension's floor
        candidate_diseases=[_candidate("Hypertension")],  # but Migraine is the model's argmax
    )

    result = ml_corroborate(state)

    assert "ml_corroboration" not in result["candidate_diseases"][0]


def test_feature_vector_passed_to_predict_proba_matches_bundle_feature_order_length(monkeypatch):
    bundle = _bundle_with_hypertension_as_argmax()
    monkeypatch.setattr(ml_corroborate_module, "load_bundle", lambda: bundle)
    state = _state(
        symptoms=["ألم في الصدر", "دوخة", "صداع"],
        candidate_diseases=[_candidate("Hypertension")],
    )

    ml_corroborate(state)

    [call] = bundle.model.calls
    assert len(call[0]) == len(_FEATURE_ORDER)
