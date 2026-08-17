import pytest
from pydantic import ValidationError

import schemas.diagnosis as diagnosis_schema
from schemas.diagnosis import build_diagnosis_schema


# --- valid parse -------------------------------------------------------------


def test_parses_a_differential_selecting_a_valid_candidate():
    Schema = build_diagnosis_schema(("Influenza", "Migraine"))

    result = Schema.model_validate({"status": "differential", "differential": ["Influenza"]})

    assert result.status == "differential"
    assert result.differential == ["Influenza"]


def test_parses_insufficient_information_with_an_empty_differential():
    Schema = build_diagnosis_schema(("Influenza",))

    result = Schema.model_validate({"status": "insufficient_information"})

    assert result.status == "insufficient_information"
    assert result.differential == []


# --- the enum is built fresh per call, from the given names only ----------------


def test_rejects_a_disease_name_outside_the_given_candidate_set():
    # The whole point of building the enum per call: a plausible-looking
    # name that ISN'T one of this turn's retrieved candidates must not
    # parse, mirroring schemas.symptoms.SymptomExtraction's enum guard.
    Schema = build_diagnosis_schema(("Influenza", "Migraine"))

    with pytest.raises(ValidationError):
        Schema.model_validate({"status": "differential", "differential": ["Asthma"]})


def test_a_schema_built_for_one_candidate_set_does_not_accept_another_turns_names():
    schema_a = build_diagnosis_schema(("Influenza",))
    schema_b = build_diagnosis_schema(("Migraine",))

    with pytest.raises(ValidationError):
        schema_a.model_validate({"status": "differential", "differential": ["Migraine"]})
    with pytest.raises(ValidationError):
        schema_b.model_validate({"status": "differential", "differential": ["Influenza"]})


def test_requires_at_least_one_candidate_name():
    # nodes.diagnose.diagnose never calls this for an empty
    # candidate_diseases list (it short-circuits in code instead) — this
    # pins down why: a zero-length Literal has no valid values at all.
    with pytest.raises(ValueError):
        build_diagnosis_schema(())


# --- consistency repair: status vs differential, repaired not rejected ----------


def test_a_stray_differential_on_insufficient_information_is_dropped():
    Schema = build_diagnosis_schema(("Influenza",))

    result = Schema.model_validate(
        {"status": "insufficient_information", "differential": ["Influenza"]}
    )

    assert result.differential == []


def test_dropping_a_stray_differential_is_audited(monkeypatch):
    logged = []
    monkeypatch.setattr(diagnosis_schema, "log_malformed_output", lambda **kw: logged.append(kw))
    Schema = build_diagnosis_schema(("Influenza",))

    Schema.model_validate({"status": "insufficient_information", "differential": ["Influenza"]})

    assert len(logged) == 1
    assert logged[0]["reason"] == "differential_present_while_insufficient"


def test_a_differential_status_with_no_selections_falls_back_to_insufficient():
    Schema = build_diagnosis_schema(("Influenza",))

    result = Schema.model_validate({"status": "differential", "differential": []})

    assert result.status == "insufficient_information"


def test_falling_back_on_empty_differential_is_audited(monkeypatch):
    logged = []
    monkeypatch.setattr(diagnosis_schema, "log_malformed_output", lambda **kw: logged.append(kw))
    Schema = build_diagnosis_schema(("Influenza",))

    Schema.model_validate({"status": "differential", "differential": []})

    assert len(logged) == 1
    assert logged[0]["reason"] == "differential_status_with_no_candidates_selected"
