import importlib

import pytest
from pydantic import ValidationError

import schemas.symptoms as symptoms_schema
import vocabulary.symptoms as vocabulary_module
from schemas.symptoms import (
    SYMPTOM_NAMES,
    ExtractedSymptom,
    SymptomExtraction,
)

# Two real vocabulary entries, taken from the vocabulary rather than
# hardcoded, so these tests don't break when the list is reconciled.
NAME_A, NAME_B = SYMPTOM_NAMES[0], SYMPTOM_NAMES[1]


# --- valid parse -------------------------------------------------------------


def test_parses_a_valid_extraction():
    result = SymptomExtraction.model_validate(
        {
            "symptoms": [
                {
                    "name": NAME_A,
                    "raw_mention": "بطني عم يوجعني من امبارح",
                    "duration": "يوم",
                    "severity": "moderate",
                    "onset": "gradual",
                }
            ],
            "negated_symptoms": [{"name": NAME_B}],
        }
    )

    assert result.symptoms[0].name == NAME_A
    assert result.symptoms[0].severity == "moderate"
    assert result.negated_symptoms[0].name == NAME_B


def test_empty_extraction_is_valid():
    result = SymptomExtraction.model_validate({})

    assert result.symptoms == []
    assert result.negated_symptoms == []
    assert result.unmatched_mentions == []


# --- unmatched_mentions: symptoms with no canonical name --------------------


def test_unmatched_mentions_accepts_a_phrase_outside_the_vocabulary():
    # The whole point of this field: unlike `name`, it is NOT enum
    # constrained, because a real symptom missing from the vocabulary is
    # exactly what it exists to hold onto instead of silently dropping.
    result = SymptomExtraction.model_validate({"unmatched_mentions": ["طنين بالأذن"]})

    assert result.unmatched_mentions == ["طنين بالأذن"]


def test_unmatched_mentions_defaults_to_empty_list():
    result = SymptomExtraction.model_validate({"symptoms": [{"name": SYMPTOM_NAMES[0]}]})

    assert result.unmatched_mentions == []


def test_unmatched_mentions_coexists_with_empty_symptom_lists():
    # Exactly the reported scenario: a symptom-like phrase with nothing
    # else extractable from the message.
    result = SymptomExtraction.model_validate({"unmatched_mentions": ["طنين بالأذن"]})

    assert result.symptoms == []
    assert result.negated_symptoms == []
    assert result.unmatched_mentions == ["طنين بالأذن"]


# --- out-of-vocabulary rejection ---------------------------------------------


def test_rejects_symptom_name_outside_the_vocabulary():
    # The whole point of the enum: a plausible-looking name the model
    # invented must not parse, because rules/red_flags.py would then never
    # match it and the red flag would silently never fire.
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate({"symptoms": [{"name": "وجع بالمعدة الشديد"}]})


def test_rejects_out_of_vocabulary_negated_name():
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate({"negated_symptoms": [{"name": "عرض مخترع"}]})


def test_rejects_near_miss_spelling_of_a_real_entry():
    # Byte-for-byte or nothing — a one-character variant is still invalid.
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate({"symptoms": [{"name": NAME_A + "ة"}]})


def test_rejects_free_text_smuggled_into_name():
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate(
            {"symptoms": [{"name": "المريض قال إنه يشعر بألم في صدره"}]}
        )


def test_free_text_belongs_in_raw_mention():
    result = SymptomExtraction.model_validate(
        {"symptoms": [{"name": NAME_A, "raw_mention": "حاسس بشي غريب بصدري"}]}
    )

    assert result.symptoms[0].raw_mention == "حاسس بشي غريب بصدري"


def test_rejects_severity_outside_the_constrained_set():
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate(
            {"symptoms": [{"name": NAME_A, "severity": "very bad"}]}
        )


def test_rejects_onset_outside_the_constrained_set():
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate(
            {"symptoms": [{"name": NAME_A, "onset": "whenever"}]}
        )


# --- null handling for optional fields ----------------------------------------


def test_optional_fields_default_to_none_when_omitted():
    symptom = ExtractedSymptom.model_validate({"name": NAME_A})

    assert symptom.raw_mention is None
    assert symptom.duration is None
    assert symptom.severity is None
    assert symptom.onset is None


def test_optional_fields_accept_explicit_null():
    symptom = ExtractedSymptom.model_validate(
        {"name": NAME_A, "duration": None, "severity": None, "onset": None}
    )

    assert symptom.duration is None
    assert symptom.severity is None


def test_name_is_required():
    with pytest.raises(ValidationError):
        ExtractedSymptom.model_validate({"raw_mention": "شي ما مفهوم"})


# --- duplicate handling --------------------------------------------------------


def test_duplicate_names_are_collapsed_to_one_entry():
    result = SymptomExtraction.model_validate(
        {"symptoms": [{"name": NAME_A}, {"name": NAME_A}]}
    )

    assert len(result.symptoms) == 1


def test_collapsing_duplicates_fills_missing_fields():
    result = SymptomExtraction.model_validate(
        {
            "symptoms": [
                {"name": NAME_A, "duration": "يومين"},
                {"name": NAME_A, "severity": "severe"},
            ]
        }
    )

    assert len(result.symptoms) == 1
    assert result.symptoms[0].duration == "يومين"
    assert result.symptoms[0].severity == "severe"


def test_collapsing_duplicates_keeps_the_first_conflicting_value():
    # Within one response a duplicate is a model error, so a later entry
    # must not overwrite an earlier value (unlike state.merge_symptoms,
    # which is last-wins because a later turn is newer information).
    result = SymptomExtraction.model_validate(
        {
            "symptoms": [
                {"name": NAME_A, "severity": "mild"},
                {"name": NAME_A, "severity": "severe"},
            ]
        }
    )

    assert result.symptoms[0].severity == "mild"


def test_duplicate_collapse_is_audited_not_silent(monkeypatch):
    logged = []
    monkeypatch.setattr(symptoms_schema, "log_malformed_output", lambda **kw: logged.append(kw))

    SymptomExtraction.model_validate({"symptoms": [{"name": NAME_A}, {"name": NAME_A}]})

    assert len(logged) == 1
    assert logged[0]["reason"] == "duplicate_symptom_name:symptoms"
    assert logged[0]["payload"]["name"] == NAME_A


def test_duplicates_are_collapsed_in_negated_symptoms_too():
    result = SymptomExtraction.model_validate(
        {"negated_symptoms": [{"name": NAME_A}, {"name": NAME_A}]}
    )

    assert len(result.negated_symptoms) == 1


def test_distinct_names_are_not_collapsed():
    result = SymptomExtraction.model_validate(
        {"symptoms": [{"name": NAME_A}, {"name": NAME_B}]}
    )

    assert len(result.symptoms) == 2


def test_collapse_preserves_first_seen_order():
    result = SymptomExtraction.model_validate(
        {"symptoms": [{"name": NAME_B}, {"name": NAME_A}, {"name": NAME_B}]}
    )

    assert [s.name for s in result.symptoms] == [NAME_B, NAME_A]


# --- vocabulary is the single source of truth ----------------------------------


def test_enum_is_derived_from_the_vocabulary():
    from vocabulary.symptoms import CANONICAL_SYMPTOMS

    assert set(SYMPTOM_NAMES) == set(CANONICAL_SYMPTOMS)


def test_generated_json_schema_exposes_the_vocabulary_as_an_enum():
    schema = SymptomExtraction.model_json_schema()
    enum_values = schema["$defs"]["ExtractedSymptom"]["properties"]["name"]["enum"]

    assert set(enum_values) == set(SYMPTOM_NAMES)


def test_adding_a_vocabulary_entry_is_reflected_without_editing_schemas(monkeypatch):
    """A vocabulary change must flow through with no edit to schemas/."""
    new_name = "عرض جديد للاختبار"
    assert new_name not in SYMPTOM_NAMES

    # Before: rejected.
    with pytest.raises(ValidationError):
        SymptomExtraction.model_validate({"symptoms": [{"name": new_name}]})

    monkeypatch.setattr(
        vocabulary_module,
        "CANONICAL_SYMPTOMS",
        frozenset(vocabulary_module.CANONICAL_SYMPTOMS | {new_name}),
    )
    reloaded = importlib.reload(symptoms_schema)
    try:
        # After: accepted, with schemas/symptoms.py untouched.
        result = reloaded.SymptomExtraction.model_validate(
            {"symptoms": [{"name": new_name}]}
        )
        assert result.symptoms[0].name == new_name
        assert new_name in reloaded.SYMPTOM_NAMES
    finally:
        monkeypatch.undo()
        importlib.reload(symptoms_schema)


def test_module_state_restored_after_reload_test():
    # Guards the teardown above: a leaked reload would silently corrupt
    # every later test in the session.
    assert "عرض جديد للاختبار" not in symptoms_schema.SYMPTOM_NAMES
