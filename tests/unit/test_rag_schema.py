import json

import pytest
from pydantic import ValidationError

from rag.schema import KnowledgeBaseEntry, load_all


def _entry(**overrides):
    payload = {
        "name": "Test Disease",
        "name_ar": "مرض تجريبي",
        "symptoms": ["صداع"],
        "specialties": ["طب عام"],
        "source": "Test Source",
        "note": None,
        "translation_reviewed": True,
    }
    payload.update(overrides)
    return KnowledgeBaseEntry.model_validate(payload)


def test_accepts_a_well_formed_entry():
    entry = _entry()
    assert entry.name == "Test Disease"
    assert entry.name_ar == "مرض تجريبي"
    assert entry.symptoms == ["صداع"]


def test_note_defaults_to_none():
    payload = {
        "name": "Test Disease",
        "name_ar": "مرض تجريبي",
        "symptoms": ["صداع"],
        "specialties": ["طب عام"],
        "source": "Test Source",
        "translation_reviewed": True,
    }
    entry = KnowledgeBaseEntry.model_validate(payload)
    assert entry.note is None


def test_rejects_an_empty_symptom_list():
    with pytest.raises(ValidationError):
        _entry(symptoms=[])


def test_rejects_an_empty_specialty_list():
    with pytest.raises(ValidationError):
        _entry(specialties=[])


def test_rejects_a_missing_translation_reviewed_field():
    payload = {
        "name": "Test Disease",
        "name_ar": "مرض تجريبي",
        "symptoms": ["صداع"],
        "specialties": ["طب عام"],
        "source": "Test Source",
    }
    with pytest.raises(ValidationError):
        KnowledgeBaseEntry.model_validate(payload)


def test_rejects_a_missing_name_ar_field():
    payload = {
        "name": "Test Disease",
        "symptoms": ["صداع"],
        "specialties": ["طب عام"],
        "source": "Test Source",
        "translation_reviewed": True,
    }
    with pytest.raises(ValidationError):
        KnowledgeBaseEntry.model_validate(payload)


def test_rejects_an_empty_name_ar():
    # A patient-facing name is the whole point of this field — an empty
    # string is not a placeholder to allow and fill in later (CLAUDE.md >
    # Symptom vocabulary's "fails loudly, not silently" philosophy applies
    # to disease names here too).
    with pytest.raises(ValidationError):
        _entry(name_ar="")


def test_applicable_sex_defaults_to_none():
    # None means applicable to either sex — correct default for almost
    # every entry (rag/schema.py's own field description).
    entry = _entry()
    assert entry.applicable_sex is None


@pytest.mark.parametrize("sex", ["male", "female"])
def test_accepts_a_valid_applicable_sex(sex):
    entry = _entry(applicable_sex=sex)
    assert entry.applicable_sex == sex


def test_rejects_an_invalid_applicable_sex():
    with pytest.raises(ValidationError):
        _entry(applicable_sex="other")


def test_symptoms_are_not_constrained_to_the_vocabulary_enum():
    # Deliberately not enum-constrained (CLAUDE.md > Symptom vocabulary) —
    # this is reference data being audited for vocabulary gaps, not LLM
    # output being forced into a closed set.
    entry = _entry(symptoms=["مصطلح غير موجود بعد بالمفردات"])
    assert entry.symptoms == ["مصطلح غير موجود بعد بالمفردات"]


# --- load_all ------------------------------------------------------------------


def test_load_all_reads_every_json_file_in_the_directory(tmp_path):
    (tmp_path / "b_disease.json").write_text(
        json.dumps(
            {
                "name": "B Disease",
                "name_ar": "مرض ب",
                "symptoms": ["س1"],
                "specialties": ["طب عام"],
                "source": "src",
                "translation_reviewed": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "a_disease.json").write_text(
        json.dumps(
            {
                "name": "A Disease",
                "name_ar": "مرض أ",
                "symptoms": ["س2"],
                "specialties": ["طب عام"],
                "source": "src",
                "translation_reviewed": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entries = load_all(tmp_path)

    assert [e.name for e in entries] == ["A Disease", "B Disease"]  # sorted by filename


def test_load_all_raises_on_a_malformed_file(tmp_path):
    (tmp_path / "broken.json").write_text(
        json.dumps({"name": "Broken", "symptoms": []}), encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_all(tmp_path)


def test_the_real_knowledge_base_directory_loads_and_validates():
    # Sanity check on the actual shipped data, not just the schema logic.
    entries = load_all()
    assert len(entries) == 49
    assert all(e.translation_reviewed for e in entries)
    # name_ar completeness check on the real KB — mirrors the
    # translation_reviewed assertion above. Since name_ar is required
    # and non-empty at the schema level, a genuinely missing one would
    # already have failed load_all() above with a ValidationError before
    # this line could run; asserted explicitly anyway as the documented
    # completeness guarantee (see rag/coverage.py's own name_ar report).
    assert all(e.name_ar for e in entries)
    # applicable_sex: exactly the 4 anatomically sex-restricted entries
    # are gated, and no others — reviewed the full KB for this deliberately
    # (nodes/rag_retrieve.py's module docstring), not just these in
    # isolation. urinary_tract_infection/iron_deficiency_anaemia/mumps are
    # all more-common-in-one-sex, not exclusive, so must stay None.
    by_name = {e.name: e for e in entries}
    assert by_name["Dysmenorrhea"].applicable_sex == "female"
    assert by_name["Polycystic Ovary Syndrome"].applicable_sex == "female"
    assert by_name["Vaginal Candidiasis"].applicable_sex == "female"
    assert by_name["Bacterial Vaginosis"].applicable_sex == "female"
    gated = {e.name for e in entries if e.applicable_sex is not None}
    assert gated == {
        "Dysmenorrhea",
        "Polycystic Ovary Syndrome",
        "Vaginal Candidiasis",
        "Bacterial Vaginosis",
    }
