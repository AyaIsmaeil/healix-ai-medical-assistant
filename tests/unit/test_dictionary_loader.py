"""اختبارات وحدة لـDictionaryLoader — تحميل وتحقّق بنيوي لقواميس JSON عند
الإقلاع، بلا قراءة ملفات لكل طلب. يستخدم tmp_path (نفس نمط pytest المُستخدَم
أصلاً بالمشروع عبر monkeypatch/caplog)."""

import json

import pytest

from app.exceptions import FeatureValidationError
from app.infrastructure.dictionary_loader import DictionaryLoader


def _write(tmp_path, data):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_valid_rules_file(tmp_path):
    path = _write(tmp_path, {"fields": {"age": {"type": "int", "min": 0, "max": 120}}})
    rules = DictionaryLoader.load_feature_validation_rules(path)
    assert rules["fields"]["age"]["max"] == 120


def test_loads_enum_field(tmp_path):
    path = _write(tmp_path, {"fields": {"gender": {"type": "enum", "allowed": ["male", "female"]}}})
    rules = DictionaryLoader.load_feature_validation_rules(path)
    assert rules["fields"]["gender"]["allowed"] == ["male", "female"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(tmp_path / "nope.json")


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_root_not_object_raises(tmp_path):
    path = _write(tmp_path, ["not", "an", "object"])
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_missing_fields_key_raises(tmp_path):
    path = _write(tmp_path, {"other": {}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_empty_fields_raises(tmp_path):
    path = _write(tmp_path, {"fields": {}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_numeric_field_without_min_max_raises(tmp_path):
    path = _write(tmp_path, {"fields": {"age": {"type": "int"}}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_numeric_field_min_greater_than_max_raises(tmp_path):
    path = _write(tmp_path, {"fields": {"age": {"type": "int", "min": 100, "max": 10}}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_enum_field_without_allowed_raises(tmp_path):
    path = _write(tmp_path, {"fields": {"gender": {"type": "enum"}}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_enum_field_with_empty_allowed_raises(tmp_path):
    path = _write(tmp_path, {"fields": {"gender": {"type": "enum", "allowed": []}}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_unsupported_type_raises(tmp_path):
    path = _write(tmp_path, {"fields": {"x": {"type": "weird"}}})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_invalid_on_out_of_range_strategy_raises(tmp_path):
    path = _write(
        tmp_path,
        {"fields": {"age": {"type": "int", "min": 0, "max": 10, "on_out_of_range": "explode"}}},
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_validation_rules(path)


def test_default_path_loads_real_project_file():
    """يضمن أنّ الملف الفعلي app/dictionaries/feature_validation_rules.json
    موجود وصالح البنية — لو تعطّل الإقلاع الفعلي سيظهر هنا أولاً."""
    rules = DictionaryLoader.load_feature_validation_rules()
    assert "age" in rules["fields"]
    assert "gender" in rules["fields"]
    assert "severity_0_10" in rules["fields"]
    assert "temperature_c" in rules["fields"]


# ========================================================================
# load_feature_schema (Phase 3.3) — تحقّق بنيوي + تماسك داخلي
# ========================================================================
_VALID_SCHEMA = {
    "schema_version": "test-v1",
    "feature_order": ["age", "gender_male", "gender_female"],
    "numeric_fields": {"age": {"source": "demographics.age", "nullable": True}},
    "boolean_fields": {},
    "categorical_fields": {
        "gender": {
            "source": "demographics.gender",
            "values": ["male", "female"],
            "one_hot_prefix": "gender",
            "nullable": True,
        }
    },
}


def _write_schema(tmp_path, data):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_valid_schema_file(tmp_path):
    path = _write_schema(tmp_path, _VALID_SCHEMA)
    schema = DictionaryLoader.load_feature_schema(path)
    assert schema["schema_version"] == "test-v1"


def test_schema_missing_file_raises(tmp_path):
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(tmp_path / "nope.json")


def test_schema_malformed_json_raises(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_missing_schema_version_raises(tmp_path):
    data = {k: v for k, v in _VALID_SCHEMA.items() if k != "schema_version"}
    path = _write_schema(tmp_path, data)
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_empty_feature_order_raises(tmp_path):
    path = _write_schema(tmp_path, {**_VALID_SCHEMA, "feature_order": []})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_duplicate_feature_order_raises(tmp_path):
    path = _write_schema(
        tmp_path, {**_VALID_SCHEMA, "feature_order": ["age", "age", "gender_male", "gender_female"]}
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_no_fields_defined_raises(tmp_path):
    path = _write_schema(tmp_path, {
        "schema_version": "test-v1",
        "feature_order": ["x"],
        "numeric_fields": {},
        "boolean_fields": {},
        "categorical_fields": {},
    })
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_numeric_field_missing_nullable_raises(tmp_path):
    data = {**_VALID_SCHEMA, "numeric_fields": {"age": {"source": "demographics.age"}}}
    path = _write_schema(tmp_path, data)
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_categorical_field_missing_values_raises(tmp_path):
    data = {
        **_VALID_SCHEMA,
        "categorical_fields": {
            "gender": {"source": "demographics.gender", "one_hot_prefix": "gender", "nullable": True}
        },
    }
    path = _write_schema(tmp_path, data)
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_categorical_field_missing_one_hot_prefix_raises(tmp_path):
    data = {
        **_VALID_SCHEMA,
        "categorical_fields": {
            "gender": {"source": "demographics.gender", "values": ["male", "female"], "nullable": True}
        },
    }
    path = _write_schema(tmp_path, data)
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_categorical_duplicate_values_raises(tmp_path):
    data = {
        **_VALID_SCHEMA,
        "categorical_fields": {
            "gender": {
                "source": "demographics.gender",
                "values": ["male", "male"],
                "one_hot_prefix": "gender",
                "nullable": True,
            }
        },
    }
    path = _write_schema(tmp_path, data)
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_feature_order_missing_a_defined_column_raises(tmp_path):
    """feature_order يجب أن يطابق تماماً الأعمدة الناتجة فعلياً — نقص هنا."""
    path = _write_schema(tmp_path, {**_VALID_SCHEMA, "feature_order": ["age", "gender_male"]})
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_schema_feature_order_has_extra_undefined_column_raises(tmp_path):
    """feature_order يجب ألا يحوي عموداً لا يُنتجه أي حقل مُعرَّف — زيادة هنا."""
    path = _write_schema(
        tmp_path,
        {**_VALID_SCHEMA, "feature_order": ["age", "gender_male", "gender_female", "phantom"]},
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_feature_schema(path)


def test_default_path_loads_real_project_schema():
    """يضمن أنّ الملف الفعلي app/dictionaries/feature_schemas/v1.json موجود
    ومتماسك البنية — لو تعطّل الإقلاع الفعلي سيظهر هنا أولاً."""
    schema = DictionaryLoader.load_feature_schema()
    assert schema["schema_version"] == "assessment-features-v1"
    assert "age" in schema["feature_order"]
    assert "gender_male" in schema["feature_order"]


# ========================================================================
# load_specialty_lookup (Phase 3.6) — تحقّق بنيوي لقاموس (مرض → تخصّص)
# ========================================================================
def _write_yaml(tmp_path, text):
    path = tmp_path / "specialty_lookup.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_specialty_lookup(tmp_path):
    path = _write_yaml(tmp_path, "influenza:\n  specialty: Family Medicine\n")
    lookup = DictionaryLoader.load_specialty_lookup(path)
    assert lookup["influenza"]["specialty"] == "Family Medicine"


def test_specialty_lookup_missing_file_raises(tmp_path):
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(tmp_path / "nope.yaml")


def test_specialty_lookup_malformed_yaml_raises(tmp_path):
    path = _write_yaml(tmp_path, "influenza: [unclosed\n  specialty: x")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(path)


def test_specialty_lookup_root_not_object_raises(tmp_path):
    path = _write_yaml(tmp_path, "- influenza\n- pneumonia\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(path)


def test_specialty_lookup_empty_root_raises(tmp_path):
    path = _write_yaml(tmp_path, "{}\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(path)


def test_specialty_lookup_value_not_object_raises(tmp_path):
    path = _write_yaml(tmp_path, "influenza: Family Medicine\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(path)


def test_specialty_lookup_missing_specialty_key_raises(tmp_path):
    path = _write_yaml(tmp_path, "influenza:\n  note: x\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(path)


def test_specialty_lookup_empty_specialty_value_raises(tmp_path):
    path = _write_yaml(tmp_path, "influenza:\n  specialty: ''\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_specialty_lookup(path)


def test_default_path_loads_real_project_specialty_lookup():
    """يضمن أنّ الملف الفعلي app/dictionaries/specialty_lookup.yaml موجود
    وصالح البنية — لو تعطّل الإقلاع الفعلي سيظهر هنا أولاً."""
    lookup = DictionaryLoader.load_specialty_lookup()
    assert lookup["influenza"]["specialty"] == "Family Medicine"
    assert lookup["pregnancy"]["specialty"] == "Obstetrics and Gynecology"


# ----------------------------------------------------------------------
# load_disease_metadata (ADR-04)
# ----------------------------------------------------------------------
def _write_disease_metadata_yaml(tmp_path, text):
    path = tmp_path / "disease_metadata.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_disease_metadata(tmp_path):
    path = _write_disease_metadata_yaml(
        tmp_path,
        "diseases:\n"
        "  Influenza:\n"
        "    icd10: j11.1\n"
        "    severity: 3\n"
        "    specialty: Infectious Disease\n"
        "    requires_review: false\n"
        "    review_reason: null\n",
    )
    data = DictionaryLoader.load_disease_metadata(path)
    assert data["diseases"]["Influenza"]["specialty"] == "Infectious Disease"


def test_disease_metadata_root_not_object_raises(tmp_path):
    path = _write_disease_metadata_yaml(tmp_path, "- not an object\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_disease_metadata(path)


def test_disease_metadata_missing_diseases_key_raises(tmp_path):
    path = _write_disease_metadata_yaml(tmp_path, "version: '1.0.0'\n")
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_disease_metadata(path)


def test_disease_metadata_empty_specialty_raises(tmp_path):
    path = _write_disease_metadata_yaml(
        tmp_path,
        "diseases:\n"
        "  X:\n"
        "    icd10: A00\n"
        "    specialty: ''\n"
        "    requires_review: false\n"
        "    review_reason: null\n",
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_disease_metadata(path)


def test_disease_metadata_null_specialty_raises(tmp_path):
    path = _write_disease_metadata_yaml(
        tmp_path,
        "diseases:\n"
        "  X:\n"
        "    icd10: A00\n"
        "    specialty: null\n"
        "    requires_review: false\n"
        "    review_reason: null\n",
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_disease_metadata(path)


def test_disease_metadata_requires_review_without_reason_raises(tmp_path):
    """القاعدة الحاسمة: requires_review=true بلا review_reason موثَّق ممنوع."""
    path = _write_disease_metadata_yaml(
        tmp_path,
        "diseases:\n"
        "  X:\n"
        "    icd10: A00\n"
        "    specialty: General Medicine\n"
        "    requires_review: true\n"
        "    review_reason: null\n",
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_disease_metadata(path)


def test_disease_metadata_requires_review_bool_type_enforced(tmp_path):
    path = _write_disease_metadata_yaml(
        tmp_path,
        "diseases:\n"
        "  X:\n"
        "    icd10: A00\n"
        "    specialty: General Medicine\n"
        "    requires_review: 'yes'\n"
        "    review_reason: null\n",
    )
    with pytest.raises(FeatureValidationError):
        DictionaryLoader.load_disease_metadata(path)


def test_default_path_loads_real_project_disease_metadata():
    """يضمن أنّ الملف الفعلي app/dictionaries/disease_metadata.yaml موجود
    وصالح البنية، ويغطّي الأمراض الـ٤٩ كاملة بصفر specialty فارغ."""
    data = DictionaryLoader.load_disease_metadata()
    diseases = data["diseases"]
    assert len(diseases) == 49
    for name, entry in diseases.items():
        assert entry["specialty"], f"{name}: specialty فارغ بالملف الفعلي"
    assert diseases["Sarcoidosis"]["requires_review"] is True
    assert diseases["Sarcoidosis"]["specialty"] == "General Medicine"
    assert diseases["Unstable angina"]["requires_review"] is False
    assert diseases["Unstable angina"]["specialty"] == "Cardiology"
