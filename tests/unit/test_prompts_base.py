import pytest

from prompts import base
from prompts.base import _load_template, build_prompt

KNOWN_PREAMBLE_SUBSTRING = "لا تُصدر أبدًا تشخيصًا نهائيًا أو قطعيًا"


def test_safety_preamble_contains_known_arabic_content():
    preamble = _load_template("_safety_preamble")

    assert KNOWN_PREAMBLE_SUBSTRING in preamble


def test_build_prompt_always_includes_preamble():
    prompt = build_prompt("_safety_preamble")

    assert KNOWN_PREAMBLE_SUBSTRING in prompt


def test_load_template_rejects_pasted_json_schema_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "TEMPLATES_DIR", tmp_path)
    (tmp_path / "diagnosis.txt").write_text('{"$ref": "#/$defs/Symptom"}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"diagnosis\.txt"):
        _load_template("diagnosis")


def test_load_template_rejects_unresolved_variable(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "TEMPLATES_DIR", tmp_path)
    (tmp_path / "followup.txt").write_text("مرحبا $patient_name", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\$patient_name"):
        _load_template("followup")


def test_load_template_succeeds_when_all_tokens_are_supplied(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "TEMPLATES_DIR", tmp_path)
    (tmp_path / "followup.txt").write_text("مرحبا $patient_name", encoding="utf-8")

    result = _load_template("followup", patient_name="سارة")

    assert result == "مرحبا سارة"


def test_load_template_allows_literal_json_braces_without_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "TEMPLATES_DIR", tmp_path)
    (tmp_path / "example.txt").write_text('{"symptoms": ["صداع", "حمى"]}', encoding="utf-8")

    result = _load_template("example")

    assert result == '{"symptoms": ["صداع", "حمى"]}'
