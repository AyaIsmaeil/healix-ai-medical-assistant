import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.check_red_flags import check_red_flags
from rules.red_flags import RED_FLAG_RULES, RedFlagMatch
from schemas.red_flags import RedFlagAssessment


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def red_flag_response(has_red_flag: bool, reasoning: str | None = "r") -> _ProviderResponse:
    payload = {"has_red_flag": has_red_flag, "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _state(*, symptoms=(), medical_record_summary="", unmatched_mentions=(), thread_id="thread-1"):
    return {
        "thread_id": thread_id,
        "symptoms": [dict(s) for s in symptoms],
        "medical_record_summary": medical_record_summary,
        "unmatched_mentions": list(unmatched_mentions),
    }


def _symptom(name):
    return {"name": name}


def _reason_ar(rule_id: str) -> str:
    return next(r for r in RED_FLAG_RULES if r.id == rule_id).reason_ar


# Plain "صداع" (headache) — confirmed directly against RED_FLAG_RULES to
# be referenced by no rule at all (unlike "صداع شديد ومفاجئ", a different
# string). Using it lets a test exercise the real, unmocked rule engine
# while staying sure it will never fire on its own.
ORDINARY_SYMPTOM = _symptom("صداع")

FAKE_MATCH = RedFlagMatch(
    rule_id="fake_rule",
    category="fake_category",
    reason_ar="سبب وهمي للاختبار",
    source="fake source",
    matched_symptoms=frozenset({"عرض وهمي"}),
    lowered_by_chronic_condition=False,
)


# --- OR logic: rule layer mocked --------------------------------------------


def test_check_red_flags_fires_when_only_the_rule_layer_matches(monkeypatch):
    monkeypatch.setattr(
        "nodes.check_red_flags.run_red_flag_rules", lambda symptoms, record: [FAKE_MATCH]
    )
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result == {"red_flags": [{"id": "fake_rule", "reason": "سبب وهمي للاختبار"}]}


def test_check_red_flags_fires_when_both_layers_match(monkeypatch):
    monkeypatch.setattr(
        "nodes.check_red_flags.run_red_flag_rules", lambda symptoms, record: [FAKE_MATCH]
    )
    provider = FakeProvider(responses=[red_flag_response(True, reasoning="قلق شديد")])
    set_provider(provider)

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result == {
        "red_flags": [
            {"id": "fake_rule", "reason": "سبب وهمي للاختبار"},
            {"id": "llm", "reason": "قلق شديد"},
        ]
    }


def test_check_red_flags_does_not_fire_when_neither_layer_matches(monkeypatch):
    monkeypatch.setattr("nodes.check_red_flags.run_red_flag_rules", lambda symptoms, record: [])
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result == {"red_flags": []}


# --- the node still works when only the LLM layer fires ---------------------
# No mocking of run_red_flag_rules here: real rule engine, a symptom
# guaranteed not to satisfy any real rule, exercising the actual OR
# against a genuine miss on the rule side rather than a stand-in for one.


def test_check_red_flags_fires_when_only_the_llm_layer_matches():
    provider = FakeProvider(responses=[red_flag_response(True, reasoning="نمط غير معتاد")])
    set_provider(provider)

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result == {"red_flags": [{"id": "llm", "reason": "نمط غير معتاد"}]}


def test_check_red_flags_returns_empty_via_the_real_rule_layer_when_llm_also_misses():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result == {"red_flags": []}


# --- real rule engine, real combination --------------------------------------
# First exercise of the actual 9-rule set against real extraction-shaped
# output, not a mock standing in for it.


def test_a_real_rule_combination_fires_through_the_actual_rule_engine():
    # ألم في الصدر + ضيق تنفس genuinely satisfies TWO real rules at once
    # (acs_chest_pain and pulmonary_embolism) — verified directly against
    # rules.red_flags.check_red_flags before writing this assertion, not
    # assumed by hand.
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    assert {entry["id"] for entry in result["red_flags"]} == {
        "acs_chest_pain",
        "pulmonary_embolism",
    }


def test_a_real_rule_combination_still_ors_with_the_llm():
    provider = FakeProvider(responses=[red_flag_response(True, reasoning="تأكيد إضافي")])
    set_provider(provider)

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    ids = {entry["id"] for entry in result["red_flags"]}
    assert ids == {"acs_chest_pain", "pulmonary_embolism", "llm"}


def test_chronic_condition_context_reaches_the_real_rule_layer():
    # acs_chest_pain's chronic_requirement drops the any_of clause for a
    # diabetes mention in the record summary — chest pain alone fires
    # only with that context present. Real rule engine, not mocked,
    # proving medical_record_summary actually reaches it.
    provider = FakeProvider(responses=[red_flag_response(False), red_flag_response(False)])
    set_provider(provider)

    no_context = check_red_flags(_state(symptoms=[_symptom("ألم في الصدر")]))
    with_context = check_red_flags(
        _state(
            symptoms=[_symptom("ألم في الصدر")],
            medical_record_summary="مريض سكري من النوع الثاني",
        )
    )

    assert no_context["red_flags"] == []
    assert with_context["red_flags"] == [
        {"id": "acs_chest_pain", "reason": _reason_ar("acs_chest_pain")}
    ]


# --- LLM also sees unmatched_mentions, the rule engine cannot -----------------


def test_unmatched_mentions_reach_the_llm_prompt():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM], unmatched_mentions=["طنين بالأذن"]))

    prompt = provider.calls[0]["prompt"]
    assert "طنين بالأذن" in prompt


# --- pairing: id and reason are structurally paired, never positional ---------
#
# The regression this section guards: red_flags used to be a bare list of
# id strings with a SEPARATE, same-length-assumed red_flag_reasons list —
# and the two were not even reliably the same length (the LLM's entry had
# no corresponding reason at all). Now each entry carries its own id AND
# reason together, so there is no index to misalign in the first place.
# These tests would catch a regression back to that shape, or a bug that
# paired the wrong reason with the wrong id.


def test_each_rule_matched_entry_carries_its_own_reason_not_a_swapped_one():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    # Built as a dict keyed by id specifically so a mix-up (acs_chest_pain
    # paired with pulmonary_embolism's reason, or vice versa) fails this
    # assertion — a test that only checked "both ids present" and "both
    # reasons present" as separate sets would not catch that swap.
    by_id = {entry["id"]: entry["reason"] for entry in result["red_flags"]}
    assert by_id == {
        "acs_chest_pain": _reason_ar("acs_chest_pain"),
        "pulmonary_embolism": _reason_ar("pulmonary_embolism"),
    }


def test_the_llm_entry_is_paired_with_its_own_reasoning_not_a_rule_reason():
    provider = FakeProvider(responses=[red_flag_response(True, reasoning="ملاحظة النموذج")])
    set_provider(provider)

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    by_id = {entry["id"]: entry["reason"] for entry in result["red_flags"]}
    assert by_id["llm"] == "ملاحظة النموذج"
    assert by_id["llm"] not in (
        _reason_ar("acs_chest_pain"),
        _reason_ar("pulmonary_embolism"),
    )


def test_every_red_flag_entry_has_both_id_and_reason_keys():
    provider = FakeProvider(responses=[red_flag_response(True, reasoning="قلق")])
    set_provider(provider)

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    assert len(result["red_flags"]) == 3  # 2 real rules + the LLM entry
    for entry in result["red_flags"]:
        assert set(entry) == {"id", "reason"}
        assert entry["id"]
        assert entry["reason"]


# --- node contract: partial state dict, quality tier -------------------------


def test_check_red_flags_returns_only_the_red_flags_key():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert set(result) == {"red_flags"}


def test_check_red_flags_calls_the_llm_on_the_quality_tier():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is RedFlagAssessment


# --- audit: both verdicts logged separately ----------------------------------


def test_check_red_flags_audit_logs_both_verdicts_separately(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "nodes.check_red_flags.log_red_flag_detection", lambda **kwargs: logged.append(kwargs)
    )
    monkeypatch.setattr(
        "nodes.check_red_flags.run_red_flag_rules", lambda symptoms, record: [FAKE_MATCH]
    )
    provider = FakeProvider(responses=[red_flag_response(False, reasoning=None)])
    set_provider(provider)

    check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM], thread_id="thread-42"))

    assert len(logged) == 1
    record = logged[0]
    assert record["thread_id"] == "thread-42"
    assert record["rule_matched"] is True
    assert record["rule_ids"] == ["fake_rule"]
    assert record["llm_matched"] is False
    assert record["llm_reasoning"] is None
    assert record["combined"] is True
