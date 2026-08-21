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


def red_flag_response(potential_red_flag: bool, reasoning: str | None = "r") -> _ProviderResponse:
    payload = {"potential_red_flag": potential_red_flag, "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _state(
    *,
    symptoms=(),
    negated_symptoms=(),
    medical_record_summary="",
    unmatched_mentions=(),
    thread_id="thread-1",
):
    return {
        "thread_id": thread_id,
        "symptoms": [dict(s) for s in symptoms],
        "negated_symptoms": [dict(s) for s in negated_symptoms],
        "medical_record_summary": medical_record_summary,
        "unmatched_mentions": list(unmatched_mentions),
    }


def _symptom(name):
    return {"name": name}


def _reason_ar(rule_id: str) -> str:
    return next(r for r in RED_FLAG_RULES if r.id == rule_id).reason_ar


# Plain "صداع" (headache) — confirmed directly against RED_FLAG_RULES to
# be referenced by no rule at all (unlike "صداع شديد ومفاجئ", a different
# string, or "ألم بطن"/"حمى", both real rule all_of terms). Using it lets
# a test exercise the real, unmocked rule engine while staying sure it
# will never fire — hard match OR candidate — on its own.
ORDINARY_SYMPTOM = _symptom("صداع")

FAKE_MATCH = RedFlagMatch(
    rule_id="fake_rule",
    category="fake_category",
    reason_ar="سبب وهمي للاختبار",
    source="fake source",
    matched_symptoms=frozenset({"عرض وهمي"}),
    lowered_by_chronic_condition=False,
)


def _patch_rule_layer(monkeypatch, matches):
    monkeypatch.setattr(
        "nodes.check_red_flags.run_red_flag_rules",
        lambda symptoms, record, negated=None: list(matches),
    )


# --- Layer 1 (deterministic hard match): the ONLY source of state["red_flags"] --


def test_check_red_flags_fires_from_the_rule_layer_regardless_of_the_llm(monkeypatch):
    _patch_rule_layer(monkeypatch, [FAKE_MATCH])
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result["red_flags"] == [{"id": "fake_rule", "reason": "سبب وهمي للاختبار"}]
    assert result["safety_decision"] == "HARD_EMERGENCY"


def test_a_hard_rule_match_fires_independent_of_the_llms_own_verdict(monkeypatch):
    # The corrected design (module docstring): the LLM's potential_red_flag
    # screen never adds to red_flags and never changes a HARD_EMERGENCY
    # disposition either way — proven here by asserting the SAME result
    # for both a True and a False LLM verdict against the same rule match.
    _patch_rule_layer(monkeypatch, [FAKE_MATCH])

    set_provider(FakeProvider(responses=[red_flag_response(True, reasoning="قلق شديد")]))
    with_llm_true = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    set_provider(FakeProvider(responses=[red_flag_response(False)]))
    with_llm_false = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert with_llm_true["red_flags"] == with_llm_false["red_flags"]
    assert with_llm_true["red_flags"] == [{"id": "fake_rule", "reason": "سبب وهمي للاختبار"}]
    assert with_llm_true["safety_decision"] == with_llm_false["safety_decision"] == "HARD_EMERGENCY"


def test_check_red_flags_does_not_fire_when_the_rule_layer_is_empty(monkeypatch):
    _patch_rule_layer(monkeypatch, [])
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result["red_flags"] == []
    assert result["safety_decision"] == "NO_RED_FLAG"


# --- Layer 3 (LLM screen): logged, never independently escalating ------------
# No mocking of run_red_flag_rules here: real rule engine, ORDINARY_SYMPTOM
# guaranteed not to satisfy any real rule's all_of (hard or candidate).


def test_the_llm_alone_never_produces_a_red_flag():
    set_provider(FakeProvider(responses=[red_flag_response(True, reasoning="نمط غير معتاد")]))

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result["red_flags"] == []
    assert result["safety_decision"] == "NO_RED_FLAG"


def test_the_llm_alone_never_produces_a_candidate_either():
    # potential_red_flag=True with no matching cited rule has no sourced
    # missing_any_of to build a verification question from (module
    # docstring) — it must not appear in red_flag_candidates.
    set_provider(FakeProvider(responses=[red_flag_response(True, reasoning="نمط غير معتاد")]))

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result["red_flag_candidates"] == []


def test_check_red_flags_returns_empty_via_the_real_rule_layer_when_llm_also_misses():
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert result["red_flags"] == []


# --- real rule engine, real HARD combination ----------------------------------


def test_a_real_rule_combination_fires_through_the_actual_rule_engine():
    # ألم في الصدر + ضيق تنفس genuinely satisfies TWO real rules at once
    # (acs_chest_pain and pulmonary_embolism) — verified directly against
    # rules.red_flags.check_red_flags before writing this assertion, not
    # assumed by hand.
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    assert {entry["id"] for entry in result["red_flags"]} == {
        "acs_chest_pain",
        "pulmonary_embolism",
    }
    assert result["safety_decision"] == "HARD_EMERGENCY"
    assert result["red_flag_candidates"] == []


def test_chronic_condition_context_reaches_the_real_rule_layer():
    # acs_chest_pain's chronic_requirement drops the any_of clause for a
    # diabetes mention in the record summary — chest pain alone fires
    # only with that context present. Real rule engine, not mocked,
    # proving medical_record_summary actually reaches it.
    set_provider(
        FakeProvider(responses=[red_flag_response(False), red_flag_response(False)])
    )

    no_context = check_red_flags(_state(symptoms=[_symptom("ألم في الصدر")]))
    with_context = check_red_flags(
        _state(
            symptoms=[_symptom("ألم في الصدر")],
            medical_record_summary="مريض سكري من النوع الثاني",
        )
    )

    # Without the chronic context, isolated chest pain is a CANDIDATE
    # (needs clarification), not a hard match — this is the exact false-
    # positive class this rewrite fixes.
    assert no_context["red_flags"] == []
    assert no_context["safety_decision"] == "NEEDS_CLARIFICATION"
    assert with_context["red_flags"] == [
        {"id": "acs_chest_pain", "reason": _reason_ar("acs_chest_pain")}
    ]
    assert with_context["safety_decision"] == "HARD_EMERGENCY"


# --- Layer 1b (deterministic candidate): the false-positive fix --------------


def test_isolated_chest_pain_is_a_candidate_not_a_hard_emergency():
    # The exact scenario from this project's own red-flag audit: an
    # isolated, context-dependent symptom must not reach emergency_node on
    # its own. NICE CG95 treats chest pain as needing assessment for
    # suspected ACS, not an automatic emergency from the word alone (see
    # rules/red_flags.py's own IncompleteRedFlagCandidate docstring).
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(_state(symptoms=[_symptom("ألم في الصدر")]))

    assert result["red_flags"] == []
    assert result["safety_decision"] == "NEEDS_CLARIFICATION"
    assert len(result["red_flag_candidates"]) == 1
    candidate = result["red_flag_candidates"][0]
    assert candidate["rule_id"] == "acs_chest_pain"
    assert "ضيق تنفس" in candidate["missing_any_of"] or "تعرق غزير" in candidate["missing_any_of"]


def test_explicitly_negated_discriminators_reject_the_candidate():
    # The patient explicitly ruling out every one of acs_chest_pain's
    # any_of items must not leave a pending clarification — it is a
    # resolved "no" (rules.red_flags.find_incomplete_combination_candidates's
    # own rejected flag), not an open question.
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(
        _state(
            symptoms=[_symptom("ألم في الصدر")],
            negated_symptoms=[
                _symptom("ضيق تنفس"),
                _symptom("تعرق غزير"),
                _symptom("غثيان"),
                _symptom("تقيؤ"),
                _symptom("ألم منتشر للذراع أو الفك"),
            ],
        )
    )

    assert result["red_flags"] == []
    assert result["red_flag_candidates"] == []
    assert result["safety_decision"] == "NO_RED_FLAG"


def test_negated_symptoms_prevent_a_retracted_hard_match_too():
    # Coverage gap fixed by this rewrite: the previous node never passed
    # negated_symptoms to the deterministic rule layer at all, so a
    # symptom the patient explicitly retracted (state["symptoms"] never
    # shrinks — state.merge_symptoms only grows it across turns) could
    # still count as "present" for a HARD match, not just a candidate.
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(
        _state(
            symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")],
            negated_symptoms=[_symptom("ألم في الصدر")],
        )
    )

    assert result["red_flags"] == []


def test_a_candidate_needing_clarification_never_reaches_red_flags_regardless_of_llm():
    # Same "LLM never independently escalates" proof as
    # test_a_hard_rule_match_fires_independent_of_the_llms_own_verdict
    # above, for the candidate case specifically.
    set_provider(FakeProvider(responses=[red_flag_response(True, reasoning="قلق")]))
    with_llm_true = check_red_flags(_state(symptoms=[_symptom("ألم في الصدر")]))

    set_provider(FakeProvider(responses=[red_flag_response(False)]))
    with_llm_false = check_red_flags(_state(symptoms=[_symptom("ألم في الصدر")]))

    assert with_llm_true["red_flags"] == with_llm_false["red_flags"] == []
    assert with_llm_true["safety_decision"] == with_llm_false["safety_decision"] == "NEEDS_CLARIFICATION"


# --- LLM also sees unmatched_mentions, the rule engine cannot -----------------


def test_unmatched_mentions_reach_the_llm_prompt():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM], unmatched_mentions=["طنين بالأذن"]))

    prompt = provider.calls[0]["prompt"]
    assert "طنين بالأذن" in prompt


# --- pairing: id and reason are structurally paired, never positional ---------


def test_each_rule_matched_entry_carries_its_own_reason_not_a_swapped_one():
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    by_id = {entry["id"]: entry["reason"] for entry in result["red_flags"]}
    assert by_id == {
        "acs_chest_pain": _reason_ar("acs_chest_pain"),
        "pulmonary_embolism": _reason_ar("pulmonary_embolism"),
    }


def test_every_red_flag_entry_has_both_id_and_reason_keys():
    set_provider(FakeProvider(responses=[red_flag_response(True, reasoning="قلق")]))

    result = check_red_flags(
        _state(symptoms=[_symptom("ألم في الصدر"), _symptom("ضيق تنفس")])
    )

    assert len(result["red_flags"]) == 2  # 2 real rules only — LLM never adds an entry
    for entry in result["red_flags"]:
        assert set(entry) == {"id", "reason"}
        assert entry["id"]
        assert entry["reason"]


# --- node contract: partial state dict, quality tier -------------------------


def test_check_red_flags_returns_exactly_the_three_disposition_keys():
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    result = check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert set(result) == {"red_flags", "red_flag_candidates", "safety_decision"}


def test_check_red_flags_calls_the_llm_on_the_quality_tier():
    provider = FakeProvider(responses=[red_flag_response(False)])
    set_provider(provider)

    check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is RedFlagAssessment


# --- audit: every layer's verdict logged, plus the final disposition ---------


def test_check_red_flags_audit_logs_every_layer_and_the_disposition(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "nodes.check_red_flags.log_red_flag_detection", lambda **kwargs: logged.append(kwargs)
    )
    _patch_rule_layer(monkeypatch, [FAKE_MATCH])
    set_provider(FakeProvider(responses=[red_flag_response(False, reasoning=None)]))

    check_red_flags(_state(symptoms=[ORDINARY_SYMPTOM], thread_id="thread-42"))

    assert len(logged) == 1
    record = logged[0]
    assert record["thread_id"] == "thread-42"
    assert record["rule_matched"] is True
    assert record["rule_ids"] == ["fake_rule"]
    assert record["llm_matched"] is False
    assert record["llm_reasoning"] is None
    assert record["combined"] is True
    assert record["candidate_rule_ids"] == []
    assert record["safety_decision"] == "HARD_EMERGENCY"


def test_check_red_flags_audit_logs_candidate_rule_ids(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "nodes.check_red_flags.log_red_flag_detection", lambda **kwargs: logged.append(kwargs)
    )
    set_provider(FakeProvider(responses=[red_flag_response(False)]))

    check_red_flags(_state(symptoms=[_symptom("ألم في الصدر")], thread_id="thread-7"))

    assert logged[0]["candidate_rule_ids"] == ["acs_chest_pain"]
    assert logged[0]["safety_decision"] == "NEEDS_CLARIFICATION"
