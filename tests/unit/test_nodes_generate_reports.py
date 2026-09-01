import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.diagnose import diagnose
from nodes.extract_symptoms import extract_symptoms
from nodes.generate_reports import generate_reports
from nodes.rag_retrieve import rag_retrieve
from nodes.route_specialty import route_specialty


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("generate_reports must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _differential_entry(
    name,
    match_score=1.0,
    certainty="high",
    matched=(),
    missing=(),
    negated=(),
    specialties=("طب عام",),
    name_ar=None,
    ml_corroboration=None,
    source=None,
):
    entry = {
        "name": name,
        # rag/schema.py: name_ar is authored per entry, never derived from
        # name — this fallback is a test-fixture convenience only, not a
        # real translation, and callers that care about the patient-facing
        # text pass a real one explicitly (see tests below).
        "name_ar": name_ar or f"{name}-ar",
        "match_score": match_score,
        "certainty": certainty,
        "matched_symptoms": list(matched),
        "missing_symptoms": list(missing),
        "negated_symptoms": list(negated),
        "specialties": list(specialties),
    }
    if ml_corroboration is not None:
        entry["ml_corroboration"] = ml_corroboration
    if source is not None:
        entry["source"] = source
    return entry


def _diagnosis(*, status="differential", differential=(), reasoning=None):
    return {"status": status, "differential": list(differential), "reasoning": reasoning}


def _state(
    *,
    diagnosis=None,
    specialty="عصبية",
    symptoms=(),
    negated_symptoms=(),
    unmatched_mentions=(),
    red_flags=(),
    information_limited=False,
    messages=(),
    thread_id="thread-1",
):
    if diagnosis is None:
        diagnosis = _diagnosis(differential=[_differential_entry("Migraine", name_ar="الشقيقة")])
    return {
        "thread_id": thread_id,
        "diagnosis": diagnosis,
        "specialty": specialty,
        "symptoms": list(symptoms),
        "negated_symptoms": list(negated_symptoms),
        "unmatched_mentions": list(unmatched_mentions),
        "red_flags": list(red_flags),
        "information_limited": information_limited,
        "messages": list(messages),
    }


# --- node contract: partial state dict, no LLM call ------------------------------


def test_generate_reports_returns_only_expected_keys():
    result = generate_reports(_state())

    assert set(result) == {"messages", "reports", "stage"}


def test_generate_reports_sets_stage_to_diagnosis():
    result = generate_reports(_state())

    assert result["stage"] == "diagnosis"


def test_generate_reports_appends_one_assistant_message_matching_the_patient_report():
    result = generate_reports(_state())

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"
    assert result["messages"][0]["content"] == result["reports"]["patient"]


def test_generate_reports_reports_dict_has_patient_and_doctor_keys():
    result = generate_reports(_state())

    assert set(result["reports"]) == {"patient", "doctor"}


def test_generate_reports_does_not_call_the_llm():
    set_provider(_ExplodingProvider())

    generate_reports(_state())  # must not raise


# --- a clear differential produces sensible content in both registers ------------


def test_clear_differential_produces_both_reports_with_correct_content():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Migraine",
                match_score=1.0,
                certainty="high",
                matched=["صداع نابض من جهه واحده", "غثيان"],
                specialties=["عصبية"],
                name_ar="الشقيقة",
            )
        ],
        reasoning="تطابق كامل مع الصورة السريرية",
    )

    result = generate_reports(_state(diagnosis=diagnosis, specialty="عصبية"))

    patient = result["reports"]["patient"]
    doctor = result["reports"]["doctor"]

    # Patient register: name_ar only, never the raw English/Latin name.
    assert "الشقيقة" in patient
    assert "Migraine" not in patient
    assert "عصبية" in patient
    # Safety rule 1: explicit uncertainty, never a definitive diagnosis.
    assert "تشخيص" in patient and "نهائي" in patient
    # Safety rule 1: must not replace seeing a doctor.
    assert "دكتور" in patient

    # Doctor register: both, for clinical cross-referencing.
    assert "Migraine" in doctor
    assert "الشقيقة" in doctor
    assert "differential" in doctor
    assert "match_score=1.0" in doctor
    assert "certainty=high" in doctor
    assert "صداع نابض من جهه واحده" in doctor
    assert "تطابق كامل مع الصورة السريرية" in doctor


def test_patient_report_lists_every_candidate_not_just_the_top_one():
    # Safety rule 1 says "ranked possibilities" (plural) — route_specialty's
    # top-candidate-only choice is about ROUTING, not about what the
    # patient is told the differential contains.
    diagnosis = _diagnosis(
        differential=[
            _differential_entry("Migraine", match_score=1.0, certainty="high", name_ar="الشقيقة"),
            _differential_entry(
                "Tension Headache", match_score=0.5, certainty="medium", name_ar="صداع توتري"
            ),
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    patient = result["reports"]["patient"]
    assert "الشقيقة" in patient
    assert "صداع توتري" in patient
    assert "Migraine" not in patient
    assert "Tension Headache" not in patient


# --- insufficient_information: honest in both registers, no fabrication ---------


def test_insufficient_information_produces_honest_reports_in_both_registers():
    result = generate_reports(
        _state(diagnosis=_diagnosis(status="insufficient_information", differential=[]))
    )

    patient = result["reports"]["patient"]
    doctor = result["reports"]["doctor"]

    assert "ما قدرنا" in patient
    assert "دكتور" in patient
    assert "insufficient_information" in doctor


def test_insufficient_information_patient_report_names_no_disease():
    # Nothing to name a candidate from — must not fabricate one.
    result = generate_reports(
        _state(diagnosis=_diagnosis(status="insufficient_information", differential=[]))
    )

    assert "Migraine" not in result["reports"]["patient"]
    assert "الشقيقة" not in result["reports"]["patient"]


# --- unmatched_mentions surfaced in the doctor report -----------------------------


def test_doctor_report_contains_unmatched_mentions_when_present():
    result = generate_reports(_state(unmatched_mentions=["طنين بالأذن"]))

    assert "طنين بالأذن" in result["reports"]["doctor"]


def test_doctor_report_shows_placeholder_when_no_unmatched_mentions():
    result = generate_reports(_state(unmatched_mentions=[]))

    assert "لا يوجد" in result["reports"]["doctor"]


# --- patient report: no raw numbers, no clinical/field-name jargon --------------


def test_patient_report_never_contains_the_raw_match_score():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Migraine", match_score=0.6789, certainty="medium", name_ar="الشقيقة"
            )
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    assert "0.6789" not in result["reports"]["patient"]


def test_patient_report_never_contains_field_name_jargon():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Migraine",
                matched=["غثيان"],
                missing=["صداع"],
                negated=["حمى"],
                name_ar="الشقيقة",
            )
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    patient = result["reports"]["patient"]
    for jargon in ("match_score", "matched_symptoms", "missing_symptoms", "certainty="):
        assert jargon not in patient


# --- information_limited: flagged explicitly, in both registers -----------------


def test_information_limited_is_flagged_in_both_registers():
    result = generate_reports(_state(information_limited=True))

    assert "محدود" in result["reports"]["patient"]
    assert "information_limited" in result["reports"]["doctor"]


def test_information_limited_false_adds_no_caveat_to_the_patient_report():
    limited = generate_reports(_state(information_limited=True))["reports"]["patient"]
    not_limited = generate_reports(_state(information_limited=False))["reports"]["patient"]

    assert limited != not_limited


# --- red_flags: doctor-facing only, defensive (unlikely on this path) -----------


def test_red_flags_are_surfaced_in_the_doctor_report_when_present():
    result = generate_reports(
        _state(red_flags=[{"id": "acs_chest_pain", "reason": "دليل سريري احتياطي"}])
    )

    assert "دليل سريري احتياطي" in result["reports"]["doctor"]


def test_red_flags_reason_never_leaks_into_the_patient_report():
    result = generate_reports(
        _state(red_flags=[{"id": "acs_chest_pain", "reason": "دليل سريري احتياطي"}])
    )

    assert "دليل سريري احتياطي" not in result["reports"]["patient"]


def test_no_red_flags_section_when_list_is_empty():
    result = generate_reports(_state(red_flags=[]))

    assert "علامات خطر" not in result["reports"]["doctor"]


# --- ml_corroboration: doctor-only, never a number, never in the patient report --


def test_ml_corroboration_line_appears_in_doctor_report_when_present():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Hypertension",
                name_ar="ضغط الدم المرتفع",
                ml_corroboration="model_signal_present",
            )
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    assert "XGBoost" in result["reports"]["doctor"]
    assert "إشارة" in result["reports"]["doctor"]


def test_ml_corroboration_label_never_claims_to_be_a_diagnosis():
    # The label itself (what's shown right after the model name) must
    # read as "an additional signal", never phrase the model as having
    # diagnosed anything. The disclaimer is different: it legitimately
    # USES the word "تشخيص" once, but only inside an explicit negation
    # ("لا تمثل تشخيصاً مستقلاً" — "does not represent an independent
    # diagnosis") — that is the correct, required way to disclaim it, not
    # a violation of the same rule.
    from nodes.generate_reports import _ML_CORROBORATION_DISCLAIMER, _ML_CORROBORATION_LABEL

    assert "تشخيص" not in _ML_CORROBORATION_LABEL
    assert "لا تمثل تشخيص" in _ML_CORROBORATION_DISCLAIMER


def test_ml_corroboration_never_leaks_into_the_patient_report():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Hypertension",
                name_ar="ضغط الدم المرتفع",
                ml_corroboration="model_signal_present",
            )
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    patient = result["reports"]["patient"]
    assert "model_signal_present" not in patient
    assert "XGBoost" not in patient
    assert "إحصائي" not in patient


def test_no_ml_corroboration_line_when_field_absent():
    diagnosis = _diagnosis(
        differential=[_differential_entry("Hypertension", name_ar="ضغط الدم المرتفع")]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    assert "XGBoost" not in result["reports"]["doctor"]


def test_patient_report_never_contains_any_ml_corroboration_trace_even_when_absent():
    # Same assertion as the crosswalk-matched case above, but for the far
    # more common no-signal path — the patient register must be provably
    # silent about this feature existing at all, not just silent when it
    # happens to fire.
    result = generate_reports(_state())

    patient = result["reports"]["patient"]
    for trace in ("ml_corroboration", "XGBoost", "model_signal_present"):
        assert trace not in patient


# --- source: doctor-only, verbatim from the KB, never patient-facing ------------


def test_source_line_appears_in_doctor_report_when_present():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Asthma",
                name_ar="الربو",
                source="GINA — Global Strategy for Asthma Management",
            )
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    assert "المصدر:" in result["reports"]["doctor"]
    assert "GINA — Global Strategy for Asthma Management" in result["reports"]["doctor"]


def test_source_never_leaks_into_the_patient_report():
    diagnosis = _diagnosis(
        differential=[
            _differential_entry(
                "Asthma",
                name_ar="الربو",
                source="GINA — Global Strategy for Asthma Management",
            )
        ]
    )

    result = generate_reports(_state(diagnosis=diagnosis))

    patient = result["reports"]["patient"]
    assert "GINA" not in patient
    assert "المصدر" not in patient


def test_no_source_line_when_field_absent():
    diagnosis = _diagnosis(differential=[_differential_entry("Asthma", name_ar="الربو")])

    result = generate_reports(_state(diagnosis=diagnosis))

    # .get('source') falls back to the same placeholder every other
    # missing doctor-report field uses — never a KeyError, never a blank line.
    assert "المصدر: لا يوجد" in result["reports"]["doctor"]


# --- reasoning trail: reconstructed from state["messages"], no new tracking ------


def test_doctor_report_reconstructs_the_qa_trail_from_messages():
    messages = [
        {"role": "user", "content": "عندي صداع من يومين"},
        {"role": "assistant", "content": "منذ متى بدأ هذا الصداع؟"},
        {"role": "user", "content": "من الصبح ومستمر"},
    ]

    result = generate_reports(_state(messages=messages))

    doctor = result["reports"]["doctor"]
    assert "عندي صداع من يومين" in doctor
    assert "منذ متى بدأ هذا الصداع؟" in doctor
    assert "من الصبح ومستمر" in doctor


def test_doctor_report_qa_trail_placeholder_when_no_messages():
    result = generate_reports(_state(messages=[]))

    assert "لا يوجد" in result["reports"]["doctor"]


# --- real end-to-end: extract_symptoms -> rag_retrieve -> diagnose ---------------
# -> route_specialty -> generate_reports


def _extraction_response(symptoms: list[str]) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _diagnosis_response(status, differential=(), reasoning=None) -> _ProviderResponse:
    payload = {"status": status, "differential": list(differential), "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


class _FakeProvider:
    name = "fake"

    def __init__(self, *, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def test_real_end_to_end_migraine_chain_produces_sensible_reports():
    # Real, unmocked rag_retrieve/diagnose against the real
    # rag/knowledge_base/migraine.json — its name_ar is "الشقيقة" (verified
    # directly against the shipped file), so this also confirms name_ar
    # actually flows rag_retrieve -> diagnose -> generate_reports for real
    # data, not just through the hand-built _differential_entry fixture
    # used elsewhere in this file.
    migraine_symptoms = [
        "صداع نابض من جهه واحده",
        "غثيان",
        "حساسيه للضوء",
        "حساسيه للصوت",
    ]
    set_provider(
        _FakeProvider(
            responses=[
                _extraction_response(migraine_symptoms),
                _diagnosis_response(
                    "differential", differential=["Migraine"], reasoning="تطابق كامل"
                ),
            ]
        )
    )

    state = {
        "messages": [
            {
                "role": "user",
                "content": "عندي صداع نابض من جهة وحدة، وغثيان، وحساسية من الضوء والصوت",
            }
        ]
    }
    state.update(extract_symptoms(state))
    state.update(rag_retrieve(state))
    state.update(diagnose(state))
    state.update(route_specialty(state))

    assert state["specialty"] == "عصبية"
    assert state["diagnosis"]["differential"][0]["name_ar"] == "الشقيقة"

    result = generate_reports(state)

    patient = result["reports"]["patient"]
    doctor = result["reports"]["doctor"]

    # Patient register shows the Arabic name, never the raw English one.
    assert "الشقيقة" in patient
    assert "Migraine" not in patient
    assert "عصبية" in patient
    assert "دكتور" in patient

    # Doctor register keeps both, for clinical cross-referencing.
    assert "Migraine" in doctor
    assert "الشقيقة" in doctor
    assert "match_score=1.0" in doctor
    assert "عصبية" in doctor
    assert result["stage"] == "diagnosis"
    assert result["messages"][0]["content"] == patient
