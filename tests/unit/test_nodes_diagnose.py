import json

import pytest

import llm_client
from llm_client import LLMValidationError, _ProviderResponse, set_provider
from nodes.diagnose import diagnose
from nodes.extract_symptoms import extract_symptoms
from nodes.rag_retrieve import rag_retrieve
from schemas.diagnosis import build_diagnosis_schema


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("diagnose must not call the LLM for an empty candidate list")


def _diagnosis_response(status, differential=(), reasoning=None) -> _ProviderResponse:
    payload = {"status": status, "differential": list(differential), "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _candidate(
    name, match_score, matched=(), missing=(), negated=(), specialties=("طب عام",), name_ar=None
):
    return {
        "name": name,
        "name_ar": name_ar or f"{name} (ar)",
        "match_score": match_score,
        "matched_symptoms": list(matched),
        "missing_symptoms": list(missing),
        "negated_symptoms": list(negated),
        "specialties": list(specialties),
    }


def _state(*, candidate_diseases=(), thread_id="thread-1"):
    return {"candidate_diseases": list(candidate_diseases), "thread_id": thread_id}


# --- a clear top candidate produces a sensible ranked differential --------------


def test_a_clear_top_candidate_produces_a_ranked_differential():
    candidates = [
        _candidate(
            "Influenza",
            1.0,
            matched=["حمى", "سعال"],
            specialties=["طب عام"],
            name_ar="الإنفلونزا",
        ),
        _candidate(
            "Streptococcal Pharyngitis",
            0.5,
            matched=["حمى"],
            missing=["التهاب حلق"],
            specialties=["أنف وأذن وحنجرة", "طب عام"],
            name_ar="التهاب الحلق البكتيري",
        ),
    ]
    provider = FakeProvider(
        responses=[
            _diagnosis_response(
                "differential",
                differential=["Influenza", "Streptococcal Pharyngitis"],
                reasoning="اعراض متوافقة",
            )
        ]
    )
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=candidates))

    assert result == {
        "diagnosis": {
            "status": "differential",
            "differential": [
                {
                    "name": "Influenza",
                    "name_ar": "الإنفلونزا",
                    "match_score": 1.0,
                    "certainty": "high",
                    "matched_symptoms": ["حمى", "سعال"],
                    "missing_symptoms": [],
                    "negated_symptoms": [],
                    "specialties": ["طب عام"],
                },
                {
                    "name": "Streptococcal Pharyngitis",
                    "name_ar": "التهاب الحلق البكتيري",
                    "match_score": 0.5,
                    "certainty": "medium",
                    "matched_symptoms": ["حمى"],
                    "missing_symptoms": ["التهاب حلق"],
                    "negated_symptoms": [],
                    "specialties": ["أنف وأذن وحنجرة", "طب عام"],
                },
            ],
            "reasoning": "اعراض متوافقة",
        }
    }


def test_the_llm_can_select_a_subset_not_the_whole_candidate_list():
    candidates = [
        _candidate("Influenza", 1.0, matched=["حمى", "سعال"]),
        _candidate("Streptococcal Pharyngitis", 0.3, matched=["حمى"]),
    ]
    provider = FakeProvider(
        responses=[_diagnosis_response("differential", differential=["Influenza"])]
    )
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=candidates))

    names = [entry["name"] for entry in result["diagnosis"]["differential"]]
    assert names == ["Influenza"]


def test_final_order_is_by_match_score_regardless_of_the_llms_own_list_order():
    # The LLM lists the weaker candidate first — the node must still
    # present the stronger one first (safety rule 7: ranking is
    # code-computed from match_score, never left to the model).
    candidates = [
        _candidate("Influenza", 1.0),
        _candidate("Streptococcal Pharyngitis", 0.5),
    ]
    provider = FakeProvider(
        responses=[
            _diagnosis_response(
                "differential", differential=["Streptococcal Pharyngitis", "Influenza"]
            )
        ]
    )
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=candidates))

    names = [entry["name"] for entry in result["diagnosis"]["differential"]]
    assert names == ["Influenza", "Streptococcal Pharyngitis"]


# --- ml_corroboration: carried through unchanged, never shown to the LLM --------


def test_ml_corroboration_is_carried_through_when_present_on_the_candidate():
    candidate = _candidate("Hypertension", 1.0, matched=["صداع", "دوخة"])
    candidate["ml_corroboration"] = "model_signal_present"
    provider = FakeProvider(responses=[_diagnosis_response("differential", differential=["Hypertension"])])
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=[candidate]))

    assert result["diagnosis"]["differential"][0]["ml_corroboration"] == "model_signal_present"


def test_ml_corroboration_key_absent_when_not_set_on_the_candidate():
    candidates = [_candidate("Influenza", 1.0)]
    provider = FakeProvider(responses=[_diagnosis_response("differential", differential=["Influenza"])])
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=candidates))

    assert "ml_corroboration" not in result["diagnosis"]["differential"][0]


def test_ml_corroboration_never_reaches_the_llm_prompt():
    candidate = _candidate("Hypertension", 1.0, matched=["صداع", "دوخة"])
    candidate["ml_corroboration"] = "model_signal_present"
    provider = FakeProvider(responses=[_diagnosis_response("differential", differential=["Hypertension"])])
    set_provider(provider)

    diagnose(_state(candidate_diseases=[candidate]))

    assert "ml_corroboration" not in provider.calls[0]["prompt"]
    assert "model_signal_present" not in provider.calls[0]["prompt"]


# --- empty candidate list: insufficient_information, no LLM call ---------------


def test_empty_candidate_list_produces_insufficient_information():
    result = diagnose(_state(candidate_diseases=[]))

    assert result == {
        "diagnosis": {"status": "insufficient_information", "differential": [], "reasoning": None}
    }


def test_empty_candidate_list_never_calls_the_llm():
    set_provider(_ExplodingProvider())

    diagnose(_state(candidate_diseases=[]))  # must not raise


def test_the_llm_can_also_judge_a_non_empty_candidate_set_insufficient():
    # Even when candidates exist, the model may decide none of them form
    # a sensible differential (CLAUDE.md > Non-negotiable safety rule 6:
    # insufficient_information is valid, expected, not a forced pick).
    candidates = [_candidate("Streptococcal Pharyngitis", 0.3, matched=["حمى"])]
    provider = FakeProvider(
        responses=[_diagnosis_response("insufficient_information", reasoning="دليل ضعيف جدًا")]
    )
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=candidates))

    assert result == {
        "diagnosis": {
            "status": "insufficient_information",
            "differential": [],
            "reasoning": "دليل ضعيف جدًا",
        }
    }


# --- adversarial: the model cannot select a disease outside the candidate set ---


def test_the_model_cannot_select_a_disease_outside_the_candidate_set():
    # Same style as the Ollama enum adversarial verification elsewhere in
    # this project: force the model to try smuggling in a name that was
    # never among this turn's retrieved candidates, and confirm the
    # schema — not a prompt instruction — is what stops it.
    candidates = [_candidate("Influenza", 1.0, matched=["حمى"])]
    provider = FakeProvider(
        responses=[
            _diagnosis_response(
                "differential",
                # "Migraine" was never retrieved this turn — adversarially
                # forced into the response text a real provider's
                # structured-output enforcement would reject outright.
                differential=["Migraine"],
            )
        ]
    )
    set_provider(provider)

    with pytest.raises(LLMValidationError):
        diagnose(_state(candidate_diseases=candidates))


def test_the_schema_passed_to_call_llm_is_scoped_to_this_turns_candidates_only():
    candidates = [_candidate("Influenza", 1.0), _candidate("Migraine", 0.5)]
    provider = FakeProvider(
        responses=[_diagnosis_response("differential", differential=["Influenza"])]
    )
    set_provider(provider)

    diagnose(_state(candidate_diseases=candidates))

    schema_used = provider.calls[0]["schema"]
    enum_values = schema_used.model_json_schema()["properties"]["differential"]["items"]["enum"]
    assert set(enum_values) == {"Influenza", "Migraine"}


# --- node contract: partial state dict, quality tier -----------------------------


def test_diagnose_returns_only_the_diagnosis_key():
    candidates = [_candidate("Influenza", 1.0)]
    provider = FakeProvider(
        responses=[_diagnosis_response("differential", differential=["Influenza"])]
    )
    set_provider(provider)

    result = diagnose(_state(candidate_diseases=candidates))

    assert set(result) == {"diagnosis"}


def test_diagnose_calls_the_llm_on_the_quality_tier():
    candidates = [_candidate("Influenza", 1.0)]
    provider = FakeProvider(
        responses=[_diagnosis_response("differential", differential=["Influenza"])]
    )
    set_provider(provider)

    diagnose(_state(candidate_diseases=candidates))

    assert provider.calls[0]["model"] == "fake-quality-model"


# --- real end-to-end: extract_symptoms -> rag_retrieve -> diagnose -------------


def _extraction_response(symptoms: list[str]) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def test_real_end_to_end_migraine_pattern_message_produces_a_migraine_differential():
    # The same textbook migraine picture used earlier this session
    # (unilateral throbbing headache + nausea + photophobia +
    # phonophobia). schemas.symptoms.SymptomExtraction's name enum holds
    # the NORMALIZED form, so the fake extraction response must too.
    migraine_symptoms = [
        "صداع نابض من جهه واحده",
        "غثيان",
        "حساسيه للضوء",
        "حساسيه للصوت",
    ]

    set_provider(
        FakeProvider(
            responses=[
                _extraction_response(migraine_symptoms),
                _diagnosis_response("differential", differential=["Migraine"], reasoning="تطابق كامل"),
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

    assert state["candidate_diseases"][0]["name"] == "Migraine"

    result = diagnose(state)

    assert result["diagnosis"]["status"] == "differential"
    assert result["diagnosis"]["differential"][0]["name"] == "Migraine"
    assert result["diagnosis"]["differential"][0]["name_ar"] == "الشقيقة"
    assert result["diagnosis"]["differential"][0]["match_score"] == 1.0
    assert result["diagnosis"]["differential"][0]["certainty"] == "high"
