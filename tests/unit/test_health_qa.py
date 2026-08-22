"""Tests for the Health Education Q&A module (rag/health_education/) and
its POST /health-questions route — a feature separate from the triage
graph. See docs/AHD_DATA_PROVENANCE.md for the constraints these tests
prove: never diagnose, never override an emergency, never leak raw
dataset content, and never affect POST /chat.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.route_specialty import GENERAL_REFERRAL_PHRASE
from rag.health_education.retriever import HealthEducationRetriever
from rag.health_education.service import answer_health_question

TOKEN = "test-internal-token-abc123"
AUTH_HEADERS = {"X-Healix-Internal-Token": TOKEN}


class FakeProvider:
    """Scripted provider: returns `responses` in order. Same shape as
    tests/unit/test_nodes_check_red_flags.py's own FakeProvider."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def classify_response(category: str) -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"category": category}, ensure_ascii=False))


def answer_response(answer: str) -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"answer": answer}, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_FAST", "fake-fast-model")
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _retriever(records: list[dict]) -> HealthEducationRetriever:
    return HealthEducationRetriever(records)


_ASTHMA_RECORD = {
    "question": "ما هو الربو؟",
    "answer": "الربو مرض مزمن في الشعب الهوائية يسبب ضيق تنفس ووزيز.",
    "category": "أمراض صدرية",
}

# BM25's IDF math is degenerate with a single-document corpus (every term
# then appears in 100% of documents, driving IDF negative) — real filler
# documents on unrelated topics keep the test corpus realistic enough for
# BM25 to behave the way it does against the real ~800k-row corpus.
_FILLER_RECORDS = [
    {
        "question": "شو أعراض السكري؟",
        "answer": "أعراض السكري تشمل العطش الشديد وكثرة التبول.",
        "category": "أمراض الغدد الصماء",
    },
    {
        "question": "كيف أعالج التهاب الحلق؟",
        "answer": "التهاب الحلق البسيط يتحسن بالراحة والسوائل الدافئة.",
        "category": "أنف وأذن وحنجرة",
    },
]


# --- 1. general educational question routes to the health-education module ----


def test_general_educational_question_returns_grounded_educational_answer():
    set_provider(FakeProvider(responses=[
        classify_response("educational"),
        answer_response("الربو هو مرض مزمن يصيب الشعب الهوائية."),
    ]))
    result = answer_health_question(
        "ما هو الربو؟", retriever=_retriever([_ASTHMA_RECORD, *_FILLER_RECORDS])
    )
    assert result.category == "educational"
    assert result.grounded is True
    assert result.retrieval_status == "sufficient"
    assert result.answer == "الربو هو مرض مزمن يصيب الشعب الهوائية."


# --- 2. personal symptom report is redirected to POST /chat -------------------


def test_personal_symptom_report_is_redirected_to_chat_not_answered():
    provider = FakeProvider(responses=[classify_response("personal_symptom")])
    set_provider(provider)
    result = answer_health_question("عندي ضيق نفس", retriever=_retriever([_ASTHMA_RECORD]))
    assert result.category == "triage_redirect"
    assert result.grounded is False
    # Only the classify call happened — no answer-generation call, since a
    # personal report is never answered by this module.
    assert len(provider.calls) == 1


# --- 3 & 4. emergency-like message uses the deterministic path and the LLM ----
#            never gets a chance to override it -------------------------------


def test_crisis_phrase_short_circuits_to_emergency_redirect_before_any_llm_call():
    provider = FakeProvider(responses=[])  # would raise IndexError if ever called
    set_provider(provider)
    result = answer_health_question("بدي موت", retriever=_retriever([_ASTHMA_RECORD]))
    assert result.category == "emergency_redirect"
    assert result.grounded is False
    assert len(provider.calls) == 0  # the LLM never ran — nothing to override


def test_red_flag_combination_short_circuits_to_emergency_redirect_before_any_llm_call():
    """Literal phrasing matching acs_chest_pain's cited all_of/any_of terms
    (rules/red_flags.py) — raw-text matching is literal-substring, not
    paraphrase-aware, same limitation as the rest of that module."""
    provider = FakeProvider(responses=[])
    set_provider(provider)
    result = answer_health_question(
        "عندي ألم في الصدر مع ضيق تنفس", retriever=_retriever([_ASTHMA_RECORD])
    )
    assert result.category == "emergency_redirect"
    assert len(provider.calls) == 0


# --- 5. weak retrieval returns insufficient_information ------------------------


def test_weak_retrieval_returns_insufficient_information_not_an_invented_answer():
    set_provider(FakeProvider(responses=[classify_response("educational")]))
    result = answer_health_question(
        "شو رأيك بموضوع غريب جدا ما إلو علاقة بالصحة؟", retriever=_retriever([])
    )
    assert result.category == "out_of_scope"
    assert result.retrieval_status == "insufficient"
    assert result.grounded is False


# --- 6. sufficient answers include at least one source --------------------------


def test_sufficient_answer_includes_at_least_one_source():
    set_provider(FakeProvider(responses=[
        classify_response("educational"),
        answer_response("ملخص تثقيفي."),
    ]))
    result = answer_health_question("ما هو الربو؟", retriever=_retriever([_ASTHMA_RECORD, *_FILLER_RECORDS]))
    assert len(result.sources) >= 1
    assert result.sources[0].category == "أمراض صدرية"


# --- 8 & 9. medication/dosage questions get a fixed, dosage-free response -------


def test_medication_dosage_question_gets_fixed_safe_response_with_no_llm_generated_dosage():
    provider = FakeProvider(responses=[classify_response("medication_dosage")])
    set_provider(provider)
    result = answer_health_question(
        "شو الجرعة المناسبة من الباراسيتامول؟", retriever=_retriever([_ASTHMA_RECORD])
    )
    assert result.category == "medication_safety"
    # Fixed, deterministic message — never LLM-generated for this category.
    assert len(provider.calls) == 1
    assert "مجم" not in result.answer
    assert "قرص" not in result.answer
    assert "ملغ" not in result.answer


# --- 10. the module failing does not break POST /chat --------------------------


class _FakeCompiledGraph:
    def __init__(self, return_value):
        self._return_value = return_value

    def invoke(self, input_dict, config=None):
        return self._return_value


_SUFFICIENT_CHAT_RESULT = {
    "stage": "diagnosis",
    "messages": [
        {"role": "user", "content": "عندي صداع"},
        {"role": "assistant", "content": "هاد تقييم أولي."},
    ],
    "severity": None,
    "red_flags": [],
    "diagnosis": {"status": "insufficient_information", "differential": [], "reasoning": None},
    "specialty": "طب عام",
    "specialty_laravel": GENERAL_REFERRAL_PHRASE,
    "reports": {"patient": "تقرير المريض", "doctor": "تقرير الطبيب"},
}


def test_health_questions_route_failure_does_not_break_chat_route(monkeypatch):
    monkeypatch.setenv(api_main._INTERNAL_TOKEN_ENV_VAR, TOKEN)

    def _boom(question, *, thread_id=None):
        raise RuntimeError("boom")

    # Patch the name as bound into api.main's own namespace (`from
    # rag.health_education.service import answer_health_question`) — patching
    # rag.health_education.service.answer_health_question instead would not
    # affect api.main's already-bound reference.
    monkeypatch.setattr(api_main, "answer_health_question", _boom)
    api_main.app.dependency_overrides[api_main.get_graph] = lambda: _FakeCompiledGraph(
        _SUFFICIENT_CHAT_RESULT
    )
    try:
        with TestClient(api_main.app) as client:
            broken_response = client.post(
                "/health-questions", json={"question": "ما هو الربو؟"}, headers=AUTH_HEADERS
            )
            assert broken_response.status_code == 500

            chat_response = client.post(
                "/chat", json={"thread_id": "t1", "message": "عندي صداع"}, headers=AUTH_HEADERS
            )
            assert chat_response.status_code == 200
            assert chat_response.json()["stage"] == "diagnosis"
    finally:
        api_main.app.dependency_overrides.clear()


# --- 12. no raw scores, prompts, or raw dataset answers leak into the response --


def test_response_never_leaks_raw_retrieved_text_prompts_or_scores():
    raw_marker = "نص المريض الخام السري الذي يجب ألا يظهر حرفيا"
    record = {
        "question": "ما هو الربو؟",
        "answer": raw_marker,
        "category": "أمراض صدرية",
    }
    set_provider(FakeProvider(responses=[
        classify_response("educational"),
        answer_response("الربو مرض مزمن في الشعب الهوائية."),  # clean, LLM-authored summary
    ]))
    result = answer_health_question("ما هو الربو؟", retriever=_retriever([record, *_FILLER_RECORDS]))

    assert raw_marker not in result.answer
    dumped = result.model_dump()
    assert "score" not in json.dumps(dumped)
    assert "prompt" not in json.dumps(dumped)
    # SourceReference carries only dataset/category/license — never raw text.
    assert dumped["sources"] == [
        {"dataset": "AHD: Arabic Healthcare Dataset", "category": "أمراض صدرية", "license": "CC BY 4.0"}
    ]


# --- disclaimer is always present, for every category ---------------------------


def test_every_response_includes_a_disclaimer():
    set_provider(FakeProvider(responses=[classify_response("personal_symptom")]))
    result = answer_health_question("عندي ضيق نفس", retriever=_retriever([_ASTHMA_RECORD]))
    assert result.disclaimer
    assert "معلومات تثقيفية" in result.disclaimer
