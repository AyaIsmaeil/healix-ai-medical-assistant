"""Tests for api/main.py's real, authenticated POST /chat route (and
GET /health).

Mocked-graph tests override the get_graph FastAPI dependency with a fake
compiled graph rather than exercising the real checkpointer/LLM stack —
the lifespan still runs for every test (HEALIX_INTERNAL_TOKEN and a
temp-file HEALIX_SQLITE_PATH are set by the autouse fixture below so
build_checkpointer() succeeds cheaply), it just isn't what answers
graph.invoke() in those tests. The one real end-to-end test at the
bottom does NOT override get_graph — it exercises the real compiled
graph through the real HTTP route, with only the LLM provider faked.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import llm_client
from llm_client import LLMUnavailable, _ProviderResponse, set_provider
from nodes.route_specialty import GENERAL_REFERRAL_PHRASE

TOKEN = "test-internal-token-abc123"
AUTH_HEADERS = {"X-Healix-Internal-Token": TOKEN}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setenv(api_main._INTERNAL_TOKEN_ENV_VAR, TOKEN)
    monkeypatch.setenv("HEALIX_SQLITE_PATH", str(tmp_path / "test_checkpoints.sqlite"))
    monkeypatch.delenv("HEALIX_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)
    api_main.app.dependency_overrides.clear()


class _FakeCompiledGraph:
    """Stands in for the real compiled LangGraph — get_graph's override target."""

    def __init__(self, *, return_value: dict | None = None, exception: Exception | None = None):
        self._return_value = return_value
        self._exception = exception
        self.invoke_calls: list[dict] = []

    def invoke(self, input_dict, config=None):
        self.invoke_calls.append({"input": input_dict, "config": config})
        if self._exception is not None:
            raise self._exception
        return self._return_value


def _use_fake_graph(fake_graph: _FakeCompiledGraph) -> None:
    api_main.app.dependency_overrides[api_main.get_graph] = lambda: fake_graph


_SUFFICIENT_TURN_RESULT = {
    "stage": "diagnosis",
    "messages": [
        {"role": "user", "content": "عندي صداع"},
        {"role": "assistant", "content": "هاد تقييم أولي بناءً على الأعراض."},
    ],
    "severity": None,
    "red_flags": [],
    "diagnosis": {"status": "insufficient_information", "differential": [], "reasoning": None},
    "specialty": "طب عام",
    "specialty_laravel": GENERAL_REFERRAL_PHRASE,
    "reports": {"patient": "تقرير المريض", "doctor": "تقرير الطبيب"},
}


# --- valid request, mocked graph: correct shape ----------------------------------


def test_chat_with_valid_request_and_mocked_graph_returns_the_right_shape():
    fake_graph = _FakeCompiledGraph(return_value=_SUFFICIENT_TURN_RESULT)
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/chat",
            json={"thread_id": "t1", "message": "عندي صداع"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "t1"
    assert body["reply"] == "هاد تقييم أولي بناءً على الأعراض."
    assert body["stage"] == "diagnosis"
    assert body["is_crisis"] is False
    # ChatResponse.specialty carries specialty_laravel, not the raw KB
    # string — api/main.py's own gating comment explains why.
    assert body["specialty"] == GENERAL_REFERRAL_PHRASE
    assert body["reports"] == {"patient": "تقرير المريض", "doctor": "تقرير الطبيب"}
    assert len(fake_graph.invoke_calls) == 1


def test_chat_passes_patient_sex_and_message_through_to_the_graph():
    fake_graph = _FakeCompiledGraph(return_value=_SUFFICIENT_TURN_RESULT)
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        client.post(
            "/chat",
            json={"thread_id": "t1", "message": "عندي صداع", "patient_sex": "female"},
            headers=AUTH_HEADERS,
        )

    sent = fake_graph.invoke_calls[0]["input"]
    assert sent["patient_sex"] == "female"
    assert sent["messages"] == [{"role": "user", "content": "عندي صداع"}]
    assert sent["thread_id"] == "t1"


def test_chat_medical_record_summary_absent_is_forwarded_as_empty_string_not_none():
    # Regression: nodes read state.get("medical_record_summary", "") —
    # that default only applies when the KEY is absent, not when it's
    # explicitly None. Forwarding ChatRequest's None default unchanged
    # previously broke rules/red_flags.py's normalize() call.
    fake_graph = _FakeCompiledGraph(return_value=_SUFFICIENT_TURN_RESULT)
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        client.post(
            "/chat", json={"thread_id": "t1", "message": "عندي صداع"}, headers=AUTH_HEADERS
        )

    assert fake_graph.invoke_calls[0]["input"]["medical_record_summary"] == ""


# --- auth: rejected before any graph work happens ---------------------------------


def test_chat_without_auth_token_returns_401_before_any_graph_work():
    fake_graph = _FakeCompiledGraph(return_value=_SUFFICIENT_TURN_RESULT)
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        response = client.post("/chat", json={"thread_id": "t1", "message": "عندي صداع"})

    assert response.status_code == 401
    assert fake_graph.invoke_calls == []


def test_chat_with_wrong_auth_token_returns_401_before_any_graph_work():
    fake_graph = _FakeCompiledGraph(return_value=_SUFFICIENT_TURN_RESULT)
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/chat",
            json={"thread_id": "t1", "message": "عندي صداع"},
            headers={"X-Healix-Internal-Token": "not-the-real-token"},
        )

    assert response.status_code == 401
    assert fake_graph.invoke_calls == []


# --- error handling: clean response, never a leaked internal detail ---------------


def test_chat_when_graph_raises_an_llm_error_returns_a_clean_502_not_the_real_detail():
    secret_detail = "API key sk-shouldnotleak-12345 was rejected by the provider"
    fake_graph = _FakeCompiledGraph(exception=LLMUnavailable(secret_detail))
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/chat", json={"thread_id": "t1", "message": "عندي صداع"}, headers=AUTH_HEADERS
        )

    assert response.status_code == 502
    assert secret_detail not in response.text
    assert "LLMUnavailable" not in response.text
    assert response.json() == {"detail": api_main._UPSTREAM_ERROR_DETAIL}


def test_chat_when_graph_raises_an_unexpected_error_returns_a_clean_500_not_a_traceback():
    fake_graph = _FakeCompiledGraph(exception=ValueError("some internal programming detail"))
    _use_fake_graph(fake_graph)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/chat", json={"thread_id": "t1", "message": "عندي صداع"}, headers=AUTH_HEADERS
        )

    assert response.status_code == 500
    assert "some internal programming detail" not in response.text
    assert "Traceback" not in response.text
    assert response.json() == {"detail": api_main._INTERNAL_ERROR_DETAIL}


# --- GET /health: unauthenticated -------------------------------------------------


def test_health_returns_ok_without_any_auth_header():
    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- real end-to-end: real compiled graph, real HTTP route, LLM layer faked -------


class _FakeLLMProvider:
    name = "fake"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def _no_crisis_response() -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"is_crisis": False, "reasoning": None}, ensure_ascii=False))


def _extraction_response(symptoms: list[str]) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _no_red_flag_response() -> _ProviderResponse:
    return _ProviderResponse(text=json.dumps({"has_red_flag": False, "reasoning": None}, ensure_ascii=False))


def _sufficient_response() -> _ProviderResponse:
    payload = {"is_sufficient": True, "next_question": None, "reasoning": None}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _diagnosis_response(status: str, differential=(), reasoning: str | None = None) -> _ProviderResponse:
    payload = {"status": status, "differential": list(differential), "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def test_real_end_to_end_chat_round_trip_through_the_authenticated_route():
    # No get_graph override here — this is the real compiled graph
    # (real checkpointer, real rag_retrieve/diagnose/route_specialty/
    # generate_reports), reached through the real HTTP route with real
    # auth enforced. Only the LLM provider is faked. Bare "صداع" mirrors
    # tests/unit/test_graph.py's own minimal sufficient-information
    # scenario: zero rag_retrieve candidates against the real knowledge
    # base, so diagnose short-circuits with no 5th LLM call needed.
    set_provider(
        _FakeLLMProvider(
            [
                _no_crisis_response(),
                _extraction_response(["صداع"]),
                _no_red_flag_response(),
                _sufficient_response(),
            ]
        )
    )

    with TestClient(api_main.app) as client:
        response = client.post(
            "/chat",
            json={"thread_id": "real-e2e-thread", "message": "عندي صداع"},
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "real-e2e-thread"
    assert body["stage"] == "diagnosis"
    assert body["is_crisis"] is False
    # insufficient_information -> GENERAL_PRACTICE clinically -> the
    # referral phrase Laravel-facing (GENERAL_PRACTICE has no real
    # Laravel specialty — nodes/route_specialty.py's own docstring).
    assert body["specialty"] == GENERAL_REFERRAL_PHRASE
    assert body["reply"]
    assert body["reports"]["patient"]
    assert body["reports"]["doctor"]
    assert "insufficient_information" in body["reports"]["doctor"]


def test_emergency_stage_response_does_not_carry_a_stale_diagnosis_from_an_earlier_turn():
    # Regression for a real bug: diagnosis/specialty/reports have no
    # reducer (state.py) and are only ever written by
    # diagnose/route_specialty/generate_reports — none of which run on
    # an emergency turn. Before api/main.py gated these three fields on
    # stage == "diagnosis", a later emergency turn on a thread that had
    # ALREADY produced a real diagnosis silently returned that stale
    # differential/specialty/reports alongside the "go to the ER now"
    # reply — observed for real against the live service, not just
    # theorized. Two real turns on the SAME thread_id, real compiled
    # graph, real checkpointer — only the LLM provider is faked.
    set_provider(
        _FakeLLMProvider(
            [
                # Turn 1: a real Migraine differential. Symptom names here
                # are the NORMALIZED spelling (schemas.symptoms.SymptomName's
                # Literal enum is built from CANONICAL_SYMPTOMS, which holds
                # normalized forms, not migraine.json's own authored/hamza
                # spelling — CLAUDE.md > Symptom vocabulary > "Compare
                # against the normalized form, never the authored spelling").
                _no_crisis_response(),
                _extraction_response(
                    ["صداع نابض من جهه واحده", "غثيان", "حساسيه للضوء", "حساسيه للصوت"]
                ),
                _no_red_flag_response(),
                _sufficient_response(),
                _diagnosis_response("differential", differential=["Migraine"], reasoning="تطابق كامل"),
                # Turn 2: a deterministic emergency (acs_chest_pain fires
                # from the RULE layer alone — the LLM red-flag layer below
                # deliberately says has_red_flag=False, proving stage
                # becomes "emergency" independent of the LLM's own verdict,
                # same OR-combination safety rule 3 already guarantees).
                _no_crisis_response(),
                _extraction_response(["الم في الصدر", "ضيق تنفس", "تعرق غزير"]),
                _no_red_flag_response(),
            ]
        )
    )

    with TestClient(api_main.app) as client:
        first = client.post(
            "/chat",
            json={"thread_id": "stale-diagnosis-thread", "message": "عندي صداع نابض من جهة وحدة وغثيان وحساسية للضوء والصوت"},
            headers=AUTH_HEADERS,
        )
        second = client.post(
            "/chat",
            json={"thread_id": "stale-diagnosis-thread", "message": "عندي ألم بصدري وضيق تنفس وتعرق غزير"},
            headers=AUTH_HEADERS,
        )

    # Sanity check turn 1 really did produce a real, non-empty diagnosis —
    # otherwise turn 2 having none would prove nothing.
    first_body = first.json()
    assert first_body["stage"] == "diagnosis"
    # Migraine's KB specialty is عصبية; SPECIALTY_MAP sends that to
    # Laravel's real الأمراض العصبية (Neurology) row.
    assert first_body["specialty"] == "الأمراض العصبية"
    assert first_body["reports"] is not None

    second_body = second.json()
    assert second_body["stage"] == "emergency"
    assert second_body["is_crisis"] is False
    # The actual regression: none of the three stale fields leak through.
    assert second_body["diagnosis"] is None
    assert second_body["specialty"] is None
    assert second_body["reports"] is None
