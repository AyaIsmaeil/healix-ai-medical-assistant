"""
اختبارات تكامل لمزوّد Qwen عبر OpenRouter + تبديل المزوّدات من الإعدادات.

تُستخدم عمائل HTTP وهمية (بلا شبكة) لإثبات:
- عمل مزوّد Qwen بنفس منفذ LLMProvider.
- إعادة المحاولة تلقائياً عند JSON غير صالح، وخطأ منظَّم عند الاستنفاد.
- أن تبديل المزوّد يتطلّب تغيير الإعدادات فقط (لا تغيير في محرك المقابلة).
"""

import json
from dataclasses import dataclass
from typing import List

import pytest

import app.config as config_module
from app.exceptions import LLMProviderError
from app.infrastructure.session_store import InMemorySessionStore
from app.llm.factory import build_llm_provider
from app.llm.mock_provider import MockLLMProvider
from app.llm.openrouter_provider import QwenOpenRouterProvider
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService

VALID_JSON = '{"finished": false, "next_slot": "onset@صداع", "question": "منذ متى؟"}'


# ----------------------------------------------------------------------
# عملاء HTTP وهمية
# ----------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, content="", model="qwen/qwen3-32b", raw=None):
        self.status_code = status_code
        self._raw = raw if raw is not None else {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }

    def json(self):
        if self._raw is ValueError:
            raise ValueError("not json")
        return self._raw


class FakeClient:
    """عميل httpx وهمي يعيد استجابات مُجهّزة بالترتيب ويسجّل الطلبات."""

    def __init__(self, responses: List[FakeResponse]):
        self._responses = list(responses)
        self.requests = []

    def post(self, url, json=None, headers=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return self._responses.pop(0)

    def get(self, url, headers=None):
        self.requests.append({"url": url, "headers": headers, "method": "get"})
        return self._responses.pop(0)


def _provider(responses, **kwargs):
    return QwenOpenRouterProvider(
        api_key="test-key",
        model="qwen/qwen3-32b",
        base_url="https://openrouter.test/api/v1",
        max_attempts=3,
        client=FakeClient(responses),
        **kwargs,
    )


# ----------------------------------------------------------------------
# مزوّد Qwen
# ----------------------------------------------------------------------
def test_qwen_provider_returns_valid_completion():
    provider = _provider([FakeResponse(content=VALID_JSON)])
    out = provider.generate("sys", "user")
    assert out.model == "qwen/qwen3-32b"
    decision = json.loads(out.text)
    assert decision["finished"] is False
    assert decision["next_slot"] == "onset@صداع"


def test_qwen_sends_deterministic_low_temperature_payload():
    client = FakeClient([FakeResponse(content=VALID_JSON)])
    provider = QwenOpenRouterProvider(
        api_key="k", base_url="https://x.test", client=client
    )
    provider.generate("sys", "user")
    payload = client.requests[0]["json"]
    assert payload["temperature"] == 0.0
    assert payload["model"]
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert client.requests[0]["headers"]["Authorization"] == "Bearer k"


def test_retries_on_invalid_json_then_succeeds():
    provider = _provider([
        FakeResponse(content="آسف، لا أستطيع."),        # غير صالح
        FakeResponse(content=VALID_JSON),                 # صالح
    ])
    out = provider.generate("sys", "user")
    assert json.loads(out.text)["question"] == "منذ متى؟"
    # أُرسلت رسالة تصحيحية في المحاولة الثانية.
    client = provider._client
    assert len(client.requests) == 2
    second_messages = client.requests[1]["json"]["messages"]
    assert second_messages[-1]["role"] == "user"
    assert "JSON" in second_messages[-1]["content"]


def test_structured_error_after_exhausted_retries():
    provider = _provider([
        FakeResponse(content="ليس JSON 1"),
        FakeResponse(content="ليس JSON 2"),
        FakeResponse(content="ليس JSON 3"),
    ])
    with pytest.raises(LLMProviderError) as err:
        provider.generate("sys", "user")
    assert "3" in str(err.value)  # يذكر عدد المحاولات


def test_transient_server_error_is_retried():
    provider = _provider([
        FakeResponse(status_code=503),
        FakeResponse(content=VALID_JSON),
    ])
    out = provider.generate("sys", "user")
    assert json.loads(out.text)["finished"] is False


def test_client_error_is_not_retried():
    provider = _provider([FakeResponse(status_code=401)])
    with pytest.raises(LLMProviderError):
        provider.generate("sys", "user")
    assert len(provider._client.requests) == 1


def test_missing_api_key_raises():
    with pytest.raises(LLMProviderError):
        QwenOpenRouterProvider(api_key="", client=FakeClient([]))


# ----------------------------------------------------------------------
# التبديل من الإعدادات فقط
# ----------------------------------------------------------------------
def test_factory_returns_mock_by_config(monkeypatch):
    monkeypatch.setattr(config_module.config, "LLM_PROVIDER", "mock")
    assert isinstance(build_llm_provider(), MockLLMProvider)


def test_factory_returns_qwen_by_config_only(monkeypatch):
    monkeypatch.setattr(config_module.config, "LLM_PROVIDER", "QWEN_OPENROUTER")
    monkeypatch.setattr(config_module.config, "OPENROUTER_API_KEY", "test-key")
    provider = build_llm_provider()
    assert isinstance(provider, QwenOpenRouterProvider)
    assert provider.name == "qwen_openrouter"


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(config_module.config, "LLM_PROVIDER", "gpt9000")
    with pytest.raises(LLMProviderError):
        build_llm_provider()


# ----------------------------------------------------------------------
# تكامل: محرك المقابلة يعمل مع Qwen دون أي تغيير في كوده
# ----------------------------------------------------------------------
# العقد الموحّد: الأعراض تأتي من نفس ردّ الـLLM (لا مستخرج خارجي).
UNIFIED_JSON = json.dumps({
    "chief_complaint": "صداع",
    "symptoms": [{"text": "صداع", "negated": False, "confidence": 0.9}],
    "severity": None, "duration": None, "body_location": None,
    "medications": [], "allergies": [], "chronic_conditions": [],
    "family_history": [], "missing_fields": [],
    "finished": False, "next_slot": "onset@صداع", "question": "منذ متى؟",
}, ensure_ascii=False)


def test_interview_engine_unchanged_with_qwen_provider():
    provider = _provider([FakeResponse(content=UNIFIED_JSON)])
    svc = ConversationService(
        provider=provider,
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=20,
    )
    state, decision = svc.handle_message("أعاني من صداع", None)
    assert decision.finished is False
    assert decision.question == "منذ متى؟"
    assert decision.next_slot == "onset@صداع"
    assert state.asked_slots == ["onset@صداع"]
    # الاستخراج وصل من نفس الاستدعاء — بلا مستخرج مستقل.
    assert [s.text for s in state.symptoms] == ["صداع"]
    assert state.record.chief_complaint == "صداع"


def test_mock_provider_still_works_end_to_end():
    svc = ConversationService(
        provider=MockLLMProvider(),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=20,
    )
    state, decision = svc.handle_message("أعاني من صداع", None)
    assert decision.finished is False
    assert decision.next_slot == "onset@صداع"
    assert isinstance(decision.question, str) and decision.question


# ----------------------------------------------------------------------
# المخرجات المنظَّمة (Structured Output)
# ----------------------------------------------------------------------
def test_schema_mode_sends_json_schema_response_format():
    provider = _provider([FakeResponse(content=VALID_JSON)], json_mode="schema")
    provider.generate("sys", "user")
    payload = provider._client.requests[0]["json"]
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["finished"]["type"] == "boolean"


def test_object_mode_sends_json_object_response_format():
    provider = _provider([FakeResponse(content=VALID_JSON)], json_mode="object")
    provider.generate("sys", "user")
    payload = provider._client.requests[0]["json"]
    assert payload["response_format"] == {"type": "json_object"}


def test_off_mode_sends_no_response_format():
    provider = _provider([FakeResponse(content=VALID_JSON)], json_mode="off")
    provider.generate("sys", "user")
    assert "response_format" not in provider._client.requests[0]["json"]


def test_degrades_json_mode_when_model_rejects_it():
    # 400 برفض schema → object، ثم 400 أخرى → off، ثم نجاح بلا response_format.
    provider = _provider([
        FakeResponse(status_code=400),
        FakeResponse(status_code=400),
        FakeResponse(content=VALID_JSON),
    ], json_mode="schema")
    out = provider.generate("sys", "user")
    assert json.loads(out.text)["finished"] is False
    reqs = provider._client.requests
    assert len(reqs) == 3
    assert reqs[0]["json"]["response_format"]["type"] == "json_schema"
    assert reqs[1]["json"]["response_format"] == {"type": "json_object"}
    assert "response_format" not in reqs[2]["json"]
    assert provider._json_mode == "off"  # يبقى مخفَّضاً للجلسة


# ----------------------------------------------------------------------
# المهلة القابلة للضبط
# ----------------------------------------------------------------------
def test_openrouter_timeout_config_is_used(monkeypatch):
    monkeypatch.setattr(config_module.config, "OPENROUTER_TIMEOUT", 123.0)
    provider = QwenOpenRouterProvider(api_key="k", client=FakeClient([]))
    assert provider._timeout == 123.0


# ----------------------------------------------------------------------
# الرجوع الاحتياطي للوهمي (Fallback)
# ----------------------------------------------------------------------
from app.llm.fallback import FallbackLLMProvider


class _AlwaysFailingProvider:
    name = "failing"

    def generate(self, system_prompt, user_prompt):
        raise LLMProviderError("down")


def test_fallback_uses_mock_when_primary_unavailable(caplog):
    wrapper = FallbackLLMProvider(_AlwaysFailingProvider(), MockLLMProvider())
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        out = wrapper.generate("sys", '{"symptoms": [], "answered_slots": {}, "asked_slots": []}')
    assert out.model == "mock"
    assert any("الرجوع" in r.message for r in caplog.records)


def test_fallback_prefers_primary_when_healthy():
    primary = _provider([FakeResponse(content=VALID_JSON)])
    wrapper = FallbackLLMProvider(primary, MockLLMProvider())
    out = wrapper.generate("sys", "user")
    assert out.model == "qwen/qwen3-32b"


def test_factory_fallback_enabled_wraps_provider(monkeypatch):
    monkeypatch.setattr(config_module.config, "LLM_PROVIDER", "qwen_openrouter")
    monkeypatch.setattr(config_module.config, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(config_module.config, "LLM_FALLBACK_TO_MOCK", True)
    provider = build_llm_provider()
    assert isinstance(provider, FallbackLLMProvider)


def test_factory_fallback_on_missing_key_returns_mock(monkeypatch):
    monkeypatch.setattr(config_module.config, "LLM_PROVIDER", "qwen_openrouter")
    monkeypatch.setattr(config_module.config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config_module.config, "LLM_FALLBACK_TO_MOCK", True)
    assert isinstance(build_llm_provider(), MockLLMProvider)


def test_factory_no_fallback_raises_on_missing_key(monkeypatch):
    monkeypatch.setattr(config_module.config, "LLM_PROVIDER", "qwen_openrouter")
    monkeypatch.setattr(config_module.config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config_module.config, "LLM_FALLBACK_TO_MOCK", False)
    with pytest.raises(LLMProviderError):
        build_llm_provider()


# ----------------------------------------------------------------------
# فحص الجاهزية (health)
# ----------------------------------------------------------------------
def test_health_ok_when_model_listed():
    provider = _provider([
        FakeResponse(raw={"data": [{"id": "qwen/qwen3-32b"}, {"id": "other/model"}]}),
    ])
    report = provider.health()
    assert report["ok"] is True
    assert report["reachable"] is True
    assert report["model_accessible"] is True
    assert report["error"] is None


def test_health_reports_missing_model():
    provider = _provider([FakeResponse(raw={"data": [{"id": "other/model"}]})])
    report = provider.health()
    assert report["ok"] is False
    assert report["model_accessible"] is False
    assert "qwen/qwen3-32b" in report["error"]


def test_health_reports_unreachable():
    import httpx as _httpx

    class _DownClient:
        def get(self, url, headers=None):
            raise _httpx.ConnectError("no route")

    provider = QwenOpenRouterProvider(api_key="k", client=_DownClient())
    report = provider.health()
    assert report["ok"] is False
    assert report["reachable"] is False


def test_mock_health_always_ok():
    assert MockLLMProvider().health()["ok"] is True


# ----------------------------------------------------------------------
# عقد JSON قابل للتخصيص لكل مثيل (schema/validator/نص تصحيح) — يتيح
# لمستهلكين آخرين (كمحرك التقييم) استخدام المزوّد بعقد مختلف تماماً عن
# عقد المقابلة، دون أي تغيير بمحرك المقابلة نفسه (الاختبارات أعلاه لا تمرّر
# أياً من هذه المعاملات، فتبقى بالافتراضي = عقد المقابلة تماماً كالسابق).
# ----------------------------------------------------------------------
_EXTRACTION_SCHEMA = {
    "name": "custom_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"age": {"type": ["integer", "null"]}},
        "required": ["age"],
        "additionalProperties": False,
    },
}


def _always_valid(text: str) -> None:
    """مُتحقِّق مخصَّص متساهل — يقبل أي نص بلا شرط (للاختبار)."""
    return None


def test_custom_response_schema_is_sent_instead_of_interview_schema():
    provider = _provider(
        [FakeResponse(content='{"age": 30}')],
        json_mode="schema",
        response_schema=_EXTRACTION_SCHEMA,
        response_validator=_always_valid,
    )
    provider.generate("sys", "user")
    sent_schema = provider._client.requests[0]["json"]["response_format"]["json_schema"]
    assert sent_schema["name"] == "custom_extraction"
    assert "finished" not in sent_schema["schema"]["properties"]


def test_custom_response_validator_accepts_non_interview_shape():
    """بلا مُتحقِّق مخصَّص، رد كهذا كان سيُرفض (لا يطابق عقد المقابلة) ويُعاد
    المحاولة حتى الاستنفاد. بمُتحقِّق مخصَّص متساهل، يُقبل من أول محاولة."""
    provider = _provider(
        [FakeResponse(content='{"age": 30}')],
        response_validator=_always_valid,
    )
    out = provider.generate("sys", "user")
    assert json.loads(out.text) == {"age": 30}
    assert len(provider._client.requests) == 1  # بلا إعادة محاولة


def test_default_schema_is_the_clinical_interview_contract():
    """بلا معاملات، المزوّد يستخدم عقد المقابلة السريرية الموحّد — وهو نفسه
    المُعرَّف في interview_builder (مصدر واحد، لا تعريفان ينحرفان)."""
    from app.prompts.interview_builder import INTERVIEW_JSON_SCHEMA

    provider = _provider([FakeResponse(content=UNIFIED_JSON)], json_mode="schema")
    provider.generate("sys", "user")
    sent_schema = provider._client.requests[0]["json"]["response_format"]["json_schema"]
    assert sent_schema["name"] == "clinical_interview_turn"
    assert sent_schema is INTERVIEW_JSON_SCHEMA


def test_custom_nudge_message_used_on_retry():
    provider = _provider(
        [FakeResponse(content="غير صالح"), FakeResponse(content='{"age": 5}')],
        response_validator=_always_valid_then_reject_first,
        response_format_hint="نص تصحيح مخصَّص للاستخراج",
    )
    provider.generate("sys", "user")
    second_messages = provider._client.requests[1]["json"]["messages"]
    assert second_messages[-1]["content"] == "نص تصحيح مخصَّص للاستخراج"


def _always_valid_then_reject_first(text: str) -> None:
    from app.exceptions import FeatureExtractionError

    if text == "غير صالح":
        raise FeatureExtractionError("شكل خاطئ")
    return None
