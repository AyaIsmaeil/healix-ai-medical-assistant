import groq
import httpx
import ollama
import pytest
from pydantic import BaseModel

import llm_client
from llm_client import (
    LLMConfigError,
    LLMUnavailable,
    LLMValidationError,
    _classify_gemini_error,
    _classify_groq_error,
    _classify_ollama_error,
    _get_provider,
    _GeminiProvider,
    _GroqProvider,
    _OllamaProvider,
    _PermanentProviderError,
    _ProviderResponse,
    _resolve_provider_name,
    _strip_reasoning,
    _TransientProviderError,
    call_llm,
    set_provider,
)


class Extraction(BaseModel):
    symptom: str
    days: int


class FakeProvider:
    """Stand-in for a real provider. No network, no SDK."""

    name = "fake"

    def __init__(self, *, responses=None, errors=None, duration=0.0, clock=None):
        # `errors` is consumed one per attempt; None means "succeed".
        # `duration` advances `clock` per call, to drive the wall-clock budget.
        self._errors = list(errors or [])
        self._responses = list(responses or [])
        self._duration = duration
        self._clock = clock
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "schema": schema,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self._clock is not None and self._duration:
            # Honour the timeout we were handed, as a real provider does —
            # a fake that overruns it would test a scenario that cannot happen.
            self._clock.advance(min(self._duration, timeout_seconds))

        if self._errors:
            error = self._errors.pop(0)
            if error is not None:
                raise error

        if self._responses:
            return self._responses.pop(0)
        return _ProviderResponse(text='{"symptom": "صداع", "days": 2}')


class FakeClock:
    """Controllable monotonic clock, so budget tests don't sleep for real."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    """Pin config and neutralise sleep so tests are fast and hermetic."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("HEALIX_MODEL_FAST", "fake-fast-model")
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "gemini")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "gemini")
    monkeypatch.setenv("HEALIX_LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(llm_client, "_sleep", lambda _seconds: None)
    # Real provider instances get constructed (but never used to make network
    # calls) by the provider-selection tests below. A fresh cache per test
    # stops one test's constructed provider leaking into another's, e.g. a
    # test asserting "missing GROQ_API_KEY raises" running after a test that
    # already cached a working groq provider under the same name.
    monkeypatch.setattr(llm_client, "_provider_cache", {})
    yield
    set_provider(None)


@pytest.fixture
def clock(monkeypatch):
    """Install a fake clock; backoff sleeps advance it instead of blocking."""
    fake = FakeClock()
    monkeypatch.setattr(llm_client, "_monotonic", fake.monotonic)
    monkeypatch.setattr(llm_client, "_sleep", fake.advance)
    return fake


# --- successful structured call ---------------------------------------------


def test_structured_call_returns_validated_model():
    set_provider(FakeProvider())

    result = call_llm("أعطني الأعراض", schema=Extraction)

    assert isinstance(result, Extraction)
    assert result.symptom == "صداع"
    assert result.days == 2


def test_structured_call_passes_schema_to_provider_not_prompt():
    # Structured output must come from the provider's native mechanism,
    # never from JSON instructions smuggled into the prompt text.
    provider = FakeProvider()
    set_provider(provider)

    call_llm("أعطني الأعراض", schema=Extraction)

    assert provider.calls[0]["schema"] is Extraction
    assert "json" not in provider.calls[0]["prompt"].lower()


def test_prefers_provider_parsed_object_when_supplied():
    provider = FakeProvider(
        responses=[_ProviderResponse(text="{}", parsed=Extraction(symptom="حمى", days=1))]
    )
    set_provider(provider)

    result = call_llm("...", schema=Extraction)

    assert result == Extraction(symptom="حمى", days=1)


def test_call_without_schema_returns_raw_text():
    set_provider(FakeProvider(responses=[_ProviderResponse(text="نص حر")]))

    assert call_llm("...") == "نص حر"


def test_tier_selects_the_configured_model():
    provider = FakeProvider()
    set_provider(provider)

    call_llm("...", schema=Extraction, tier="quality")

    assert provider.calls[0]["model"] == "fake-quality-model"


# --- retry on transient error ------------------------------------------------


def test_retries_transient_error_then_succeeds():
    provider = FakeProvider(errors=[_TransientProviderError("429 rate limit"), None])
    set_provider(provider)

    result = call_llm("...", schema=Extraction)

    assert isinstance(result, Extraction)
    assert len(provider.calls) == 2


def test_does_not_retry_permanent_provider_error():
    provider = FakeProvider(errors=[_PermanentProviderError("400 bad request")])
    set_provider(provider)

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert len(provider.calls) == 1


# --- no retry on validation error --------------------------------------------


def test_validation_error_is_not_retried():
    # Response reaches us intact but doesn't fit the schema — retrying the
    # same prompt would just burn quota and hide a prompt/schema defect.
    provider = FakeProvider(
        responses=[_ProviderResponse(text='{"symptom": "صداع", "days": "not-a-number"}')]
    )
    set_provider(provider)

    with pytest.raises(LLMValidationError):
        call_llm("...", schema=Extraction)

    assert len(provider.calls) == 1


def test_malformed_json_raises_validation_error_not_unavailable():
    set_provider(FakeProvider(responses=[_ProviderResponse(text="not json at all")]))

    with pytest.raises(LLMValidationError):
        call_llm("...", schema=Extraction)


def test_empty_response_body_raises_validation_error():
    set_provider(FakeProvider(responses=[_ProviderResponse(text="   ")]))

    with pytest.raises(LLMValidationError):
        call_llm("...", schema=Extraction)


def test_never_returns_unvalidated_dict():
    provider = FakeProvider(
        responses=[_ProviderResponse(text="{}", parsed={"symptom": "غثيان", "days": 3})]
    )
    set_provider(provider)

    result = call_llm("...", schema=Extraction)

    assert isinstance(result, Extraction)
    assert not isinstance(result, dict)


# --- exhausted retries --------------------------------------------------------


def test_raises_llm_unavailable_when_retries_exhausted():
    provider = FakeProvider(
        errors=[_TransientProviderError("503 unavailable")] * 3
    )
    set_provider(provider)

    with pytest.raises(LLMUnavailable) as excinfo:
        call_llm("...", schema=Extraction)

    assert len(provider.calls) == 3  # HEALIX_LLM_MAX_ATTEMPTS
    assert "503" in str(excinfo.value)


def test_exhausted_retries_never_returns_none_or_default():
    set_provider(FakeProvider(errors=[_TransientProviderError("timeout")] * 3))

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)


# --- per-attempt timeout ------------------------------------------------------


def test_timeout_is_passed_to_every_provider_attempt(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "12.5")
    provider = FakeProvider(errors=[_TransientProviderError("504 deadline exceeded"), None])
    set_provider(provider)

    call_llm("...", schema=Extraction)

    assert [c["timeout_seconds"] for c in provider.calls] == [12.5, 12.5]


def test_timeout_has_a_default_when_env_var_is_unset(monkeypatch):
    monkeypatch.delenv("HEALIX_LLM_TIMEOUT_SECONDS", raising=False)
    # Budget raised so it doesn't cap the timeout — the default budget (25s)
    # is deliberately below the default per-attempt timeout (30s), so without
    # this the budget would win and mask the value under test.
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "300")
    provider = FakeProvider()
    set_provider(provider)

    call_llm("...", schema=Extraction)

    assert provider.calls[0]["timeout_seconds"] == 30.0


def test_default_budget_caps_the_default_per_attempt_timeout(monkeypatch):
    # Documents the shipped defaults: budget 25s < timeout 30s, so the
    # budget is what actually bounds a call out of the box.
    monkeypatch.delenv("HEALIX_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", raising=False)
    provider = FakeProvider()
    set_provider(provider)

    call_llm("...", schema=Extraction)

    assert provider.calls[0]["timeout_seconds"] == 25.0


def test_all_attempts_timing_out_raises_llm_unavailable():
    # A hung provider must surface as LLMUnavailable after the retry budget,
    # never as an indefinite block on the graph. A real provider classifies
    # its own timeout before raising (see _classify_gemini_error tests below,
    # which cover TimeoutError -> transient), so that is what is simulated.
    provider = FakeProvider(errors=[_TransientProviderError("504 deadline exceeded")] * 3)
    set_provider(provider)

    with pytest.raises(LLMUnavailable) as excinfo:
        call_llm("...", schema=Extraction)

    assert len(provider.calls) == 3
    assert "deadline" in str(excinfo.value)


def test_timeout_is_retried_and_can_succeed():
    provider = FakeProvider(
        errors=[_TransientProviderError("504 deadline exceeded"), None]
    )
    set_provider(provider)

    result = call_llm("...", schema=Extraction)

    assert isinstance(result, Extraction)
    assert len(provider.calls) == 2


def test_unclassified_provider_exception_surfaces_as_llm_unavailable():
    # Safety net: if a provider leaks an exception it did not classify, the
    # caller must still get a typed LLMError, not a raw SDK exception.
    provider = FakeProvider(errors=[RuntimeError("something the provider missed")])
    set_provider(provider)

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert len(provider.calls) == 1  # unknown failures are not retried


def test_unclassified_provider_exception_is_audited(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    set_provider(FakeProvider(errors=[RuntimeError("leaked")]))

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert logged[0]["outcome"] == "unavailable"


def test_config_error_from_provider_is_not_reshaped_as_unavailable():
    provider = FakeProvider(errors=[LLMConfigError("google-genai is not installed")])
    set_provider(provider)

    with pytest.raises(LLMConfigError):
        call_llm("...", schema=Extraction)


def test_invalid_timeout_env_var_raises_config_error(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "thirty")
    set_provider(FakeProvider())

    with pytest.raises(LLMConfigError, match="HEALIX_LLM_TIMEOUT_SECONDS"):
        call_llm("...", schema=Extraction)


def test_timeout_below_min_attempt_timeout_fails_fast_without_calling_the_provider(monkeypatch):
    # _MIN_ATTEMPT_TIMEOUT_SECONDS is 12.0 (Gemini's own SDK rejects any
    # deadline under ~10s outright — see llm_client.py's own comment on
    # this check). A configured timeout below that floor is a permanent
    # misconfiguration, not something a retry loop can ever recover from:
    # every attempt would request the identical guaranteed-failing
    # deadline. This must raise LLMConfigError immediately, before the
    # provider is ever invoked — not after burning the whole retry budget
    # on repeated permanent failures.
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "5")
    provider = FakeProvider()
    set_provider(provider)

    with pytest.raises(LLMConfigError, match="HEALIX_LLM_TIMEOUT_SECONDS"):
        call_llm("...", schema=Extraction)

    assert provider.calls == []


# --- total wall-clock budget ---------------------------------------------------


def test_budget_cuts_retries_short_before_attempts_are_exhausted(monkeypatch, clock):
    # Several attempts are allowed, but each burns 15s of a 40s budget, so
    # the budget runs out first. The graph must not keep retrying past it.
    # (Budget/timeout scaled up from an earlier 10s/4s version so several
    # attempts still clear _MIN_ATTEMPT_TIMEOUT_SECONDS before running out —
    # see that constant's docstring for why a too-small remainder isn't
    # attempted at all.)
    monkeypatch.setenv("HEALIX_LLM_MAX_ATTEMPTS", "10")
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "40")
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "15")
    provider = FakeProvider(
        errors=[_TransientProviderError("503")] * 10, duration=15.0, clock=clock
    )
    set_provider(provider)

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert len(provider.calls) < 10  # budget stopped it, not the attempt count
    assert clock.now <= 40.0


def test_final_attempt_timeout_is_shortened_to_remaining_budget(monkeypatch, clock):
    # Budget 33s, per-attempt timeout 20s. First attempt burns 20s, leaving
    # 13s — so the second attempt must be given 13s, not another full 20s,
    # or it could overrun the budget it is supposed to respect. 13s is kept
    # above _MIN_ATTEMPT_TIMEOUT_SECONDS deliberately: this test is about
    # shortening to a still-viable remainder, not about the remainder
    # running out entirely (see test_budget_below_the_minimum... below).
    monkeypatch.setenv("HEALIX_LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "33")
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "20")
    # Backoff zeroed so the assertion is exact arithmetic. Jitter genuinely
    # consumes budget too (covered by the cuts-retries-short test); here it
    # would just add noise to the number under test.
    monkeypatch.setenv("HEALIX_LLM_BACKOFF_BASE_SECONDS", "0")
    monkeypatch.setenv("HEALIX_LLM_BACKOFF_CAP_SECONDS", "0")
    provider = FakeProvider(
        errors=[_TransientProviderError("503"), None], duration=20.0, clock=clock
    )
    set_provider(provider)

    call_llm("...", schema=Extraction)

    assert provider.calls[0]["timeout_seconds"] == 20.0
    assert provider.calls[1]["timeout_seconds"] == 13.0


def test_no_attempt_may_outlive_the_remaining_budget(monkeypatch, clock):
    # Budget (15s) kept above _MIN_ATTEMPT_TIMEOUT_SECONDS so the attempt
    # actually happens; the point under test is that it's capped to the
    # budget rather than skipped.
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "15")
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "30")
    provider = FakeProvider(clock=clock)
    set_provider(provider)

    call_llm("...", schema=Extraction)

    # Budget is smaller than the per-attempt timeout, so the budget wins.
    assert provider.calls[0]["timeout_seconds"] == 15.0


def test_budget_below_the_minimum_viable_attempt_is_never_attempted(monkeypatch, clock):
    # The bug this guards against was found against the real Gemini API:
    # a shrunk per-attempt timeout under ~10s is rejected by the provider's
    # own SDK before any real work happens ("Manually set deadline Ns is
    # too short. Minimum allowed deadline is 10s."). A remaining budget
    # below _MIN_ATTEMPT_TIMEOUT_SECONDS must be treated as exhausted, the
    # same as none left — never spent on a call that cannot succeed.
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "5")  # < the 12s floor
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "30")
    provider = FakeProvider(clock=clock)
    set_provider(provider)

    with pytest.raises(LLMUnavailable, match="budget"):
        call_llm("...", schema=Extraction)

    assert provider.calls == []


def test_remaining_budget_exactly_at_the_floor_still_attempts(monkeypatch, clock):
    # Boundary check: the floor comparison is strict (<), so a remainder
    # exactly equal to it is still viable, not exhausted.
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "12")
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "30")
    provider = FakeProvider(clock=clock)
    set_provider(provider)

    call_llm("...", schema=Extraction)

    assert provider.calls[0]["timeout_seconds"] == 12.0


def test_budget_exhaustion_message_is_distinct_from_attempt_exhaustion(
    monkeypatch, clock
):
    monkeypatch.setenv("HEALIX_LLM_MAX_ATTEMPTS", "10")
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "20")
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "15")
    provider = FakeProvider(
        errors=[_TransientProviderError("503")] * 10, duration=15.0, clock=clock
    )
    set_provider(provider)

    with pytest.raises(LLMUnavailable) as excinfo:
        call_llm("...", schema=Extraction)

    message = str(excinfo.value)
    assert "budget" in message
    assert "failed after" not in message


def test_attempt_exhaustion_message_does_not_mention_budget(monkeypatch):
    # Same exception type, different cause — the message must say which,
    # because raising max_attempts and raising the budget are different fixes.
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "300")
    set_provider(FakeProvider(errors=[_TransientProviderError("503")] * 3))

    with pytest.raises(LLMUnavailable) as excinfo:
        call_llm("...", schema=Extraction)

    message = str(excinfo.value)
    assert "failed after 3 attempt(s)" in message
    assert "budget" not in message


def test_budget_exhaustion_is_audited_with_ended_by(monkeypatch, clock):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    monkeypatch.setenv("HEALIX_LLM_MAX_ATTEMPTS", "10")
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "20")
    monkeypatch.setenv("HEALIX_LLM_TIMEOUT_SECONDS", "15")
    set_provider(
        FakeProvider(errors=[_TransientProviderError("503")] * 10, duration=15.0, clock=clock)
    )

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert logged[0]["ended_by"] == "budget"
    assert logged[0]["latency_ms"] > 0


def test_attempt_exhaustion_is_audited_as_attempts(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "300")
    set_provider(FakeProvider(errors=[_TransientProviderError("503")] * 3))

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert logged[0]["ended_by"] == "attempts"


def test_fast_success_is_unaffected_by_the_budget(monkeypatch, clock):
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "25")
    provider = FakeProvider(duration=0.05, clock=clock)
    set_provider(provider)

    result = call_llm("...", schema=Extraction)

    assert isinstance(result, Extraction)
    assert len(provider.calls) == 1
    assert clock.now < 1.0


def test_successful_call_is_audited_as_ended_by_success(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    set_provider(FakeProvider())

    call_llm("...", schema=Extraction)

    assert logged[0]["ended_by"] == "success"


def test_invalid_budget_env_var_raises_config_error(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_TOTAL_BUDGET_SECONDS", "soon")
    set_provider(FakeProvider())

    with pytest.raises(LLMConfigError, match="HEALIX_LLM_TOTAL_BUDGET_SECONDS"):
        call_llm("...", schema=Extraction)


# --- audit logging ------------------------------------------------------------


def test_successful_call_is_audited_with_required_fields(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    set_provider(FakeProvider())

    call_llm("...", schema=Extraction, tier="quality", prompt_name="diagnosis", thread_id="t-1")

    assert len(logged) == 1
    entry = logged[0]
    assert entry["prompt_name"] == "diagnosis"
    assert entry["tier"] == "quality"
    assert entry["model"] == "fake-quality-model"
    assert entry["provider"] == "fake"
    assert entry["outcome"] == "success"
    assert entry["thread_id"] == "t-1"
    assert entry["raw_response"] == '{"symptom": "صداع", "days": 2}'
    assert isinstance(entry["latency_ms"], float)


def test_usage_tokens_are_audited_when_provider_reports_them(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    set_provider(
        FakeProvider(
            responses=[
                _ProviderResponse(
                    text='{"symptom": "صداع", "days": 2}',
                    usage={"prompt_tokens": 10, "response_tokens": 5},
                )
            ]
        )
    )

    call_llm("...", schema=Extraction)

    assert logged[0]["usage"] == {"prompt_tokens": 10, "response_tokens": 5}


def test_validation_failure_is_audited(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    set_provider(FakeProvider(responses=[_ProviderResponse(text="nonsense")]))

    with pytest.raises(LLMValidationError):
        call_llm("...", schema=Extraction)

    assert logged[0]["outcome"] == "validation_error"


def test_unavailable_is_audited_with_attempt_count(monkeypatch):
    logged = []
    monkeypatch.setattr(llm_client, "log_llm_call", lambda **kw: logged.append(kw))
    set_provider(FakeProvider(errors=[_TransientProviderError("503")] * 3))

    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)

    assert logged[0]["outcome"] == "unavailable"
    assert logged[0]["attempts"] == 3


def test_audit_failure_does_not_break_a_successful_call(monkeypatch):
    def exploding_logger(**_kwargs):
        raise RuntimeError("audit sink is down")

    monkeypatch.setattr(llm_client, "log_llm_call", exploding_logger)
    set_provider(FakeProvider())

    # A broken audit sink degrades the record; it must not fail the turn.
    result = call_llm("...", schema=Extraction)

    assert isinstance(result, Extraction)


def test_audit_failure_does_not_mask_the_real_exception(monkeypatch):
    def exploding_logger(**_kwargs):
        raise RuntimeError("audit sink is down")

    monkeypatch.setattr(llm_client, "log_llm_call", exploding_logger)
    set_provider(FakeProvider(errors=[_TransientProviderError("503")] * 3))

    # The caller must still see LLMUnavailable, not RuntimeError.
    with pytest.raises(LLMUnavailable):
        call_llm("...", schema=Extraction)


# --- configuration -------------------------------------------------------------


def test_missing_model_env_var_raises_config_error(monkeypatch):
    monkeypatch.delenv("HEALIX_MODEL_FAST", raising=False)
    set_provider(FakeProvider())

    with pytest.raises(LLMConfigError, match="HEALIX_MODEL_FAST"):
        call_llm("...", schema=Extraction)


def test_unknown_tier_raises_config_error():
    set_provider(FakeProvider())

    with pytest.raises(LLMConfigError, match="Unknown tier"):
        call_llm("...", schema=Extraction, tier="cheap")  # type: ignore[arg-type]


# --- provider error classification ---------------------------------------------
#
# Misclassifying here is costly in both directions: a transient error read as
# permanent fails a medical turn that a retry would have saved, and a permanent
# error read as transient burns the retry budget on a request that cannot work.


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED: Quota exceeded",
        "503 UNAVAILABLE: The model is overloaded",
        "500 INTERNAL error encountered",
        "504 deadline exceeded",
        "Rate limit reached",
    ],
)
def test_transient_provider_errors_are_classified_for_retry(message):
    assert isinstance(_classify_gemini_error(Exception(message)), _TransientProviderError)


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("deadline exceeded"), ConnectionError("connection reset by peer")],
)
def test_transient_errors_classified_by_exception_type_name(exc):
    assert isinstance(_classify_gemini_error(exc), _TransientProviderError)


@pytest.mark.parametrize(
    "message",
    [
        "400 INVALID_ARGUMENT: bad request",
        "403 PERMISSION_DENIED: API key not valid",
        "404 NOT_FOUND: model does not exist",
    ],
)
def test_permanent_provider_errors_are_not_retried(message):
    assert isinstance(_classify_gemini_error(Exception(message)), _PermanentProviderError)


# --- Groq: error classification --------------------------------------------------
#
# Unlike Gemini's, Groq's SDK gives typed exceptions with real status codes,
# so these are constructed directly rather than matched by message text.


def _groq_status_error(exc_class, status_code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "boom"}})
    return exc_class(message="boom", response=response, body=None)


def _groq_transient_cases():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return [
        _groq_status_error(groq.RateLimitError, 429),
        _groq_status_error(groq.InternalServerError, 500),
        _groq_status_error(groq.InternalServerError, 503),
        _groq_status_error(groq.ConflictError, 409),
        groq.APITimeoutError(request=request),
        groq.APIConnectionError(request=request),
    ]


@pytest.mark.parametrize("exc", _groq_transient_cases())
def test_groq_transient_errors_are_classified_for_retry(exc):
    assert isinstance(_classify_groq_error(exc), _TransientProviderError)


@pytest.mark.parametrize(
    "exc",
    [
        _groq_status_error(groq.BadRequestError, 400),
        _groq_status_error(groq.AuthenticationError, 401),
        _groq_status_error(groq.PermissionDeniedError, 403),
        _groq_status_error(groq.NotFoundError, 404),
        _groq_status_error(groq.UnprocessableEntityError, 422),
    ],
)
def test_groq_permanent_errors_are_not_retried(exc):
    assert isinstance(_classify_groq_error(exc), _PermanentProviderError)


def test_groq_unrecognized_exception_defaults_to_permanent():
    # An allow-list of transient types, not a deny-list of permanent ones:
    # an exception type this module has never seen must not be retried
    # blindly just because it wasn't explicitly marked permanent.
    assert isinstance(_classify_groq_error(RuntimeError("mystery")), _PermanentProviderError)


# --- Groq: provider ----------------------------------------------------------------


class Extraction2(BaseModel):
    symptom: str
    days: int


class _FakeGroqCompletions:
    """Stands in for client.chat.completions on a groq.Groq instance."""

    def __init__(self, *, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _fake_groq_response(content, usage=None):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    response = type("Response", (), {"choices": [choice], "usage": usage})()
    return response


def _groq_provider_with_fake_client(completions):
    provider = _GroqProvider(api_key="test-groq-key")
    provider._client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    return provider


def test_groq_provider_sends_prompt_as_a_user_message():
    completions = _FakeGroqCompletions(response=_fake_groq_response("hi"))
    provider = _groq_provider_with_fake_client(completions)

    provider.generate(
        model="llama-3.3-70b-versatile", prompt="مرحبا", schema=None, timeout_seconds=10.0
    )

    assert completions.calls[0]["messages"] == [{"role": "user", "content": "مرحبا"}]
    assert completions.calls[0]["timeout"] == 10.0
    assert "response_format" not in completions.calls[0]


def test_groq_provider_returns_response_text():
    completions = _FakeGroqCompletions(response=_fake_groq_response("نص حر"))
    provider = _groq_provider_with_fake_client(completions)

    result = provider.generate(
        model="llama-3.3-70b-versatile", prompt="...", schema=None, timeout_seconds=10.0
    )

    assert result.text == "نص حر"


def test_groq_provider_uses_strict_json_schema_for_a_supported_model():
    completions = _FakeGroqCompletions(response=_fake_groq_response('{"symptom":"a","days":1}'))
    provider = _groq_provider_with_fake_client(completions)

    provider.generate(
        model="openai/gpt-oss-20b", prompt="...", schema=Extraction2, timeout_seconds=10.0
    )

    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "Extraction2"
    assert response_format["json_schema"]["schema"] == Extraction2.model_json_schema()


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
def test_groq_provider_accepts_schema_on_both_verified_models(model):
    completions = _FakeGroqCompletions(response=_fake_groq_response('{"symptom":"a","days":1}'))
    provider = _groq_provider_with_fake_client(completions)

    provider.generate(model=model, prompt="...", schema=Extraction2, timeout_seconds=10.0)

    assert len(completions.calls) == 1


def test_groq_provider_refuses_schema_on_an_unverified_model_without_calling_the_api():
    # The core safety property this whole feature exists for: a model
    # without verified strict support must never be trusted with a schema,
    # since that would silently fall back to unenforced JSON.
    completions = _FakeGroqCompletions(response=_fake_groq_response("should not be reached"))
    provider = _groq_provider_with_fake_client(completions)

    with pytest.raises(LLMConfigError, match="llama-3.3-70b-versatile"):
        provider.generate(
            model="llama-3.3-70b-versatile",
            prompt="...",
            schema=Extraction2,
            timeout_seconds=10.0,
        )

    assert completions.calls == []


def test_groq_provider_allows_an_unverified_model_without_a_schema():
    # The restriction is specifically schema + unverified model — plain
    # text generation on any model is unaffected.
    completions = _FakeGroqCompletions(response=_fake_groq_response("ok"))
    provider = _groq_provider_with_fake_client(completions)

    result = provider.generate(
        model="llama-3.3-70b-versatile", prompt="...", schema=None, timeout_seconds=10.0
    )

    assert result.text == "ok"


def test_groq_provider_extracts_usage():
    usage = type(
        "Usage", (), {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    )()
    completions = _FakeGroqCompletions(response=_fake_groq_response("ok", usage=usage))
    provider = _groq_provider_with_fake_client(completions)

    result = provider.generate(
        model="llama-3.3-70b-versatile", prompt="...", schema=None, timeout_seconds=10.0
    )

    assert result.usage == {"prompt_tokens": 12, "response_tokens": 4, "total_tokens": 16}


def test_groq_provider_usage_is_none_when_sdk_does_not_report_it():
    completions = _FakeGroqCompletions(response=_fake_groq_response("ok", usage=None))
    provider = _groq_provider_with_fake_client(completions)

    result = provider.generate(
        model="llama-3.3-70b-versatile", prompt="...", schema=None, timeout_seconds=10.0
    )

    assert result.usage is None


def test_groq_provider_wraps_sdk_errors_via_the_classifier():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    completions = _FakeGroqCompletions(error=groq.APITimeoutError(request=request))
    provider = _groq_provider_with_fake_client(completions)

    with pytest.raises(_TransientProviderError):
        provider.generate(
            model="llama-3.3-70b-versatile", prompt="...", schema=None, timeout_seconds=10.0
        )


def test_groq_provider_end_to_end_through_call_llm(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "openai/gpt-oss-120b")
    completions = _FakeGroqCompletions(response=_fake_groq_response('{"symptom":"صداع","days":2}'))
    provider = _groq_provider_with_fake_client(completions)
    set_provider(provider)

    result = call_llm("...", schema=Extraction2, tier="quality")

    assert isinstance(result, Extraction2)
    assert result.symptom == "صداع"


# --- per-tier provider selection ----------------------------------------------


def test_resolve_provider_name_reads_the_right_env_var_per_tier(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "groq")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "gemini")

    assert _resolve_provider_name("fast") == "groq"
    assert _resolve_provider_name("quality") == "gemini"


def test_resolve_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "GROQ")

    assert _resolve_provider_name("fast") == "groq"


def test_resolve_provider_name_missing_raises_config_error(monkeypatch):
    monkeypatch.delenv("HEALIX_LLM_PROVIDER_FAST", raising=False)

    with pytest.raises(LLMConfigError, match="HEALIX_LLM_PROVIDER_FAST"):
        _resolve_provider_name("fast")


def test_resolve_provider_name_rejects_an_unsupported_provider(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "openai")

    with pytest.raises(LLMConfigError, match="openai"):
        _resolve_provider_name("fast")


def test_resolve_provider_name_rejects_unknown_tier():
    with pytest.raises(LLMConfigError, match="Unknown tier"):
        _resolve_provider_name("cheap")  # type: ignore[arg-type]


def test_get_provider_constructs_gemini_when_configured(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "gemini")
    set_provider(None)  # no override — exercise real construction

    provider = _get_provider("fast")

    assert isinstance(provider, _GeminiProvider)


def test_get_provider_constructs_groq_when_configured(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "groq")
    set_provider(None)

    provider = _get_provider("quality")

    assert isinstance(provider, _GroqProvider)


def test_get_provider_caches_by_provider_name(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "gemini")
    set_provider(None)

    assert _get_provider("fast") is _get_provider("fast")


def test_get_provider_shares_one_instance_across_tiers_using_the_same_provider(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "groq")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "groq")
    set_provider(None)

    assert _get_provider("fast") is _get_provider("quality")


def test_get_provider_uses_different_instances_for_different_providers(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "gemini")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "groq")
    set_provider(None)

    fast = _get_provider("fast")
    quality = _get_provider("quality")

    assert isinstance(fast, _GeminiProvider)
    assert isinstance(quality, _GroqProvider)
    assert fast is not quality


def test_get_provider_override_wins_regardless_of_env(monkeypatch):
    # The override is what every existing test in this file relies on: it
    # must short-circuit real provider selection entirely, even when the
    # env vars it would otherwise read are missing or invalid.
    monkeypatch.delenv("HEALIX_LLM_PROVIDER_FAST", raising=False)
    fake = FakeProvider()
    set_provider(fake)

    assert _get_provider("fast") is fake


def test_missing_gemini_api_key_raises_config_error(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_provider(None)

    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        _get_provider("fast")


def test_missing_groq_api_key_raises_config_error(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    set_provider(None)

    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        _get_provider("fast")


def test_call_llm_uses_the_tiers_configured_provider(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "gemini")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "groq")
    set_provider(None)

    fast_provider = _get_provider("fast")
    quality_provider = _get_provider("quality")

    assert fast_provider.name == "google-genai"
    assert quality_provider.name == "groq"


# --- Ollama: reasoning stripping ------------------------------------------------
#
# Empirically motivated (see _OllamaProvider's module-level note in
# llm_client.py): think=False without a schema leaves qwen3:4b's raw
# chain-of-thought in the response, terminated by a bare "</think>" with
# no matching opening tag. These tests cover exactly that shape, plus the
# well-formed-block case in case a different model/version emits one.


def test_strip_reasoning_removes_a_well_formed_think_block():
    text = "<think>let me consider this</think>الجواب النهائي"
    assert _strip_reasoning(text) == "الجواب النهائي"


def test_strip_reasoning_handles_bare_trailing_close_tag_with_no_opening_tag():
    # The actual qwen3:4b + think=False (no schema) shape, observed directly.
    text = "Okay, the user wants X. Let me think...\n</think>\n\nالجواب النهائي"
    assert _strip_reasoning(text) == "الجواب النهائي"


def test_strip_reasoning_is_a_noop_on_clean_text():
    assert _strip_reasoning("الجواب النهائي بدون أي تفكير") == "الجواب النهائي بدون أي تفكير"


def test_strip_reasoning_handles_empty_text():
    assert _strip_reasoning("") == ""


def test_strip_reasoning_uses_the_last_close_tag_when_more_than_one_appears():
    text = "أول </think> جزء </think> الجواب الحقيقي"
    assert _strip_reasoning(text) == "الجواب الحقيقي"


def test_strip_reasoning_trims_surrounding_whitespace():
    text = "<think>x</think>   \n  الجواب  \n"
    assert _strip_reasoning(text) == "الجواب"


# --- Ollama: error classification ------------------------------------------------
#
# Verified against the real local server (see module note): connection
# refusal is a builtin ConnectionError, an over-short timeout is
# httpx.ReadTimeout, and an unknown model is
# ollama.ResponseError(status_code=404).


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("Failed to connect to Ollama."),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("timed out"),
        ollama.ResponseError("internal error", status_code=500),
        ollama.ResponseError("service unavailable", status_code=503),
        ollama.ResponseError("unknown", status_code=-1),
    ],
)
def test_ollama_transient_errors_are_classified_for_retry(exc):
    assert isinstance(_classify_ollama_error(exc), _TransientProviderError)


@pytest.mark.parametrize(
    "exc",
    [
        ollama.ResponseError("model 'x' not found", status_code=404),
        ollama.ResponseError("bad request", status_code=400),
        ollama.RequestError("malformed request"),
    ],
)
def test_ollama_permanent_errors_are_not_retried(exc):
    assert isinstance(_classify_ollama_error(exc), _PermanentProviderError)


def test_ollama_unrecognized_exception_defaults_to_permanent():
    assert isinstance(_classify_ollama_error(RuntimeError("mystery")), _PermanentProviderError)


# --- Ollama: provider --------------------------------------------------------------


class Extraction3(BaseModel):
    symptom: str
    days: int


class _FakeOllamaClient:
    """Stands in for ollama.Client(...) — generate() builds a fresh one
    per call (see module note on why), so tests replace _build_client
    rather than caching a client on the provider instance."""

    def __init__(self, *, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _fake_ollama_response(content, *, prompt_tokens=None, completion_tokens=None):
    response = {"message": {"content": content}}
    if prompt_tokens is not None:
        response["prompt_eval_count"] = prompt_tokens
    if completion_tokens is not None:
        response["eval_count"] = completion_tokens
    return response


def _ollama_provider_with_fake_client(fake_client, *, captured_timeouts=None):
    provider = _OllamaProvider(base_url="http://localhost:11434")

    def fake_build_client(timeout_seconds):
        if captured_timeouts is not None:
            captured_timeouts.append(timeout_seconds)
        return fake_client

    provider._build_client = fake_build_client
    return provider


def test_ollama_provider_sends_prompt_as_a_user_message_with_thinking_disabled():
    client = _FakeOllamaClient(response=_fake_ollama_response("hi"))
    provider = _ollama_provider_with_fake_client(client)

    provider.generate(model="qwen3:4b", prompt="مرحبا", schema=None, timeout_seconds=30.0)

    assert client.calls[0]["messages"] == [{"role": "user", "content": "مرحبا"}]
    assert client.calls[0]["think"] is False
    assert "format" not in client.calls[0]


def test_ollama_provider_builds_a_fresh_client_with_the_given_timeout():
    client = _FakeOllamaClient(response=_fake_ollama_response("hi"))
    timeouts: list[float] = []
    provider = _ollama_provider_with_fake_client(client, captured_timeouts=timeouts)

    provider.generate(model="qwen3:4b", prompt="...", schema=None, timeout_seconds=17.5)

    assert timeouts == [17.5]


def test_ollama_provider_sets_format_to_the_json_schema_when_schema_given():
    client = _FakeOllamaClient(response=_fake_ollama_response('{"symptom":"a","days":1}'))
    provider = _ollama_provider_with_fake_client(client)

    provider.generate(model="qwen3:4b", prompt="...", schema=Extraction3, timeout_seconds=30.0)

    assert client.calls[0]["format"] == Extraction3.model_json_schema()


def test_ollama_provider_returns_response_text():
    client = _FakeOllamaClient(response=_fake_ollama_response("نص حر"))
    provider = _ollama_provider_with_fake_client(client)

    result = provider.generate(model="qwen3:4b", prompt="...", schema=None, timeout_seconds=30.0)

    assert result.text == "نص حر"


def test_ollama_provider_strips_leaked_reasoning_from_the_response():
    leaked = "Let me think about this carefully.\n</think>\n\nالجواب"
    client = _FakeOllamaClient(response=_fake_ollama_response(leaked))
    provider = _ollama_provider_with_fake_client(client)

    result = provider.generate(model="qwen3:4b", prompt="...", schema=None, timeout_seconds=30.0)

    assert result.text == "الجواب"


def test_ollama_provider_extracts_usage_when_both_counts_present():
    client = _FakeOllamaClient(
        response=_fake_ollama_response("ok", prompt_tokens=1173, completion_tokens=49)
    )
    provider = _ollama_provider_with_fake_client(client)

    result = provider.generate(model="qwen3:4b", prompt="...", schema=None, timeout_seconds=30.0)

    assert result.usage == {
        "prompt_tokens": 1173,
        "response_tokens": 49,
        "total_tokens": 1222,
    }


def test_ollama_provider_usage_is_none_when_sdk_does_not_report_it():
    client = _FakeOllamaClient(response=_fake_ollama_response("ok"))
    provider = _ollama_provider_with_fake_client(client)

    result = provider.generate(model="qwen3:4b", prompt="...", schema=None, timeout_seconds=30.0)

    assert result.usage is None


def test_ollama_provider_wraps_sdk_errors_via_the_classifier():
    client = _FakeOllamaClient(error=httpx.ReadTimeout("timed out"))
    provider = _ollama_provider_with_fake_client(client)

    with pytest.raises(_TransientProviderError):
        provider.generate(model="qwen3:4b", prompt="...", schema=None, timeout_seconds=30.0)


def test_ollama_provider_end_to_end_through_call_llm(monkeypatch):
    client = _FakeOllamaClient(response=_fake_ollama_response('{"symptom":"صداع","days":2}'))
    provider = _ollama_provider_with_fake_client(client)
    set_provider(provider)

    result = call_llm("...", schema=Extraction3, tier="fast")

    assert isinstance(result, Extraction3)
    assert result.symptom == "صداع"


# --- Ollama: provider selection ---------------------------------------------------


def test_get_provider_constructs_ollama_when_configured(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "ollama")
    set_provider(None)

    provider = _get_provider("fast")

    assert isinstance(provider, _OllamaProvider)


def test_missing_ollama_base_url_raises_config_error(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "ollama")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    set_provider(None)

    with pytest.raises(LLMConfigError, match="OLLAMA_BASE_URL"):
        _get_provider("fast")


def test_get_provider_distinguishes_all_three_providers(monkeypatch):
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_FAST", "ollama")
    monkeypatch.setenv("HEALIX_LLM_PROVIDER_QUALITY", "groq")
    set_provider(None)

    fast = _get_provider("fast")
    quality = _get_provider("quality")

    assert isinstance(fast, _OllamaProvider)
    assert isinstance(quality, _GroqProvider)
    assert fast is not quality
