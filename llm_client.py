"""Single entry point for every LLM call in this service.

CLAUDE.md > Conventions: no node calls a provider SDK directly, and
provider switching happens here alone. Nodes see `call_llm` and the typed
exceptions below; everything provider-shaped stays behind `_Provider`.

Deliberately NOT implemented (see the service design notes): no response
caching, no fallback to a second provider on failure, no streaming. Each
of those changes the failure semantics of a medical path and needs to be
decided explicitly rather than inherited from a client library default.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from audit.logger import log_llm_call

load_dotenv()

Tier = Literal["fast", "quality"]

# Clock and sleep are indirected so tests can drive the wall-clock budget
# deterministically without patching the global `time` module out from under
# pytest itself.
_monotonic = time.monotonic
_sleep = time.sleep


# --- exceptions ------------------------------------------------------------
#
# Callers in the medical path must never receive a silently degraded
# result: there is no None return, no default value, no partial object.
# Either a validated result comes back or one of these is raised.


class LLMError(Exception):
    """Base for every failure this module raises."""


class LLMConfigError(LLMError):
    """Required configuration (API key, model name) is missing or unusable."""


class LLMUnavailable(LLMError):
    """The provider could not be reached successfully within the retry budget."""


class LLMValidationError(LLMError):
    """The provider responded, but the response did not satisfy the schema.

    Never retried: a schema mismatch means the prompt or the schema is
    wrong, and repeating the same request just burns quota and hides the
    defect. This surfaces so it gets fixed.
    """


# --- provider abstraction ---------------------------------------------------


class _TransientProviderError(Exception):
    """Provider-side failure worth retrying (rate limit, timeout, 5xx)."""


class _PermanentProviderError(Exception):
    """Provider-side failure that retrying cannot fix (bad key, bad request)."""


@dataclass(frozen=True)
class _ProviderResponse:
    text: str
    parsed: Any | None = None  # pre-parsed object when the SDK supplies one
    usage: dict[str, int] | None = None


class _Provider(Protocol):
    """What llm_client needs from a provider. Implement this to add one.

    Implementations own all SDK-specific knowledge, including translating
    SDK exceptions into _TransientProviderError / _PermanentProviderError.
    The retry loop below deliberately knows nothing about any vendor's
    error taxonomy.
    """

    name: str

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[BaseModel] | None,
        timeout_seconds: float,
    ) -> _ProviderResponse: ...


class _GeminiProvider:
    """google-genai backed provider.

    The SDK is imported lazily so this module (and the test suite, which
    uses a fake provider) does not require google-genai to be installed.
    """

    name = "google-genai"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMConfigError(
                    "google-genai is not installed. Install it with "
                    "`pip install google-genai`, or set HEALIX_LLM_PROVIDER "
                    "to a provider that is available."
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[BaseModel] | None,
        timeout_seconds: float,
    ) -> _ProviderResponse:
        client = self._get_client()

        # google-genai expects the HTTP timeout in milliseconds. Applied
        # per request rather than on the client so the value stays a
        # per-attempt budget the retry loop controls.
        config: dict[str, Any] = {
            "http_options": {"timeout": int(timeout_seconds * 1000)},
        }
        if schema is not None:
            # Native structured output. Per CLAUDE.md the schema is never
            # described to the model in prompt text — the provider enforces
            # it, so the model cannot return a shape we did not ask for.
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except Exception as exc:
            raise _classify_gemini_error(exc) from exc

        return _ProviderResponse(
            text=getattr(response, "text", "") or "",
            parsed=getattr(response, "parsed", None),
            usage=_extract_gemini_usage(response),
        )


# Substrings that mark a retryable provider failure. Matched against the
# exception's class name and message because google-genai surfaces several
# distinct error types and the set has changed between releases; keying on
# concrete classes would silently stop matching after an SDK bump, turning
# a retryable error into a hard failure in the medical path.
_TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "resourceexhausted",
    "unavailable",
    "deadline",
    "timeout",
    "timedout",
    "internal error",
    "serviceunavailable",
    "connection",
    "temporarily",
    "overloaded",
    "rate limit",
    "ratelimit",
)


def _classify_gemini_error(exc: Exception) -> Exception:
    """Translate an SDK exception into transient vs permanent."""
    haystack = f"{type(exc).__name__} {exc}".lower()
    if any(marker in haystack for marker in _TRANSIENT_MARKERS):
        return _TransientProviderError(str(exc))
    return _PermanentProviderError(str(exc))


def _extract_gemini_usage(response: Any) -> dict[str, int] | None:
    """Pull token counts out of a response, if the SDK reported any."""
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None

    fields = {
        "prompt_tokens": "prompt_token_count",
        "response_tokens": "candidates_token_count",
        "total_tokens": "total_token_count",
    }
    usage = {}
    for out_key, attr in fields.items():
        value = getattr(metadata, attr, None)
        if isinstance(value, int):
            usage[out_key] = value
    return usage or None


# --- Groq provider -----------------------------------------------------------
#
# Verified against the Groq SDK (types/chat/completion_create_params.py) and
# https://console.groq.com/docs/structured-outputs before writing this:
# native structured output (`response_format: {"type": "json_schema", ...,
# "strict": true}`) is real schema enforcement, equivalent to Gemini's
# response_schema — NOT the older `json_object` mode, which only guarantees
# syntactically valid JSON with no schema adherence. Using json_object here
# would be exactly the "prompt-level JSON instructions" workaround this
# module exists to avoid, so it is never used.
#
# The equivalence is model-gated, though: as of this writing, strict mode is
# documented as supported only on openai/gpt-oss-20b and openai/gpt-oss-120b.
# Every other model on Groq either has no schema enforcement at all or only
# best-effort adherence. Requesting a schema against an unlisted model would
# silently downgrade the exact guarantee rules/red_flags.py relies on, so
# that combination is refused outright below rather than attempted and
# hoped to work — see scripts/verify_groq_enum.py for the live check this
# claim needs before any node actually depends on it.
_GROQ_STRICT_SCHEMA_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})


class _GroqProvider:
    """Groq (OpenAI-compatible) backed provider.

    The SDK is imported lazily so this module (and the test suite, which
    uses a fake provider) does not require groq to be installed.
    """

    name = "groq"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import groq
            except ImportError as exc:  # pragma: no cover - depends on env
                raise LLMConfigError(
                    "groq is not installed. Install it with `pip install groq`, "
                    "or configure a different provider for this tier."
                ) from exc
            self._client = groq.Groq(api_key=self._api_key)
        return self._client

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[BaseModel] | None,
        timeout_seconds: float,
    ) -> _ProviderResponse:
        client = self._get_client()

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout_seconds,
        }
        if schema is not None:
            if model not in _GROQ_STRICT_SCHEMA_MODELS:
                raise LLMConfigError(
                    f"Groq model {model!r} is not one of the models with verified "
                    f"strict structured-output support ({sorted(_GROQ_STRICT_SCHEMA_MODELS)}). "
                    "Sending a schema to it would rely on unenforced or best-effort "
                    "JSON, silently weakening the exact-match guarantee "
                    "rules/red_flags.py depends on — configure a supported model "
                    "for this tier (HEALIX_MODEL_FAST / HEALIX_MODEL_QUALITY) "
                    "instead of proceeding unguarded."
                )
            # Native structured output. Per CLAUDE.md the schema is never
            # described to the model in prompt text — the provider enforces
            # it, so the model cannot return a shape we did not ask for.
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }

        try:
            response = client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            raise _classify_groq_error(exc) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        return _ProviderResponse(text=text, usage=_extract_groq_usage(response))


def _classify_groq_error(exc: Exception) -> Exception:
    """Translate an SDK exception into transient vs permanent.

    Unlike Gemini's, Groq's SDK gives typed exceptions with real status
    codes (groq/_exceptions.py), so classification is by type rather than
    string-matching. Kept as an explicit allow-list of transient types
    (not "isinstance APIStatusError and status_code not in permanent set")
    so an SDK exception type this module has never seen defaults to
    permanent — not retried blindly.
    """
    try:
        import groq
    except ImportError:  # pragma: no cover - only reachable if groq vanished
        # _get_client() already raised LLMConfigError for a missing SDK
        # before this could run; this is just defensive.
        return _PermanentProviderError(str(exc))

    transient_types = (
        groq.RateLimitError,
        groq.InternalServerError,
        groq.APITimeoutError,
        groq.APIConnectionError,
        groq.ConflictError,  # Groq's own docs: 409 = lock timeout, retryable
    )
    if isinstance(exc, transient_types):
        return _TransientProviderError(str(exc))
    return _PermanentProviderError(str(exc))


def _extract_groq_usage(response: Any) -> dict[str, int] | None:
    """Pull token counts out of a response, if the SDK reported any."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    fields = {
        "prompt_tokens": "prompt_tokens",
        "response_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
    }
    result = {}
    for out_key, attr in fields.items():
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            result[out_key] = value
    return result or None


# --- Ollama provider -----------------------------------------------------------
#
# For local, quota-free bulk work (4GB GPU, qwen3:4b). Every claim below was
# checked empirically against a running local server (ollama 0.32.5,
# qwen3:4b) before writing this, not assumed from documentation:
#
# 1. format=<json schema dict> IS hard-enforced, not best-effort. Ollama
#    implements it on top of llama.cpp's GBNF grammar system, which masks
#    invalid tokens during sampling (confirmed via source: llama.cpp
#    src/llama-grammar.cpp; see also https://ollama.com/blog/structured-outputs).
#    Verified directly, not just from documentation: (a) asked about a
#    symptom deliberately absent from a small test enum — the model was
#    forced into an enum-valid but semantically WRONG answer, exactly the
#    signature of token-level masking; (b) explicitly instructed the model
#    to "ignore JSON formatting, respond in plain text" while a schema was
#    set — it still returned valid schema-conforming JSON. Unlike Groq,
#    this is not model-gated: the grammar sits beneath the model, so it
#    applies to any model Ollama serves, not just specific ones.
#
# 2. Qwen3 emits chain-of-thought by default. think=False is the supported
#    mechanism to disable it, but verified behavior differs by whether a
#    schema is also given:
#      - think=False + format=<schema>: clean — reasoning is suppressed
#        entirely (confirmed: ~2s response vs ~20-40s with thinking on for
#        the same prompt), content is exactly the JSON, nothing to strip.
#      - think=False + no schema: reasoning is NOT separated into the
#        dedicated `message.thinking` field the way think=True/omitted
#        does — it leaks directly into `message.content`, terminated by a
#        bare "</think>" marker with NO matching opening tag. Observed
#        directly, not assumed.
#    Since call_llm(schema=None) is a real, used path (free-text generation
#    tiers), _strip_reasoning() runs unconditionally on every response as
#    a second, independent safeguard — cheap no-op when format= already
#    produced clean output, necessary when it didn't.
#
# 3. Timeout has no per-request parameter on this SDK — ollama.Client()
#    forwards **kwargs straight to httpx.Client(**kwargs) at CONSTRUCTION,
#    and .chat() has no timeout kwarg of its own (checked via
#    inspect.signature). A fresh client is therefore constructed per
#    generate() call so each retry attempt gets the timeout the budget
#    loop actually computed for it, rather than one fixed value for the
#    whole call_llm invocation. Confirmed this genuinely enforces the
#    given value: a 0.05s client timeout raised httpx.ReadTimeout in
#    ~0.13s against the real server, not a hang.
#
# On the timeout/budget defaults (HEALIX_LLM_TIMEOUT_SECONDS=30,
# HEALIX_LLM_TOTAL_BUDGET_SECONDS=25): measured a genuine cold start (model
# not resident in VRAM, keep_alive=0) at 20.0s total for a realistic
# ~2900-char/1173-token prompt (7.2s model load + 8.5s prompt eval + 4.2s
# generation) on this 4GB card. That leaves the 25s default budget almost
# no room — a single retry after any transient failure on the first
# attempt would have under 5s left, below _MIN_ATTEMPT_TIMEOUT_SECONDS
# (12s), so the call would exhaust its budget without ever getting a
# retry. THE SHIPPED DEFAULTS ARE TOO TIGHT FOR THIS CARD. They are left
# unchanged here because they are global, not per-provider, and are
# reasonable for the cloud providers they were tuned against; a
# deployment running a tier on Ollama should raise
# HEALIX_LLM_TOTAL_BUDGET_SECONDS (45-60s) and HEALIX_LLM_TIMEOUT_SECONDS
# (35-40s) in its own .env — see .env.example. Warm-model calls are fast
# (~2s with a schema, per point 2 above); it is specifically the cold
# start plus longer prompts (a fuller conversation, not this script's
# short test message) that need the headroom.
_OLLAMA_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class _OllamaProvider:
    """Local Ollama backed provider.

    The SDK is imported lazily so this module (and the test suite, which
    uses a fake provider) does not require ollama to be installed.
    """

    name = "ollama"

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def _build_client(self, timeout_seconds: float) -> Any:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on env
            raise LLMConfigError(
                "ollama is not installed. Install it with `pip install ollama`, "
                "or configure a different provider for this tier."
            ) from exc
        # See module-level note on _OllamaProvider: timeout has no per-call
        # parameter on this SDK, only at Client construction, so a fresh
        # client is built for every attempt to honour the retry loop's
        # (possibly budget-shrunk) timeout for that specific attempt.
        return ollama.Client(host=self._base_url, timeout=timeout_seconds)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[BaseModel] | None,
        timeout_seconds: float,
    ) -> _ProviderResponse:
        client = self._build_client(timeout_seconds)

        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # Verified empirically (see module note): the fastest path and
            # the model's actual supported mechanism for suppressing
            # chain-of-thought — not a prompt-level request it could ignore.
            "think": False,
        }
        if schema is not None:
            # Native structured output via llama.cpp's GBNF grammar — see
            # module note for how this was verified to be a hard
            # constraint. Per CLAUDE.md the schema is never described to
            # the model in prompt text.
            chat_kwargs["format"] = schema.model_json_schema()

        try:
            response = client.chat(**chat_kwargs)
        except Exception as exc:
            raise _classify_ollama_error(exc) from exc

        raw_text = response["message"].get("content", "") or ""
        return _ProviderResponse(
            text=_strip_reasoning(raw_text), usage=_extract_ollama_usage(response)
        )


def _strip_reasoning(text: str) -> str:
    """Remove any chain-of-thought that leaked into the response text.

    Belt-and-braces alongside think=False — see _OllamaProvider's module
    note for exactly when this is and is not a no-op. Two shapes are
    handled: a well-formed <think>...</think> block (removed outright),
    and the observed qwen3:4b + think=False-without-schema case — a bare
    trailing "</think>" with no opening tag — handled by dropping
    everything up to and including the LAST such marker.
    """
    text = _OLLAMA_THINK_BLOCK.sub("", text)
    close_index = text.rfind("</think>")
    if close_index != -1:
        text = text[close_index + len("</think>") :]
    return text.strip()


def _classify_ollama_error(exc: Exception) -> Exception:
    """Translate an SDK/transport exception into transient vs permanent.

    Verified against the real local server, not assumed: connection
    refusal surfaces as a builtin ConnectionError (the ollama SDK's own
    wrapping), an over-short timeout as httpx.ReadTimeout, and an unknown
    model as ollama.ResponseError(status_code=404).
    """
    import httpx
    import ollama

    if isinstance(exc, (ConnectionError, httpx.TimeoutException)):
        # Ollama not running / still starting / briefly unreachable, or an
        # attempt that ran past its timeout — both worth retrying.
        return _TransientProviderError(str(exc))
    if isinstance(exc, ollama.ResponseError):
        # >=500: server-side — a 4GB card is a real OOM risk under any
        # concurrent load, and that is exactly the kind of failure a retry
        # (possibly after another request frees VRAM) can recover from.
        # <0 (-1, the SDK's own "unknown" default) is treated the same
        # way rather than assumed benign.
        if exc.status_code >= 500 or exc.status_code < 0:
            return _TransientProviderError(str(exc))
        return _PermanentProviderError(str(exc))
    if isinstance(exc, ollama.RequestError):
        return _PermanentProviderError(str(exc))
    return _PermanentProviderError(str(exc))


def _extract_ollama_usage(response: Any) -> dict[str, int] | None:
    """Pull token counts out of a response, if the SDK reported any."""
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")

    usage: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["response_tokens"] = completion_tokens
    if "prompt_tokens" in usage and "response_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["response_tokens"]
    return usage or None


# --- configuration ----------------------------------------------------------

_MODEL_ENV_VARS: dict[str, str] = {
    "fast": "HEALIX_MODEL_FAST",
    "quality": "HEALIX_MODEL_QUALITY",
}

_DEFAULT_MAX_ATTEMPTS = 4
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5
_DEFAULT_BACKOFF_CAP_SECONDS = 8.0
# Per-attempt, not per-call. The total budget below is what actually bounds
# a call_llm; this only stops any single attempt hanging. See .env.example.
_DEFAULT_TIMEOUT_SECONDS = 30.0
# Hard wall-clock ceiling for one call_llm: every attempt plus every backoff
# sleep. This is the number that must sit below Laravel's HTTP client
# timeout — without it, worst-case time is max_attempts * timeout + backoff,
# and this service keeps working on a request whose caller already gave up.
_DEFAULT_TOTAL_BUDGET_SECONDS = 25.0
# Floor below which a shrunk per-attempt timeout is not worth attempting.
# Found empirically, not chosen defensively: Gemini's own SDK rejects an
# explicit deadline under 10s with "Manually set deadline Ns is too short"
# — a 400, immediately, before any real work happens. A tiny slice of
# remaining budget must therefore count as exhausted, the same as none
# left, rather than spending a whole retry attempt on a call that cannot
# succeed. Set a little above that observed provider floor so this stays
# correct if a future provider's own minimum differs slightly.
_MIN_ATTEMPT_TIMEOUT_SECONDS = 12.0


def _resolve_model(tier: Tier) -> str:
    """Model name for a tier, from the environment. Never hardcoded."""
    try:
        env_var = _MODEL_ENV_VARS[tier]
    except KeyError:
        raise LLMConfigError(
            f"Unknown tier {tier!r}. Expected one of: {', '.join(sorted(_MODEL_ENV_VARS))}."
        ) from None

    model = os.getenv(env_var, "").strip()
    if not model:
        raise LLMConfigError(
            f"{env_var} is not set. Every tier's model is configured via the "
            "environment — see .env.example."
        )
    return model


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise LLMConfigError(f"{name} must be an integer, got {raw!r}.") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise LLMConfigError(f"{name} must be a number, got {raw!r}.") from None


_PROVIDER_ENV_VARS: dict[str, str] = {
    "fast": "HEALIX_LLM_PROVIDER_FAST",
    "quality": "HEALIX_LLM_PROVIDER_QUALITY",
}


def _build_gemini_provider() -> _Provider:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigError("GEMINI_API_KEY is not set — see .env.example.")
    return _GeminiProvider(api_key=api_key)


def _build_groq_provider() -> _Provider:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigError("GROQ_API_KEY is not set — see .env.example.")
    return _GroqProvider(api_key=api_key)


def _build_ollama_provider() -> _Provider:
    base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
    if not base_url:
        raise LLMConfigError("OLLAMA_BASE_URL is not set — see .env.example.")
    return _OllamaProvider(base_url=base_url)


# Adding a provider: one factory here, one _Provider implementation above.
# Nothing outside this module moves.
_PROVIDER_FACTORIES: dict[str, Callable[[], _Provider]] = {
    "gemini": _build_gemini_provider,
    "groq": _build_groq_provider,
    "ollama": _build_ollama_provider,
}


def _resolve_provider_name(tier: Tier) -> str:
    """Which provider a tier is configured to use, from the environment.

    Independent from _resolve_model: a tier's provider and its model are
    two separate settings, so "quality" can run gemini-2.5-pro or
    openai/gpt-oss-120b on Groq without the other tier changing at all.
    """
    try:
        env_var = _PROVIDER_ENV_VARS[tier]
    except KeyError:
        raise LLMConfigError(
            f"Unknown tier {tier!r}. Expected one of: {', '.join(sorted(_PROVIDER_ENV_VARS))}."
        ) from None

    name = os.getenv(env_var, "").strip().lower()
    if not name:
        raise LLMConfigError(
            f"{env_var} is not set. Every tier's provider is configured via the "
            "environment — see .env.example."
        )
    if name not in _PROVIDER_FACTORIES:
        raise LLMConfigError(
            f"{env_var}={name!r} is not a supported provider. Expected one of: "
            f"{', '.join(sorted(_PROVIDER_FACTORIES))}."
        )
    return name


# Real providers are cached by name (constructed once, reused across tiers
# that happen to share one) so e.g. fast=groq, quality=groq doesn't open two
# Groq clients. `_provider_override`, set only via set_provider(), takes
# priority over all of this unconditionally — tests use it to substitute a
# fake for every tier at once without needing to know which providers are
# configured.
_provider_cache: dict[str, _Provider] = {}
_provider_override: _Provider | None = None


def _get_provider(tier: Tier) -> _Provider:
    if _provider_override is not None:
        return _provider_override

    name = _resolve_provider_name(tier)
    if name not in _provider_cache:
        _provider_cache[name] = _PROVIDER_FACTORIES[name]()
    return _provider_cache[name]


def set_provider(provider: _Provider | None) -> None:
    """Override the provider for every tier. For tests; production reads configuration."""
    global _provider_override
    _provider_override = provider


# --- validation -------------------------------------------------------------


def _validate(schema: type[BaseModel], response: _ProviderResponse) -> BaseModel:
    """Coerce a provider response into `schema`, or raise LLMValidationError.

    Always ends in a real schema instance — a dict that merely looks right
    is never returned.
    """
    parsed = response.parsed

    if isinstance(parsed, schema):
        return parsed

    try:
        if isinstance(parsed, BaseModel):
            return schema.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return schema.model_validate(parsed)
        if not response.text.strip():
            raise LLMValidationError(
                f"{schema.__name__}: provider returned an empty response body."
            )
        return schema.model_validate_json(response.text)
    except ValidationError as exc:
        raise LLMValidationError(f"{schema.__name__}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMValidationError(
            f"{schema.__name__}: provider response was not valid JSON: {exc}"
        ) from exc


# --- public entry point ------------------------------------------------------


def call_llm(
    prompt: str,
    schema: type[BaseModel] | None = None,
    tier: Tier = "fast",
    *,
    prompt_name: str = "unknown",
    thread_id: str | None = None,
) -> BaseModel | str:
    """Call the configured LLM and return a validated result.

    prompt: the fully assembled prompt text (build it with
        prompts.base.build_prompt so the safety preamble is included).
    schema: when given, the provider's native structured-output mechanism
        is used and the response is validated into this model before
        return. When omitted, the raw response text is returned.
    tier: "fast" for high-frequency/low-stakes calls, "quality" for
        low-frequency/high-stakes ones (CLAUDE.md > LLM tiers).

    prompt_name and thread_id are keyword-only and exist for the audit
    trail, which CLAUDE.md requires to record which prompt ran and in
    which conversation. They carry no behaviour.

    Two independent limits bound the call:

    * HEALIX_LLM_TIMEOUT_SECONDS caps each individual attempt, so one hung
      connection cannot block the graph. A timeout is transient and retries.
    * HEALIX_LLM_TOTAL_BUDGET_SECONDS caps the whole call — every attempt
      plus every backoff sleep. When less budget remains than the
      per-attempt timeout, the attempt is given the remainder instead, so
      no attempt can outlive the call — unless the remainder is too small
      to be a viable attempt at all (_MIN_ATTEMPT_TIMEOUT_SECONDS), in
      which case the call stops rather than spend an attempt guaranteed
      to fail. This is the limit that must stay below Laravel's HTTP
      client timeout (see .env.example).

    Raises LLMValidationError if the response does not satisfy `schema`,
    LLMUnavailable if the provider could not be reached — the message
    distinguishes running out of attempts from running out of budget —
    and LLMConfigError if configuration is missing. Never returns None or
    a degraded placeholder.
    """
    model = _resolve_model(tier)  # validates tier; runs before provider lookup
    provider = _get_provider(tier)

    max_attempts = _env_int("HEALIX_LLM_MAX_ATTEMPTS", _DEFAULT_MAX_ATTEMPTS)
    backoff_base = _env_float("HEALIX_LLM_BACKOFF_BASE_SECONDS", _DEFAULT_BACKOFF_BASE_SECONDS)
    backoff_cap = _env_float("HEALIX_LLM_BACKOFF_CAP_SECONDS", _DEFAULT_BACKOFF_CAP_SECONDS)
    timeout_seconds = _env_float("HEALIX_LLM_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
    if timeout_seconds < _MIN_ATTEMPT_TIMEOUT_SECONDS:
        # Fail now, not after burning the whole retry budget. Every attempt
        # uses min(timeout_seconds, remaining) as its per-attempt deadline
        # (see the loop below) — with timeout_seconds already this low, that
        # min() is timeout_seconds itself on every attempt, so nothing about
        # retrying ever raises it. Gemini's SDK rejects any deadline under
        # ~10s outright with "Manually set deadline Ns is too short" — a
        # permanent 400 — but that message contains "deadline", one of
        # _TRANSIENT_MARKERS, so _classify_gemini_error misreads it as
        # transient and the loop retries the identical guaranteed-failing
        # deadline max_attempts times before giving up. Checked here, once,
        # before any attempt spends real time or quota on it.
        raise LLMConfigError(
            f"HEALIX_LLM_TIMEOUT_SECONDS={timeout_seconds:g} is below "
            f"_MIN_ATTEMPT_TIMEOUT_SECONDS ({_MIN_ATTEMPT_TIMEOUT_SECONDS:g}s). "
            "Every attempt would request a per-call deadline providers are "
            "known to reject outright (Gemini: 'Manually set deadline Ns is "
            "too short' below ~10s) — raise HEALIX_LLM_TIMEOUT_SECONDS in "
            ".env instead of letting the retry loop burn its whole budget "
            "repeating the same permanent failure."
        )
    total_budget = _env_float("HEALIX_LLM_TOTAL_BUDGET_SECONDS", _DEFAULT_TOTAL_BUDGET_SECONDS)

    started = _monotonic()
    last_error: Exception | None = None
    attempts_made = 0
    ended_by = "attempts"

    for attempt in range(1, max_attempts + 1):
        remaining = total_budget - (_monotonic() - started)
        if remaining < _MIN_ATTEMPT_TIMEOUT_SECONDS:
            ended_by = "budget"
            break

        # Never let one attempt outlive the whole call's budget.
        attempt_timeout = min(timeout_seconds, remaining)
        attempts_made = attempt

        try:
            response = provider.generate(
                model=model, prompt=prompt, schema=schema, timeout_seconds=attempt_timeout
            )
        except _TransientProviderError as exc:
            last_error = exc
            if attempt >= max_attempts:
                ended_by = "attempts"
                break
            delay = _backoff_delay(attempt, backoff_base, backoff_cap)
            if total_budget - (_monotonic() - started) - delay < _MIN_ATTEMPT_TIMEOUT_SECONDS:
                # What would remain after sleeping isn't enough for a viable
                # attempt anyway (see _MIN_ATTEMPT_TIMEOUT_SECONDS) — stop
                # now rather than sleep first and bail immediately after.
                ended_by = "budget"
                break
            _sleep(delay)
            continue
        except _PermanentProviderError as exc:
            last_error = exc
            ended_by = "provider_error"
            break
        except LLMError:
            # Config problems (missing SDK, bad key) are not availability
            # problems — surface them unchanged rather than reshaping them
            # into LLMUnavailable.
            raise
        except Exception as exc:  # noqa: BLE001 - safety net, see below
            # A provider leaked an exception it did not classify. Do not
            # retry something we don't understand, but do not let a raw SDK
            # error reach a caller either: the contract is that this module
            # only ever raises LLMError subclasses.
            last_error = exc
            ended_by = "provider_error"
            break

        # Reached the provider. Validation failures below are NOT retried.
        try:
            result: BaseModel | str = (
                _validate(schema, response) if schema is not None else response.text
            )
        except LLMValidationError as exc:
            _safe_audit(
                prompt_name=prompt_name,
                tier=tier,
                model=model,
                provider=provider.name,
                latency_ms=_elapsed_ms(started),
                outcome="validation_error",
                raw_response=response.text,
                usage=response.usage,
                attempts=attempt,
                thread_id=thread_id,
                error=str(exc),
                ended_by="validation",
            )
            raise

        _safe_audit(
            prompt_name=prompt_name,
            tier=tier,
            model=model,
            provider=provider.name,
            latency_ms=_elapsed_ms(started),
            outcome="success",
            raw_response=response.text,
            usage=response.usage,
            attempts=attempt,
            thread_id=thread_id,
            ended_by="success",
        )
        return result

    elapsed_ms = _elapsed_ms(started)
    _safe_audit(
        prompt_name=prompt_name,
        tier=tier,
        model=model,
        provider=provider.name,
        latency_ms=elapsed_ms,
        outcome="unavailable",
        attempts=attempts_made,
        thread_id=thread_id,
        error=str(last_error) if last_error is not None else None,
        ended_by=ended_by,
    )

    cause = f": {last_error}" if last_error is not None else ""
    if ended_by == "budget":
        # Distinct from attempt exhaustion on purpose: this one means the
        # call ran out of wall clock, so raising max_attempts would not
        # help — the budget (or whatever is making attempts slow) is what
        # needs looking at.
        detail = (
            f"exhausted its {total_budget:g}s total budget after "
            f"{attempts_made} attempt(s) ({elapsed_ms:.0f}ms elapsed)"
        )
    elif ended_by == "provider_error":
        detail = f"failed on a non-retryable error after {attempts_made} attempt(s)"
    else:
        detail = f"failed after {attempts_made} attempt(s)"

    raise LLMUnavailable(f"{provider.name}/{model} {detail}{cause}") from last_error


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter, capped.

    Jitter matters even at this scale: without it, concurrent turns that
    hit the same rate limit retry in lockstep and re-trigger it.
    """
    return random.uniform(0.0, min(cap, base * (2 ** (attempt - 1))))


def _elapsed_ms(started: float) -> float:
    return (_monotonic() - started) * 1000.0


def _safe_audit(**fields: Any) -> None:
    """Audit-log without ever letting a logging failure break the call.

    A broken audit sink degrades the record; a raised exception here would
    fail a medical turn outright. The former is strictly preferable.
    """
    try:
        log_llm_call(**fields)
    except Exception:  # noqa: BLE001 - deliberately swallowing everything
        pass
