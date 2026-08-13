# LLM Providers — Healix Interview Engine

The interview engine talks to language models through a single port
(`app/domain/ports.py → LLMProvider`):

```py.fthon
provider.generate(system_prompt, user_prompt) -> Completion(text, model)
```

The engine (`ConversationService`) never knows which provider is behind the
port. Everything below is configuration-only — **no business-logic changes are
ever needed to switch providers or models.**

---

## Switching providers

Set `LLM_PROVIDER` in `.env` and restart the service:

| Value | Provider | Notes |
|-------|----------|-------|
| `mock` *(default)* | `MockLLMProvider` | Deterministic, offline, walks the clinical checklist. |
| `qwen_openrouter` | `QwenOpenRouterProvider` | Qwen3 via the OpenRouter API. |

```env
LLM_PROVIDER=qwen_openrouter
OPENROUTER_API_KEY=sk-or-...
```

## Switching models

Any model available on OpenRouter can be selected without code changes:

```env
OPENROUTER_MODEL=qwen/qwen3-32b     # or qwen/qwen3-14b, qwen/qwen3-235b-a22b, ...
```

## All OpenRouter settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | *(empty — required)* | API key. |
| `OPENROUTER_MODEL` | `qwen/qwen3-14b` | Model id on OpenRouter. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL. |
| `OPENROUTER_TIMEOUT` | `60` | Request timeout in seconds (`LLM_TIMEOUT` kept as legacy fallback). |
| `OPENROUTER_JSON_MODE` | `schema` | Structured output: `schema` (strict json_schema) → `object` (json_object) → `off` (prompt-only). |
| `LLM_FALLBACK_TO_MOCK` | `false` | `true`: automatically fall back to the mock provider (with a warning log) when OpenRouter is unavailable. |
| `LLM_TEMPERATURE` | `0.0` | Deterministic generation. |
| `LLM_MAX_TOKENS` | `512` | Completion cap (interview JSON is small). |
| `LLM_JSON_ATTEMPTS` | `3` | Total attempts when output is invalid JSON or a transient error occurs. |

## Reliability layers (in order)

1. **Structured output** — the request carries `response_format` with a strict
   JSON schema of the interview contract. If the selected model rejects it
   (HTTP 400/404/422), the provider automatically degrades one level
   (`schema → object → off`) for the rest of the session and retries the call.
2. **Contract validation** — every completion is validated with the same
   parser the engine uses (`parse_interview_decision`) *before* being returned.
3. **Automatic retry** — invalid output triggers a corrective nudge message and
   a retry (up to `LLM_JSON_ATTEMPTS`). Transient errors (connection, 5xx, 429)
   are retried too; auth/config errors (401/403) fail fast.
4. **Structured failure** — if all attempts fail, `LLMProviderError` is raised
   and the API layer returns a structured JSON error (HTTP 502) — no crash.
5. **Optional mock fallback** — with `LLM_FALLBACK_TO_MOCK=true`, a failing
   OpenRouter call falls back to the mock provider so the interview keeps
   working (each fallback logs a warning).

## Health check

`QwenOpenRouterProvider.health()` returns a structured readiness object
(intended for a future health endpoint):

```json
{
  "provider": "qwen_openrouter",
  "model": "qwen/qwen3-14b",
  "json_mode": "schema",
  "api_key_configured": true,
  "reachable": true,
  "model_accessible": true,
  "ok": true,
  "error": null
}
```

It verifies: the API key exists, OpenRouter is reachable (`GET /models`), and
the configured model is in the returned list. `MockLLMProvider.health()` always
reports `ok: true`. When fallback is enabled the report also carries
`fallback_enabled` / `fallback_provider`.

## Logging

Each LLM call logs: provider, model, json_mode, latency (ms), prompt/completion
tokens (when reported), and retry count. **Patient text is never logged.**

---

## Adding another provider later (e.g. Gemma, Llama, local Ollama)

1. Create `app/llm/<name>_provider.py` with a class exposing:
   ```python
   class MyProvider:
       name = "my_provider"
       def generate(self, system_prompt: str, user_prompt: str) -> Completion: ...
       def health(self) -> dict: ...   # optional but recommended
   ```
   (Match the `LLMProvider` protocol — no inheritance needed.)
2. Validate the interview contract before returning
   (`parse_interview_decision`) and raise `LLMProviderError` on failure —
   reuse the retry pattern from `openrouter_provider.py`.
3. Register it in `app/llm/factory.py`:
   ```python
   if provider_name == "my_provider":
       return MyProvider()
   ```
4. Select it: `LLM_PROVIDER=my_provider`. Nothing else changes — not the
   engine, not the prompts, not Laravel, not the API contract.
