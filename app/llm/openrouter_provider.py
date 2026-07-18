"""
Healix - Qwen3 via OpenRouter Provider
مزوّد LLM حقيقي (Qwen3) عبر OpenRouter، خلف نفس منفذ ``LLMProvider``.

الضمانات:
- نفس الواجهة تماماً: ``generate(system_prompt, user_prompt) -> Completion``
  فلا يتغيّر أي منطق في محرك المقابلة.
- مخرجات منظَّمة (Structured Output): يُرسل ``response_format`` بوضع
  ``json_schema`` الصارم (أو ``json_object``) لرفع موثوقية JSON إلى أقصى حد.
  إن رفض النموذج الوضع المُهيّأ يُخفَّض تلقائياً درجة واحدة
  (schema → object → off) ويبقى التلقين + إعادة المحاولة صمّام الأمان.
- توليد حتمي منخفض الحرارة، ردود متزامنة فقط (بلا Streaming).
- التحقّق من عقد JSON قبل الإرجاع؛ عند مخرجات غير صالحة يعيد المحاولة تلقائياً،
  وعند الاستنفاد يرفع ``LLMProviderError`` الذي تحوّله طبقة الـ API إلى خطأ
  منظَّم بدل الانهيار.
- ``health()``: فحص جاهزية منظَّم (المفتاح، الوصول، توفّر النموذج) ليُعرض
  لاحقاً في نقطة فحص صحّة.
- تسجيل: المزوّد، النموذج، زمن الاستجابة، الرموز (إن توفّرت)، عدد المحاولات.
  لا يُسجَّل أي نصّ من رسائل المريض إطلاقاً.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from app.config import config
from app.domain.ports import Completion
from app.exceptions import InterviewParsingError, LLMProviderError
from app.parsing.interview_parser import parse_interview_decision

logger = logging.getLogger(__name__)

# رسالة تصحيحية تُرسل عند مخرجات غير صالحة (لا تحتوي أي بيانات مريض).
_JSON_NUDGE = (
    "ردّك السابق لم يكن JSON صالحاً بالعقد المطلوب. "
    'أعد JSON فقط، دون أي نصّ خارجه، بالشكل: '
    '{"finished": false, "next_slot": "...", "question": "..."} '
    'أو {"finished": true}.'
)

# مخطط عقد المقابلة (JSON Schema صارم) — يطابق parse_interview_decision تماماً.
_INTERVIEW_SCHEMA: Dict[str, Any] = {
    "name": "interview_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "finished": {"type": "boolean"},
            "next_slot": {"type": ["string", "null"]},
            "question": {"type": ["string", "null"]},
        },
        "required": ["finished", "next_slot", "question"],
        "additionalProperties": False,
    },
}

# ترتيب التخفيض التلقائي لوضع المخرجات المنظَّمة.
_JSON_MODE_DEGRADE = {"schema": "object", "object": "off"}


class QwenOpenRouterProvider:
    """مزوّد Qwen3 عبر OpenRouter Chat Completions API (متزامن، بلا Streaming)."""

    name = "qwen_openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_attempts: Optional[int] = None,
        json_mode: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._api_key = config.OPENROUTER_API_KEY if api_key is None else api_key
        if not self._api_key:
            raise LLMProviderError(
                "OPENROUTER_API_KEY غير مضبوط — مطلوب لمزوّد qwen_openrouter."
            )

        self._model = model or config.OPENROUTER_MODEL
        self._base_url = (base_url or config.OPENROUTER_BASE_URL).rstrip("/")
        self._temperature = (
            config.LLM_TEMPERATURE if temperature is None else float(temperature)
        )
        self._max_tokens = int(max_tokens or config.LLM_MAX_TOKENS)
        self._timeout = float(timeout or config.OPENROUTER_TIMEOUT)
        self._max_attempts = max(1, int(max_attempts or config.LLM_JSON_ATTEMPTS))

        mode = (json_mode or config.OPENROUTER_JSON_MODE).strip().lower()
        self._json_mode = mode if mode in ("schema", "object", "off") else "schema"

        # قابل للحقن في الاختبارات (بلا شبكة).
        self._client = client or httpx.Client(timeout=self._timeout)

    # ------------------------------------------------------------------
    # الواجهة العامة (نفس منفذ LLMProvider)
    # ------------------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            try:
                data = self._chat(messages)
            except LLMProviderError as exc:
                if not exc.retryable or attempt >= self._max_attempts:
                    raise
                last_error = exc
                logger.warning(
                    "خطأ عابر من OpenRouter (محاولة %d/%d): %s",
                    attempt, self._max_attempts, exc,
                )
                continue

            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            content = self._extract_content(data)
            usage = data.get("usage") or {}

            logger.info(
                "LLM call | provider=%s | model=%s | json_mode=%s | latency_ms=%s | "
                "prompt_tokens=%s | completion_tokens=%s | retries=%d",
                self.name,
                data.get("model", self._model),
                self._json_mode,
                latency_ms,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                attempt - 1,
            )

            # التحقّق من العقد قبل الإرجاع — لا يُمرَّر JSON غير صالح للمحرك.
            try:
                parse_interview_decision(content)
                return Completion(text=content, model=data.get("model", self._model))
            except InterviewParsingError as exc:
                last_error = exc
                logger.warning(
                    "مخرجات LLM غير صالحة (محاولة %d/%d) — إعادة المحاولة.",
                    attempt, self._max_attempts,
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": _JSON_NUDGE})

        raise LLMProviderError(
            f"أعاد الـ LLM مخرجات غير صالحة بعد {self._max_attempts} محاولات: {last_error}"
        )

    # ------------------------------------------------------------------
    # فحص الجاهزية (يُعرض لاحقاً في نقطة صحّة)
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """
        فحص جاهزية منظَّم: المفتاح مضبوط، OpenRouter قابل للوصول،
        والنموذج المُهيّأ متاح ضمن قائمة النماذج.
        """
        report: Dict[str, Any] = {
            "provider": self.name,
            "model": self._model,
            "json_mode": self._json_mode,
            "api_key_configured": bool(self._api_key),
            "reachable": False,
            "model_accessible": None,
            "ok": False,
            "error": None,
        }

        try:
            response = self._client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            report["error"] = f"تعذّر الاتصال: {exc.__class__.__name__}"
            return report

        if response.status_code != 200:
            report["error"] = f"HTTP {response.status_code} من OpenRouter."
            return report

        report["reachable"] = True
        try:
            models = [m.get("id") for m in response.json().get("data", [])]
            report["model_accessible"] = self._model in models
        except ValueError:
            report["error"] = "قائمة النماذج ليست JSON صالحاً."
            return report

        if not report["model_accessible"]:
            report["error"] = f"النموذج {self._model} غير متاح في OpenRouter."

        report["ok"] = bool(
            report["api_key_configured"]
            and report["reachable"]
            and report["model_accessible"]
        )
        return report

    # ------------------------------------------------------------------
    # طبقة النقل
    # ------------------------------------------------------------------
    def _response_format(self) -> Optional[Dict[str, Any]]:
        if self._json_mode == "schema":
            return {"type": "json_schema", "json_schema": _INTERVIEW_SCHEMA}
        if self._json_mode == "object":
            return {"type": "json_object"}
        return None

    def _chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """استدعاء واحد لـ OpenRouter chat completions (متزامن)."""
        url = f"{self._base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        response_format = self._response_format()
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "Healix Interview Engine",
        }

        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise _retryable(f"تعذّر الاتصال بـ OpenRouter: {exc.__class__.__name__}") from exc

        if response.status_code >= 500:
            raise _retryable(f"خطأ خادم من OpenRouter (HTTP {response.status_code}).")

        if response.status_code == 429:
            raise _retryable("تجاوز حدّ المعدّل لدى OpenRouter (HTTP 429).")

        if response.status_code in (400, 404, 422) and self._json_mode != "off":
            # الأرجح أن النموذج لا يدعم وضع المخرجات المنظَّمة المُهيّأ —
            # نخفّضه درجة ونعيد نفس الطلب (التلقين + إعادة المحاولة تبقى الضمانة).
            degraded = _JSON_MODE_DEGRADE[self._json_mode]
            logger.warning(
                "رُفض response_format (HTTP %d) — تخفيض json_mode: %s → %s.",
                response.status_code, self._json_mode, degraded,
            )
            self._json_mode = degraded
            return self._chat(messages)

        if response.status_code >= 400:
            # أخطاء الإعداد/المفتاح (401/403...) لا يُعاد المحاولة عليها.
            raise LLMProviderError(
                f"رفض OpenRouter الطلب (HTTP {response.status_code})."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise _retryable("استجابة OpenRouter ليست JSON صالحاً.") from exc

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise _retryable("بنية استجابة OpenRouter غير متوقعة (لا يوجد محتوى).")
        return content if isinstance(content, str) else ""


def _retryable(message: str) -> LLMProviderError:
    """خطأ مزوّد قابل لإعادة المحاولة."""
    error = LLMProviderError(message)
    error.retryable = True
    return error
