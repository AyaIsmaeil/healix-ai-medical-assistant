"""
Healix - LLM Provider Factory
اختيار مزوّد الـ LLM حسب الإعدادات (فصل بناء الكائن عن الاستخدام).
"""

from __future__ import annotations

import logging

from app.config import config
from app.domain.ports import LLMProvider
from app.exceptions import LLMProviderError
from app.llm.mock_provider import MockLLMProvider

logger = logging.getLogger(__name__)


def build_llm_provider() -> LLMProvider:
    """
    بناء مزوّد الـ LLM المُهيّأ في الإعدادات (LLM_PROVIDER).

    القيم المدعومة (غير حسّاسة لحالة الأحرف):
    - "mock"            → مزوّد حتمي بلا شبكة (يبقى متاحاً دائماً).
    - "qwen_openrouter" → Qwen3 عبر OpenRouter API.

    تبديل المزوّد = تغيير الإعدادات فقط؛ لا يتغيّر أي منطق أعمال.
    """
    provider_name = (config.LLM_PROVIDER or "mock").strip().lower()

    if provider_name == "mock":
        logger.info("استخدام مزوّد LLM الوهمي (mock).")
        return MockLLMProvider()

    if provider_name == "qwen_openrouter":
        from app.llm.openrouter_provider import QwenOpenRouterProvider

        try:
            provider = QwenOpenRouterProvider()
        except LLMProviderError as exc:
            # تعذّر تهيئة OpenRouter (مثلاً مفتاح مفقود): رجوع اختياري للوهمي.
            if config.LLM_FALLBACK_TO_MOCK:
                logger.warning(
                    "تعذّر تهيئة OpenRouter (%s) — الرجوع إلى المزوّد الوهمي.", exc
                )
                return MockLLMProvider()
            raise

        logger.info(
            "استخدام مزوّد Qwen عبر OpenRouter (model=%s, json_mode=%s).",
            config.OPENROUTER_MODEL, config.OPENROUTER_JSON_MODE,
        )

        if config.LLM_FALLBACK_TO_MOCK:
            from app.llm.fallback import FallbackLLMProvider

            logger.info("الرجوع الاحتياطي للمزوّد الوهمي مُفعَّل (LLM_FALLBACK_TO_MOCK).")
            return FallbackLLMProvider(primary=provider, fallback=MockLLMProvider())

        return provider

    raise LLMProviderError(f"مزوّد LLM غير مدعوم: {provider_name}")
