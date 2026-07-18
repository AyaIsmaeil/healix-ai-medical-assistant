"""
Healix - Fallback LLM Provider
غلاف رجوع تلقائي: يحاول المزوّد الأساسي، وعند فشله يرجع للمزوّد الوهمي.

يُفعَّل من الإعدادات فقط (LLM_FALLBACK_TO_MOCK=true) ويحافظ على نفس منفذ
``LLMProvider``، فلا يعلم محرك المقابلة بوجوده أصلاً. كل رجوع يُسجَّل بتحذير.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from app.domain.ports import Completion, LLMProvider
from app.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class FallbackLLMProvider:
    """يجرّب المزوّد الأساسي ثم يرجع للاحتياطي عند تعذّره (مع تحذير)."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}+fallback:{fallback.name}"

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        try:
            return self._primary.generate(system_prompt, user_prompt)
        except LLMProviderError as exc:
            logger.warning(
                "تعذّر المزوّد الأساسي (%s) — الرجوع إلى %s. السبب: %s",
                self._primary.name, self._fallback.name, exc,
            )
            return self._fallback.generate(system_prompt, user_prompt)

    def health(self) -> Dict[str, Any]:
        """صحّة المزوّد الأساسي مع الإشارة إلى تفعيل الرجوع الاحتياطي."""
        primary_health = (
            self._primary.health() if hasattr(self._primary, "health") else {}
        )
        return {**primary_health, "fallback_enabled": True, "fallback_provider": self._fallback.name}
