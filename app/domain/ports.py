"""
Healix - Domain Ports
الواجهات (Ports) التي تعتمد عليها طبقة التطبيق، وتُنفَّذ في الطبقات الخارجية.

استخدام ``Protocol`` يجعل الحقن (DI) بنيوياً: أي كائن يطابق الشكل يُقبل،
دون وراثة أو اقتران بالتنفيذ الملموس (تحقيق SOLID / قابلية الاختبار).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from app.domain.conversation import ConversationState


@dataclass
class Completion:
    """نتيجة توليد نصّي من مزوّد الـ LLM."""

    text: str
    model: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """منفذ مزوّد نموذج اللغة."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        """توليد ردّ نصّي (يُفترض أنه JSON) من التعليمات المُعطاة."""
        ...


@runtime_checkable
class ExtractedSymptomLike(Protocol):
    """الشكل المتوقَّع لعرض مستخرَج (يطابق ``SymptomExtractor`` الحالي)."""

    text: str
    negated: bool
    confidence: float


@runtime_checkable
class SymptomExtractorPort(Protocol):
    """منفذ استخراج الأعراض — يطابقه ``SymptomExtractor`` الحالي دون تعديل."""

    def extract(self, text: str) -> List[ExtractedSymptomLike]:
        ...


class SessionStore(Protocol):
    """منفذ تخزين جلسات المحادثة."""

    def get(self, session_id: str) -> Optional[ConversationState]:
        ...

    def get_or_create(self, session_id: Optional[str]) -> ConversationState:
        ...

    def save(self, state: ConversationState) -> None:
        ...
