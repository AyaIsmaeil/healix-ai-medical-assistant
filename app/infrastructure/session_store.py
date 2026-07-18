"""
Healix - In-Memory Session Store
تخزين جلسات المحادثة في الذاكرة (تنفيذ بسيط لمنفذ ``SessionStore``).

آمن للخيوط عبر Lock. يمكن استبداله لاحقاً بـ Redis/DB دون تغيير طبقة التطبيق.
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, Optional

from app.domain.conversation import ConversationState


class InMemorySessionStore:
    """مخزن جلسات في الذاكرة."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[ConversationState]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str]) -> ConversationState:
        """إرجاع الجلسة الموجودة أو إنشاء واحدة جديدة عند غياب المعرّف/الجلسة."""
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            new_id = session_id or str(uuid.uuid4())
            state = ConversationState(session_id=new_id)
            self._sessions[new_id] = state
            return state

    def save(self, state: ConversationState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state
