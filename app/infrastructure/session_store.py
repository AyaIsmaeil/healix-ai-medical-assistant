"""
Healix - In-Memory Session Store
تخزين جلسات المحادثة في الذاكرة مع انتهاء صلاحية وحدّ أقصى.

آمن للخيوط عبر Lock. يمكن استبداله لاحقاً بـ Redis/DB دون تغيير طبقة التطبيق
(يحقّق نفس منفذ ``SessionStore``).

لماذا انتهاء الصلاحية إلزامي
-----------------------------
الجلسات تحمل **نصّ المريض الخام** (بيانات صحّية شخصية) وسجلّه الطبي المنظَّم.
مخزن بلا انتهاء صلاحية يعني احتفاظاً غير محدود بهذه البيانات في الذاكرة —
تسرّب موارد ومشكلة خصوصية معاً. الصلاحية تُحتسب من **آخر نشاط** لا من
الإنشاء، فمقابلة طويلة نشطة لا تنتهي في منتصفها.

قيد معماري صريح (مقصود ومؤقّت)
-------------------------------
هذا المخزن **يعمل داخل عملية واحدة**. مع أكثر من عامل (worker) لكلّ عامل
ذاكرته المستقلّة، فيفقد المريض جلسته عشوائياً بين الطلبات. الحلّ الجذري هو
جعل الخدمة عديمة الحالة (Laravel يحفظ السجل) أو Redis مشترك — وكلاهما قرار
مؤجَّل. حتى ذلك الحين **يجب تشغيل الخدمة بعامل واحد**، وهذا قيد معلن لا
مفاجأة كامنة.
"""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Tuple

from app.domain.conversation import ConversationState

# افتراضات محافظة: ساعة واحدة تكفي مقابلة طبية طويلة، وألف جلسة تحدّ
# استهلاك الذاكرة دون أن تُضيّق على استخدام واقعي.
DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_SESSIONS = 1000


class InMemorySessionStore:
    """مخزن جلسات في الذاكرة مع TTL وإخلاء عند بلوغ الحدّ."""

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        # OrderedDict لتتبّع ترتيب الاستخدام: الأقدم استخداماً يُخلى أولاً.
        self._sessions: "OrderedDict[str, Tuple[ConversationState, float]]" = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._ttl = int(ttl_seconds)
        self._max = int(max_sessions)

    # ------------------------------------------------------------------
    # منفذ SessionStore
    # ------------------------------------------------------------------
    def get(self, session_id: str) -> Optional[ConversationState]:
        with self._lock:
            return self._get_live(session_id)

    def get_or_create(self, session_id: Optional[str]) -> ConversationState:
        """إرجاع الجلسة الحيّة أو إنشاء واحدة جديدة.

        الجلسة المنتهية تُعامَل كغير موجودة: يُنشأ سجلّ جديد **بنفس المعرّف**
        (حفاظاً على التوافق مع العميل)، لكنّ ``turn_count == 0`` يبقى إشارة
        يقرأها المستدعي ليُعلم المريض أنّ جلسته انتهت بدل أن يتابع بسجلّ
        فارغ صامت.
        """
        with self._lock:
            if session_id:
                existing = self._get_live(session_id)
                if existing is not None:
                    return existing

            new_id = session_id or str(uuid.uuid4())
            state = ConversationState(session_id=new_id)
            self._put(new_id, state)
            self._enforce_capacity()
            return state

    def save(self, state: ConversationState) -> None:
        """تثبيت لقطة من الحالة.

        تُخزَّن **نسخة** لا المرجع: لو احتفظ المخزن بمرجع المستدعي، لأصبح أي
        تعديل لاحق على الكائن المُعاد تعديلاً صامتاً لما هو مثبَّت — وهو
        بالضبط صنف التسرّب المرجعي الذي سبّب تكرار الرسائل عند إعادة المحاولة.
        """
        with self._lock:
            self._put(state.session_id, deepcopy(state))
            self._enforce_capacity()

    # ------------------------------------------------------------------
    # مراقبة (تشخيص فقط — لا تُعرَض على المريض)
    # ------------------------------------------------------------------
    @property
    def active_sessions(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._sessions)

    # ------------------------------------------------------------------
    # داخلي (يُستدعى دائماً والقفل مأخوذ)
    # ------------------------------------------------------------------
    def _get_live(self, session_id: str) -> Optional[ConversationState]:
        """الجلسة إن كانت موجودة وغير منتهية، وإلّا None (مع حذف المنتهية)."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return None

        state, last_seen = entry
        if self._is_expired(last_seen):
            # حذف فوري: البيانات الصحّية لا تبقى بعد انتهاء صلاحيتها.
            del self._sessions[session_id]
            return None

        # لمسة وصول: الصلاحية من آخر نشاط لا من الإنشاء.
        self._sessions[session_id] = (state, time.monotonic())
        self._sessions.move_to_end(session_id)
        return state

    def _put(self, session_id: str, state: ConversationState) -> None:
        self._sessions[session_id] = (state, time.monotonic())
        self._sessions.move_to_end(session_id)

    def _is_expired(self, last_seen: float) -> bool:
        # ttl <= 0 يعني تعطيل الانتهاء (مفيد للاختبارات لا للإنتاج).
        return self._ttl > 0 and (time.monotonic() - last_seen) > self._ttl

    def _purge_expired(self) -> int:
        expired = [
            sid for sid, (_, seen) in self._sessions.items()
            if self._is_expired(seen)
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def _enforce_capacity(self) -> None:
        """تنظيف المنتهية أولاً، ثم إخلاء الأقدم استخداماً عند اللزوم."""
        if self._max <= 0 or len(self._sessions) <= self._max:
            return

        self._purge_expired()
        # الأقدم استخداماً يخرج أولاً (LRU) — الجلسة النشطة لا تُخلى.
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
