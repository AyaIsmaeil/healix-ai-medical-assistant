"""
اختبارات مخزن الجلسات: انتهاء الصلاحية، حدّ السعة، وإشارة إعادة البدء (W2).

الهدف المعلن: الجلسات تحمل بيانات صحّية شخصية، فبقاؤها بلا انتهاء تسرّب
موارد ومشكلة خصوصية معاً.
"""

import json

import pytest

from app.domain.ports import Completion
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService


# ----------------------------------------------------------------------
# انتهاء الصلاحية
# ----------------------------------------------------------------------
def test_session_survives_within_ttl():
    store = InMemorySessionStore(ttl_seconds=3600)
    state = store.get_or_create(None)
    state.record_patient_message("عندي صداع")

    assert store.get(state.session_id) is not None
    assert store.get(state.session_id).raw_messages == ["عندي صداع"]


def test_expired_session_is_removed_and_not_returned(monkeypatch):
    store = InMemorySessionStore(ttl_seconds=60)
    state = store.get_or_create(None)
    state.record_patient_message("بيانات صحّية")
    sid = state.session_id

    # تقديم الساعة إلى ما بعد الصلاحية.
    import app.infrastructure.session_store as module
    base = module.time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 61)

    assert store.get(sid) is None
    # وحُذفت فعلاً من الذاكرة، لا مجرّد إخفاء عن القارئ.
    assert store.active_sessions == 0


def test_ttl_counts_from_last_activity_not_creation(monkeypatch):
    """مقابلة طويلة نشطة لا تنتهي في منتصفها."""
    store = InMemorySessionStore(ttl_seconds=100)
    state = store.get_or_create(None)
    sid = state.session_id

    import app.infrastructure.session_store as module
    base = module.time.monotonic()

    # نشاط عند t+80 يُجدّد الصلاحية.
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 80)
    assert store.get(sid) is not None

    # t+150 = بعد 70 ثانية من آخر نشاط ⇒ ما زالت حيّة.
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 150)
    assert store.get(sid) is not None


def test_ttl_zero_disables_expiry(monkeypatch):
    store = InMemorySessionStore(ttl_seconds=0)
    sid = store.get_or_create(None).session_id

    import app.infrastructure.session_store as module
    base = module.time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 10_000)

    assert store.get(sid) is not None


# ----------------------------------------------------------------------
# حدّ السعة (منع تسرّب الذاكرة)
# ----------------------------------------------------------------------
def test_store_never_grows_beyond_max():
    """الإثبات المباشر لغياب تسرّب الذاكرة."""
    store = InMemorySessionStore(ttl_seconds=3600, max_sessions=10)

    for _ in range(200):
        store.get_or_create(None)

    assert store.active_sessions <= 10


def test_eviction_removes_least_recently_used_first():
    store = InMemorySessionStore(ttl_seconds=3600, max_sessions=3)
    first = store.get_or_create(None).session_id
    second = store.get_or_create(None).session_id
    third = store.get_or_create(None).session_id

    # لمس الأولى يجعلها الأحدث استخداماً، فتُخلى الثانية بدلها.
    store.get(first)
    store.get_or_create(None)

    assert store.get(first) is not None
    assert store.get(second) is None
    assert store.get(third) is not None


def test_expired_sessions_are_purged_before_evicting_live_ones(monkeypatch):
    store = InMemorySessionStore(ttl_seconds=50, max_sessions=3)
    old = [store.get_or_create(None).session_id for _ in range(3)]

    import app.infrastructure.session_store as module
    base = module.time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 51)

    fresh = store.get_or_create(None)

    assert store.get(fresh.session_id) is not None
    assert all(store.get(sid) is None for sid in old)


# ----------------------------------------------------------------------
# إشارة إعادة البدء — لا فقدان صامت للسجل
# ----------------------------------------------------------------------
class _Provider:
    name = "scripted"

    def generate(self, system_prompt, user_prompt):
        return Completion(json.dumps({
            "chief_complaint": None, "symptoms": [], "severity": None,
            "duration": None, "body_location": None, "medications": [],
            "allergies": [], "chronic_conditions": [], "family_history": [],
            "missing_fields": [], "finished": False,
            "next_slot": "onset", "question": "منذ متى؟",
        }, ensure_ascii=False), model="scripted")


def _service(store):
    return ConversationService(
        provider=_Provider(),
        prompt_builder=InterviewPromptBuilder(),
        store=store,
    )


def test_unknown_session_id_is_flagged_not_silently_restarted():
    """متابعة بسجلّ فارغ دون إعلام تُفقد كل ما جُمع بصمت."""
    svc = _service(InMemorySessionStore())
    state, _ = svc.handle_message("عندي صداع", "معرّف-لا-وجود-له")

    assert state.session_restarted is True


def test_first_message_without_session_id_is_not_a_restart():
    svc = _service(InMemorySessionStore())
    state, _ = svc.handle_message("عندي صداع", None)

    assert state.session_restarted is False


def test_resumed_live_session_is_not_flagged():
    svc = _service(InMemorySessionStore())
    state, _ = svc.handle_message("عندي صداع", None)
    state, _ = svc.handle_message("من يومين", state.session_id)

    assert state.session_restarted is False
    assert state.raw_messages == ["عندي صداع", "من يومين"]


def test_expired_session_restart_is_flagged_to_the_client(monkeypatch):
    store = InMemorySessionStore(ttl_seconds=60)
    svc = _service(store)
    state, _ = svc.handle_message("عندي صداع", None)
    sid = state.session_id

    import app.infrastructure.session_store as module
    base = module.time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 61)

    state, _ = svc.handle_message("من يومين", sid)

    assert state.session_restarted is True
    assert state.raw_messages == ["من يومين"]   # السجل القديم انتهى فعلاً


# ----------------------------------------------------------------------
# ذرّية الدور (W5) — لا تثبيت جزئي عند الفشل
# ----------------------------------------------------------------------
class _FailingProvider:
    """يفشل في أول استدعاء ثم ينجح — يحاكي مهلة عابرة وإعادة محاولة."""

    name = "flaky"

    def __init__(self, failures=1):
        self.calls = 0
        self._failures = failures

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError("transient timeout")
        return _Provider().generate(system_prompt, user_prompt)


def test_failed_turn_commits_nothing():
    store = InMemorySessionStore()
    svc = ConversationService(
        provider=_FailingProvider(), prompt_builder=InterviewPromptBuilder(),
        store=store,
    )
    with pytest.raises(Exception):
        svc.handle_message("عندي صداع", None)

    # الجلسة أُنشئت لكن لا رسالة ولا عدّاد مثبَّت.
    sessions = [store.get(sid) for sid in list(store._sessions)]
    assert all(s.raw_messages == [] and s.turn_count == 0 for s in sessions)


def test_retry_after_failure_records_message_exactly_once():
    """العيب المُثبَت سابقاً: الرسالة كانت تُسجَّل مرّتين والعدّاد ينتفخ."""
    store = InMemorySessionStore()
    svc = ConversationService(
        provider=_FailingProvider(), prompt_builder=InterviewPromptBuilder(),
        store=store,
    )
    with pytest.raises(Exception):
        svc.handle_message("عندي صداع", None)

    sid = next(iter(store._sessions))
    state, _ = svc.handle_message("عندي صداع", sid)      # إعادة المحاولة

    assert state.raw_messages == ["عندي صداع"]
    assert state.turn_count == 1


def test_retry_of_first_turn_is_not_reported_as_restart():
    """الجلسة أُنشئت لكن لم يُثبَّت لها دور — ليست جلسة ضائعة."""
    store = InMemorySessionStore()
    svc = ConversationService(
        provider=_FailingProvider(), prompt_builder=InterviewPromptBuilder(),
        store=store,
    )
    with pytest.raises(Exception):
        svc.handle_message("عندي صداع", None)

    sid = next(iter(store._sessions))
    state, _ = svc.handle_message("عندي صداع", sid)

    assert state.session_restarted is False


def test_successful_turn_is_committed():
    store = InMemorySessionStore()
    svc = ConversationService(
        provider=_Provider(), prompt_builder=InterviewPromptBuilder(), store=store,
    )
    state, _ = svc.handle_message("عندي صداع", None)

    persisted = store.get(state.session_id)
    assert persisted.raw_messages == ["عندي صداع"]
    assert persisted.turn_count == 1


def test_working_copy_does_not_leak_into_store_before_commit():
    """نسخة العمل معزولة: تحوّرها لا يُرى في المخزن قبل التثبيت."""
    store = InMemorySessionStore()
    svc = ConversationService(
        provider=_Provider(), prompt_builder=InterviewPromptBuilder(), store=store,
    )
    state, _ = svc.handle_message("رسالة أولى", None)
    sid = state.session_id

    # تعديل النسخة المُعادة لا يجوز أن يمسّ ما في المخزن.
    state.raw_messages.append("تلوّث")
    assert store.get(sid).raw_messages == ["رسالة أولى"]
