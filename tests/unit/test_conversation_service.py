"""
اختبارات وحدة لمحرك المحادثة (ConversationService).

تُستخدم منافذ وهمية (مستخرج أعراض + مزوّد LLM) فلا حاجة لنموذج حقيقي.
"""

import json
from dataclasses import dataclass
from typing import List

from app.domain.ports import Completion
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService


@dataclass
class FakeExtracted:
    text: str
    negated: bool
    confidence: float


class FakeExtractor:
    """يُرجع أعراضاً محدّدة مسبقاً لكل نصّ (حسب الاستدعاء)."""

    def __init__(self, scripts: List[List[FakeExtracted]]):
        self._scripts = scripts
        self._i = 0

    def extract(self, text: str) -> List[FakeExtracted]:
        result = self._scripts[self._i] if self._i < len(self._scripts) else []
        self._i += 1
        return result


class ScriptedProvider:
    """يُرجع ردوداً JSON محدّدة مسبقاً بالترتيب."""

    name = "scripted"

    def __init__(self, responses: List[dict]):
        self._responses = responses
        self._i = 0

    def generate(self, system_prompt: str, user_prompt: str) -> Completion:
        resp = self._responses[self._i]
        self._i += 1
        return Completion(json.dumps(resp, ensure_ascii=False), model="scripted")


def _service(extractor, provider, max_questions=8):
    return ConversationService(
        extractor=extractor,
        provider=provider,
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=max_questions,
    )


def test_first_turn_extracts_stores_and_asks():
    extractor = FakeExtractor([[FakeExtracted("صداع", False, 0.9),
                               FakeExtracted("حرارة", False, 0.8)]])
    provider = ScriptedProvider([
        {"finished": False, "next_slot": "onset", "question": "منذ متى؟"},
    ])
    svc = _service(extractor, provider)

    state, decision = svc.handle_message("أعاني من صداع وحرارة", session_id=None)

    assert decision.finished is False
    assert decision.question == "منذ متى؟"
    assert decision.next_slot == "onset"
    assert state.turn_count == 1
    assert {s.text for s in state.symptoms} == {"صداع", "حرارة"}
    assert state.asked_slots == ["onset"]
    assert state.pending_slot == "onset"


def test_second_turn_keeps_all_raw_messages_and_clears_pending():
    extractor = FakeExtractor([
        [FakeExtracted("صداع", False, 0.9)],
        [],  # no new symptoms on the answer
    ])
    provider = ScriptedProvider([
        {"finished": False, "next_slot": "onset", "question": "منذ متى؟"},
        {"finished": False, "next_slot": "severity", "question": "ما الشدة؟"},
    ])
    svc = _service(extractor, provider)

    state, _ = svc.handle_message("عندي صداع", session_id=None)
    sid = state.session_id

    state, decision = svc.handle_message("منذ يومين", session_id=sid)

    # كل رسائل المريض محفوظة كاملةً (لا فقدان معلومات) ولا خريطة خانة→قيمة.
    assert state.raw_messages == ["عندي صداع", "منذ يومين"]
    assert not hasattr(state, "answered_slots")
    assert decision.next_slot == "severity"
    assert state.asked_slots == ["onset", "severity"]
    assert state.pending_slot == "severity"
    assert state.turn_count == 2


def test_finished_marks_state_completed():
    extractor = FakeExtractor([[FakeExtracted("صداع", False, 0.9)]])
    provider = ScriptedProvider([{"finished": True}])
    svc = _service(extractor, provider)

    state, decision = svc.handle_message("عندي صداع", session_id=None)

    assert decision.finished is True
    assert state.status.value == "completed"
    assert state.pending_slot is None


def test_symptoms_are_deduplicated_across_turns():
    extractor = FakeExtractor([
        [FakeExtracted("صداع", False, 0.9)],
        [FakeExtracted("صداع", False, 0.9)],  # repeated
    ])
    provider = ScriptedProvider([
        {"finished": False, "next_slot": "onset", "question": "منذ متى؟"},
        {"finished": False, "next_slot": "severity", "question": "ما الشدة؟"},
    ])
    svc = _service(extractor, provider)

    state, _ = svc.handle_message("صداع", session_id=None)
    state, _ = svc.handle_message("صداع", session_id=state.session_id)

    assert len([s for s in state.symptoms if s.text == "صداع"]) == 1


def test_max_questions_cap_forces_finish():
    extractor = FakeExtractor([[], []])
    provider = ScriptedProvider([
        {"finished": False, "next_slot": "onset", "question": "س1؟"},
        # second turn should not consult the provider because cap is reached
    ])
    svc = _service(extractor, provider, max_questions=1)

    state, _ = svc.handle_message("مرحبا", session_id=None)  # asks 1 question
    state, decision = svc.handle_message("جواب", session_id=state.session_id)

    assert decision.finished is True
    assert state.status.value == "completed"


class OneShotExtractor:
    """يُرجع أعراضاً في أول دور فقط، ولا شيء بعدها."""

    def __init__(self, first):
        self._first = first
        self._done = False

    def extract(self, text):
        if self._done:
            return []
        self._done = True
        return self._first


def test_dynamic_interview_with_mock_provider_asks_fever_specifics_and_finishes():
    from app.llm.mock_provider import MockLLMProvider

    extractor = OneShotExtractor([FakeExtracted("حرارة", False, 0.9)])
    svc = ConversationService(
        extractor=extractor,
        provider=MockLLMProvider(),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=40,
    )

    sid = None
    asked = []
    questions = []
    finished = False
    for _ in range(40):
        answer = "منذ يومين" if sid is None else "لا أعرف"
        state, decision = svc.handle_message(answer, sid)
        sid = state.session_id
        if decision.finished:
            finished = True
            break
        asked.append(decision.next_slot)
        questions.append(decision.question)

    assert finished is True
    # لا تكرار لأي خانة.
    assert len(asked) == len(set(asked))
    # جُمعت OLDCARTS للحرارة.
    assert "onset@حرارة" in asked and "severity@حرارة" in asked
    # وصلت الأسئلة الخاصة بالحرارة (تساؤل ديناميكي).
    assert "temperature@حرارة" in asked
    # وأسئلة سياقية عند اللزوم.
    assert any(a.startswith("context:") for a in asked)
    # كل الأسئلة عربية غير فارغة.
    assert all(isinstance(q, str) and q for q in questions)


def test_repeated_slot_from_llm_ends_interview():
    extractor = FakeExtractor([[], []])
    provider = ScriptedProvider([
        {"finished": False, "next_slot": "onset", "question": "منذ متى؟"},
        {"finished": False, "next_slot": "onset", "question": "منذ متى مجدداً؟"},
    ])
    svc = _service(extractor, provider)

    state, _ = svc.handle_message("مرحبا", session_id=None)
    state, decision = svc.handle_message("جواب", session_id=state.session_id)

    assert decision.finished is True
