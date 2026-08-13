"""اختبارات InterviewTriage — إنهاء مبكر للمقابلة."""

from app.domain.clinical_record import ClinicalRecord
from app.domain.conversation import (
    AskedQuestion,
    ConversationState,
    Symptom,
)
from app.domain.interview_triage import should_short_circuit_interview


def _chest_dyspnea_state(**overrides) -> ConversationState:
    state = ConversationState(session_id="t1")
    state.symptoms = [
        Symptom(text="ضغط على الصدر", negated=False, confidence=0.9),
        Symptom(text="ضيق تنفس", negated=False, confidence=0.9),
    ]
    state.record = ClinicalRecord(age=23, gender="female", severity_numeric=8)
    state.asked_questions = [
        AskedQuestion(slot=f"slot{i}", question="q", turn=i)
        for i in range(8)
    ]
    state.asked_questions.extend([
        AskedQuestion(slot="severity@ضيق نفس", question="q", turn=8),
        AskedQuestion(slot="dyspnea@ضغط", question="q", turn=9),
        AskedQuestion(slot="exertion@ضغط", question="q", turn=10),
        AskedQuestion(slot="context:age", question="q", turn=11),
        AskedQuestion(slot="context:gender", question="q", turn=12),
    ])
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_short_circuit_when_chest_pathway_complete():
    assert should_short_circuit_interview(_chest_dyspnea_state()) is True


def test_no_short_circuit_without_demographics():
    state = _chest_dyspnea_state()
    state.record.age = None
    assert should_short_circuit_interview(state) is False


def test_no_short_circuit_without_dyspnea():
    state = _chest_dyspnea_state()
    state.symptoms = [Symptom(text="ضغط على الصدر", negated=False, confidence=0.9)]
    assert should_short_circuit_interview(state) is False
