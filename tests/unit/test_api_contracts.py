import pytest
from pydantic import ValidationError

from api.contracts import ChatRequest, ChatResponse, Message


# --- Message -----------------------------------------------------------------


def test_message_accepts_user_role():
    message = Message(role="user", content="مرحبا")
    assert message.role == "user"
    assert message.content == "مرحبا"


def test_message_accepts_assistant_role():
    message = Message(role="assistant", content="كيف أقدر أساعدك؟")
    assert message.role == "assistant"


def test_message_rejects_a_role_outside_user_or_assistant():
    with pytest.raises(ValidationError):
        Message(role="system", content="مرحبا")


# --- ChatRequest ---------------------------------------------------------------


def test_chat_request_accepts_a_full_first_turn():
    request = ChatRequest(
        thread_id="thread-1",
        message="عندي صداع من يومين",
        medical_record_summary="لا توجد أمراض مزمنة معروفة",
    )
    assert request.thread_id == "thread-1"
    assert request.message == "عندي صداع من يومين"
    assert request.medical_record_summary == "لا توجد أمراض مزمنة معروفة"


def test_chat_request_medical_record_summary_defaults_to_none():
    request = ChatRequest(thread_id="thread-1", message="عندي صداع")
    assert request.medical_record_summary is None


@pytest.mark.parametrize("sex", ["male", "female"])
def test_chat_request_accepts_a_valid_patient_sex(sex):
    request = ChatRequest(thread_id="thread-1", message="عندي صداع", patient_sex=sex)
    assert request.patient_sex == sex


def test_chat_request_patient_sex_defaults_to_none():
    request = ChatRequest(thread_id="thread-1", message="عندي صداع")
    assert request.patient_sex is None


def test_chat_request_rejects_an_invalid_patient_sex():
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="thread-1", message="عندي صداع", patient_sex="other")


def test_chat_request_rejects_empty_thread_id():
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="", message="عندي صداع")


def test_chat_request_thread_id_only_fills_conversation_id_to_the_same_value():
    request = ChatRequest(thread_id="thread-1", message="عندي صداع")
    assert request.thread_id == "thread-1"
    assert request.conversation_id == "thread-1"


def test_chat_request_conversation_id_only_fills_thread_id_to_the_same_value():
    request = ChatRequest(conversation_id="conv-1", message="عندي صداع")
    assert request.thread_id == "conv-1"
    assert request.conversation_id == "conv-1"


def test_chat_request_rejects_when_neither_id_is_sent():
    with pytest.raises(ValidationError):
        ChatRequest(message="عندي صداع")


def test_chat_request_rejects_mismatched_thread_id_and_conversation_id():
    with pytest.raises(ValidationError):
        ChatRequest(
            thread_id="thread-a",
            conversation_id="thread-b",
            message="عندي صداع",
        )


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="thread-1", message="")


def test_chat_request_rejects_a_missing_message():
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="thread-1")


# --- ChatResponse ---------------------------------------------------------------


def test_chat_response_accepts_a_minimal_followup_turn():
    response = ChatResponse(
        thread_id="thread-1",
        reply="من متى وأنت حاسس فيه؟",
        stage="followup",
        is_crisis=False,
    )
    assert response.thread_id == "thread-1"
    assert response.conversation_id == "thread-1"
    assert response.stage == "followup"
    assert response.severity is None
    assert response.red_flags == []
    assert response.diagnosis is None
    assert response.specialty is None
    assert response.reports is None


def test_chat_response_accepts_a_full_diagnosis_turn():
    response = ChatResponse(
        thread_id="thread-1",
        reply="هاد تقرير الحالة",
        stage="diagnosis",
        is_crisis=False,
        severity="moderate",
        red_flags=[],
        diagnosis={"possibilities": []},
        specialty="باطنية",
        reports={"patient": "...", "doctor": "..."},
    )
    assert response.severity == "moderate"
    assert response.specialty == "باطنية"
    assert response.reports == {"patient": "...", "doctor": "..."}


def test_chat_response_rejects_a_stage_outside_the_four_terminal_nodes():
    with pytest.raises(ValidationError):
        ChatResponse(
            thread_id="thread-1", reply="...", stage="diagnosing", is_crisis=False
        )


def test_chat_response_rejects_a_severity_outside_the_closed_set():
    with pytest.raises(ValidationError):
        ChatResponse(
            thread_id="thread-1",
            reply="...",
            stage="emergency",
            is_crisis=False,
            severity="critical",
        )


def test_chat_response_requires_is_crisis_and_stage():
    with pytest.raises(ValidationError):
        ChatResponse(thread_id="thread-1", reply="...")
