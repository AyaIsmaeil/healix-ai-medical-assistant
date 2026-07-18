"""اختبارات وحدة لبانِي تعليمات المقابلة (InterviewPromptBuilder)."""

import json
import re

from app.domain.conversation import ConversationState, Symptom
from app.prompts.interview_builder import InterviewPromptBuilder


def _state():
    state = ConversationState(session_id="s1")
    state.record_patient_message("أعاني من حرارة منذ يومين")
    state.record_question("severity@حرارة", "ما الشدة؟")  # سُئل عن الشدّة (لم يُجب بعد)
    state.record_patient_message("لا أعرف بالضبط")          # الرسالة الثانية
    state.add_symptoms([Symptom("حرارة", False, 0.9), Symptom("كحة", True, 0.8)])
    state.turn_count = 2
    return state


def _grab(prompt: str, label: str):
    """قراءة قيمة JSON لسطر موسوم من كتلة الحالة."""
    m = re.search(rf"^{label}:\s*(.+)$", prompt, re.MULTILINE)
    return json.loads(m.group(1).strip())


def test_turn_prompt_contains_all_raw_patient_messages():
    prompt = InterviewPromptBuilder().turn_prompt(_state())
    # كل رسائل المريض تصل للـ LLM (لا تُفقد تفاصيل الرسالة الأولى كـ"منذ يومين").
    assert _grab(prompt, "PATIENT_MESSAGES") == [
        "أعاني من حرارة منذ يومين",
        "لا أعرف بالضبط",
    ]


def test_turn_prompt_keeps_marbert_symptoms_split():
    prompt = InterviewPromptBuilder().turn_prompt(_state())
    assert _grab(prompt, "EXTRACTED_SYMPTOMS") == ["حرارة"]
    assert _grab(prompt, "NEGATED_SYMPTOMS") == ["كحة"]


def test_turn_prompt_has_no_slot_value_map():
    prompt = InterviewPromptBuilder().turn_prompt(_state())
    # لا خريطة "خانة → قيمة" قد تكون خاطئة؛ فقط الخانات المطروحة لمنع التكرار.
    assert "ANSWERED_SLOTS" not in prompt
    assert _grab(prompt, "ASKED_SLOTS") == ["severity@حرارة"]
    assert _grab(prompt, "TURN_COUNT") == 2


def test_turn_prompt_suggests_missing_high_value_slots():
    prompt = InterviewPromptBuilder().turn_prompt(_state())
    missing = _grab(prompt, "SUGGESTED_MISSING_SLOTS")
    assert isinstance(missing, list) and missing
    # ما سُئل عنه يجب ألّا يظهر ضمن المقترح.
    assert "severity@حرارة" not in missing


def test_system_prompt_enforces_constraints_and_raw_context():
    sp = InterviewPromptBuilder().system_prompt()
    assert "JSON" in sp
    assert "finished=true" in sp
    assert "تشخيص" in sp
    # يشرح استخدام كامل رسائل المريض + عدم تكرار ما ذُكر فيها.
    assert "PATIENT_MESSAGES" in sp
