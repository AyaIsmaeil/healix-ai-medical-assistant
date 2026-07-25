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


def test_turn_prompt_keeps_known_symptoms_split():
    """ما تراكم سلفاً يصل للـLLM مفصولاً (مُثبَت/منفيّ) كتأريض ضد النسيان."""
    prompt = InterviewPromptBuilder().turn_prompt(_state())
    assert _grab(prompt, "KNOWN_SYMPTOMS") == ["حرارة"]
    assert _grab(prompt, "KNOWN_NEGATED_SYMPTOMS") == ["كحة"]


def test_turn_prompt_carries_accumulated_record():
    """السجل المتراكم يُمرَّر للـLLM فلا يُعيد السؤال عمّا استُخرج سابقاً."""
    state = _state()
    state.record.chief_complaint = "حرارة"
    state.record.medications = ["بنادول"]

    record = _grab(InterviewPromptBuilder().turn_prompt(state), "KNOWN_RECORD")
    assert record["chief_complaint"] == "حرارة"
    assert record["medications"] == ["بنادول"]
    # لا حقل تشخيص إطلاقاً في ما يُمرَّر أو يُطلَب.
    assert "diagnosis" not in record


def test_turn_prompt_marks_latest_message():
    prompt = InterviewPromptBuilder().turn_prompt(_state())
    assert _grab(prompt, "LATEST_MESSAGE") == "لا أعرف بالضبط"


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
    # ويطلب الاستخراج المنظَّم صراحةً (لا سؤالاً فقط).
    for key in ("symptoms", "medications", "allergies", "chronic_conditions",
                "family_history", "missing_fields"):
        assert key in sp


def test_json_schema_forbids_any_diagnosis_field():
    """منع بنيوي لا نصّي: المخطّط الصارم لا يسمح أصلاً بحقل تشخيص."""
    from app.prompts.interview_builder import INTERVIEW_JSON_SCHEMA

    schema = INTERVIEW_JSON_SCHEMA["schema"]
    assert schema["additionalProperties"] is False
    properties = set(schema["properties"])
    assert properties.isdisjoint(
        {"diagnosis", "disease", "prediction", "specialty", "urgency", "triage"}
    )
    # كل حقول السجل والقرار مطلوبة (شكل ثابت لا يعتمد على مزاج النموذج).
    assert set(schema["required"]) == properties
