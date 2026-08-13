"""
اختبارات وحدة لوكيل المقابلة السريرية (ConversationService).

الأعراض تُستخرَج بمسار مستقلّ (SymptomExtractor) عن سجل الدور وقراره — لذلك
تُحقَن هنا عبر ScriptedSymptomExtractor موازٍ لـ ScriptedProvider، بدل حقل
داخل ردّ الـLLM كما كان قبل الانتقال لهذه المعمارية.
"""

import json
from typing import List, Optional

from app.domain.conversation import Symptom
from app.domain.ports import Completion
from app.infrastructure.session_store import InMemorySessionStore
from app.prompts.interview_builder import InterviewPromptBuilder
from app.services.conversation_service import ConversationService


def turn(
    *,
    finished: bool = False,
    next_slot: Optional[str] = None,
    question: Optional[str] = None,
    **record,
) -> dict:
    """يبني ردّ LLM مطابقاً لعقد دور المقابلة (سجل + قرار، بلا أعراض)."""
    payload = {
        "chief_complaint": None,
        "severity": None,
        "duration": None,
        "body_location": None,
        "medications": [],
        "allergies": [],
        "chronic_conditions": [],
        "family_history": [],
        "missing_fields": [],
        "finished": finished,
        "next_slot": next_slot,
        "question": question,
    }
    payload.update(record)
    return payload


def sym(text: str, negated: bool = False, confidence: float = 0.9) -> dict:
    return {"text": text, "negated": negated, "confidence": confidence}


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


class ScriptedSymptomExtractor:
    """بديل اختباري لـ CompositeSymptomExtractor — أعراض محدّدة مسبقاً بالترتيب.

    كل استدعاء لـ ``extract`` يستهلك الدفعة التالية من الردود؛ تجاوز عدد
    الردود المُعطاة يُرجع قائمة فارغة بدل رفع خطأ (أدوار لا تهتمّ باستخراج
    جديد لا تحتاج تزويده صراحةً).
    """

    def __init__(self, responses: Optional[List[List[dict]]] = None):
        self._responses = responses or []
        self._i = 0

    def extract(self, raw_messages, known_symptoms=None) -> List[Symptom]:
        if self._i >= len(self._responses):
            return []
        batch = self._responses[self._i]
        self._i += 1
        return [
            Symptom(
                text=s["text"],
                negated=s.get("negated", False),
                confidence=s.get("confidence", 0.9),
            )
            for s in batch
        ]


def _service(provider, max_questions=8, symptom_responses=None):
    return ConversationService(
        provider=provider,
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=max_questions,
        symptom_extractor=ScriptedSymptomExtractor(symptom_responses),
    )


def test_first_turn_extracts_stores_and_asks():
    provider = ScriptedProvider([
        turn(next_slot="onset", question="منذ متى؟"),
    ])
    svc = _service(provider, symptom_responses=[
        [sym("صداع"), sym("حرارة", confidence=0.8)],
    ])

    state, decision = svc.handle_message("أعاني من صداع وحرارة", session_id=None)

    assert decision.finished is False
    assert decision.question == "منذ متى؟"
    assert decision.next_slot == "onset"
    assert state.turn_count == 1
    assert {s.text for s in state.symptoms} == {"صداع", "حرارة"}
    assert state.asked_slots == ["onset"]
    assert state.pending_slot == "onset"


def test_second_turn_keeps_all_raw_messages_and_clears_pending():
    provider = ScriptedProvider([
        turn(next_slot="onset", question="منذ متى؟"),
        turn(next_slot="severity", question="ما الشدة؟"),
    ])
    svc = _service(provider, symptom_responses=[
        [sym("صداع")],
        [],  # لا أعراض جديدة في الدور الثاني
    ])

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
    provider = ScriptedProvider([turn(finished=True)])
    svc = _service(provider, symptom_responses=[[sym("صداع")]])

    state, decision = svc.handle_message("عندي صداع", session_id=None)

    assert decision.finished is True
    assert state.status.value == "completed"
    assert state.pending_slot is None


def test_symptoms_are_deduplicated_across_turns():
    provider = ScriptedProvider([
        turn(next_slot="onset", question="منذ متى؟"),
        turn(next_slot="severity", question="ما الشدة؟"),
    ])
    svc = _service(provider, symptom_responses=[
        [sym("صداع")],
        [sym("صداع")],
    ])

    state, _ = svc.handle_message("صداع", session_id=None)
    state, _ = svc.handle_message("صداع", session_id=state.session_id)

    assert len([s for s in state.symptoms if s.text == "صداع"]) == 1


def test_structured_record_accumulates_across_turns():
    """السجل المنظَّم يتراكم ولا يُستبدل بمخرجات دور واحد."""
    provider = ScriptedProvider([
        turn(next_slot="onset", question="منذ متى؟",
             chief_complaint="صداع", medications=["بنادول"]),
        turn(next_slot="severity", question="ما الشدة؟",
             duration="ثلاثة أيام", allergies=["بنسلين"], missing_fields=["age"]),
    ])
    svc = _service(provider, symptom_responses=[
        [sym("صداع")],
        [sym("صداع")],
    ])

    state, _ = svc.handle_message("عندي صداع وآخذ بنادول", session_id=None)
    state, _ = svc.handle_message("من ثلاثة أيام وعندي حساسية بنسلين",
                                  session_id=state.session_id)

    record = state.record
    assert record.chief_complaint == "صداع"      # لم تُمحَ بدور لم يُعدها
    assert record.duration == "ثلاثة أيام"
    assert record.medications == ["بنادول"]
    assert record.allergies == ["بنسلين"]
    assert record.missing_fields == ["age"]      # لقطة لحظية (تُستبدل)


def test_max_questions_cap_forces_finish():
    """السقف يفرض الإنهاء مهما اقترح النموذج — لكن الاستخراج يبقى محفوظاً.

    المزوّد يُستشار في الدور الأخير أيضاً (بخلاف التصميم القديم) حتى لا تضيع
    المعلومات الطبية في رسالة المريض الأخيرة؛ القرار وحده هو ما يُتجاوز.
    """
    provider = ScriptedProvider([
        turn(next_slot="onset", question="س1؟"),
        turn(next_slot="severity", question="س2؟", medications=["بنادول"]),
    ])
    svc = _service(provider, max_questions=1, symptom_responses=[
        [],
        [sym("صداع")],
    ])

    state, _ = svc.handle_message("مرحبا", session_id=None)  # سؤال واحد
    state, decision = svc.handle_message("عندي صداع وآخذ بنادول",
                                         session_id=state.session_id)

    assert decision.finished is True
    assert state.status.value == "completed"
    # الاستخراج من الرسالة الأخيرة لم يُفقَد رغم فرض الإنهاء.
    assert {s.text for s in state.symptoms} == {"صداع"}
    assert state.record.medications == ["بنادول"]


def test_dynamic_interview_with_mock_provider_asks_fever_specifics_and_finishes():
    from app.llm.mock_provider import MockLLMProvider
    from app.prompts.symptom_extraction_builder import SymptomExtractionPromptBuilder
    from app.services.composite_symptom_extractor import CompositeSymptomExtractor
    from app.services.llm_symptom_extractor import LLMSymptomExtractor
    from app.services.rule_based_symptom_extractor import RuleBasedSymptomExtractor

    mappings = [{"concept": "fever", "names": ["حرارة", "حرار", "حمى"]}]
    symptom_extractor = CompositeSymptomExtractor(
        llm_extractor=LLMSymptomExtractor(
            provider=MockLLMProvider(),
            prompt_builder=SymptomExtractionPromptBuilder(mappings=mappings),
        ),
        rule_extractor=RuleBasedSymptomExtractor.from_dict({"mappings": mappings}),
    )

    svc = ConversationService(
        provider=MockLLMProvider(),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=40,
        symptom_extractor=symptom_extractor,
    )

    sid = None
    asked = []
    questions = []
    finished = False
    for _ in range(40):
        answer = "عندي حرارة منذ يومين" if sid is None else "لا أعرف"
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
    # يلزم وجود عرَض، وإلا فحارس "لا إنهاء بلا أعراض" له الأولوية (أدناه).
    provider = ScriptedProvider([
        turn(next_slot="onset", question="منذ متى؟"),
        turn(next_slot="onset", question="منذ متى مجدداً؟"),
    ])
    svc = _service(provider, symptom_responses=[[sym("صداع")], []])

    state, _ = svc.handle_message("عندي صداع", session_id=None)
    state, decision = svc.handle_message("جواب", session_id=state.session_id)

    assert decision.finished is True


def test_never_finishes_before_any_symptom_is_collected():
    """حارس المجال: الـ LLM قد يُنهي بعد "مرحبا/كيفك" — المحرك يمنع ذلك."""
    provider = ScriptedProvider([turn(finished=True), turn(finished=True)])
    svc = _service(provider, symptom_responses=[[], []])

    state, decision = svc.handle_message("مرحبا", session_id=None)
    assert decision.finished is False
    assert decision.next_slot == "chief_complaint"

    state, decision = svc.handle_message("كيفك", session_id=state.session_id)
    assert decision.finished is False
    assert decision.next_slot == "chief_complaint"


def test_negated_symptoms_are_stored_and_do_not_satisfy_the_guard():
    """عرَض منفيّ ليس عرَضاً مُثبَتاً: الحارس يبقى مُفعَّلاً."""
    provider = ScriptedProvider([turn(finished=True)])
    svc = _service(provider, symptom_responses=[[sym("سعال", negated=True)]])

    state, decision = svc.handle_message("ما في سعال", session_id=None)

    assert [(s.text, s.negated) for s in state.symptoms] == [("سعال", True)]
    assert decision.finished is False
    assert decision.next_slot == "chief_complaint"


def test_llm_question_is_not_overridden_before_any_symptom():
    """ما دام الـLLM يسأل، سؤاله هو المعتمد — حتى قبل استخراج أي عرَض.

    سابقاً كان المحرك يستبدله بسؤال ثابت يكاد يطابق تحية البداية، فيرى
    المريض السؤال نفسه مرّتين.
    """
    provider = ScriptedProvider([
        turn(next_slot="context:age", question="كم عمرك؟"),  # بلا أعراض
    ])
    svc = _service(provider, symptom_responses=[[]])

    state, decision = svc.handle_message("مرحبا دكتور", session_id=None)

    assert state.symptoms == []          # فعلاً لا عرَض بعد
    assert decision.finished is False
    assert decision.question == "كم عمرك؟"        # سؤال النموذج كما هو
    assert decision.next_slot == "context:age"


def test_guard_still_blocks_finishing_with_no_symptom():
    """الضمان الطبي باقٍ: محاولة الإنهاء بلا عرَض مُثبَت تُرفض."""
    provider = ScriptedProvider([turn(finished=True)])
    svc = _service(provider, symptom_responses=[[]])

    _, decision = svc.handle_message("مرحبا", session_id=None)

    assert decision.finished is False
    assert decision.next_slot == "chief_complaint"
