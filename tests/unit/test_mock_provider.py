"""اختبارات وحدة للمزوّد الوهمي (MockLLMProvider) — تساؤل ديناميكي + عقد موحّد."""

import json
from collections import namedtuple

from app.domain import clinical
from app.llm.mock_provider import MockLLMProvider
from app.parsing.interview_parser import parse_interview_decision, parse_interview_turn
from app.parsing.symptom_extraction_parser import parse_symptoms
from app.prompts.interview_builder import (
    LABEL_ASKED,
    LABEL_KNOWN_NEGATED,
    LABEL_KNOWN_SYMPTOMS,
    LABEL_LATEST_MESSAGE,
    LABEL_PATIENT_MESSAGES,
    LABEL_SUGGESTED,
    LABEL_TURN,
)
from app.prompts.symptom_extraction_builder import SymptomExtractionPromptBuilder


def _prompt(symptoms=None, asked=None, patient_messages=None):
    """يبني كتلة الحالة الموسومة كما ينتجها InterviewPromptBuilder.

    تُستورَد أسماء الوسوم من مصدرها بدل نسخها نصّاً، فلا ينحرف الاختبار عن
    المنتِج عند أي إعادة تسمية لاحقة.
    """
    symptoms = symptoms or []
    known = [s["text"] for s in symptoms if not s.get("negated")]
    negated = [s["text"] for s in symptoms if s.get("negated")]
    messages = patient_messages or ["رسالة"]

    def line(label, value):
        return f"{label}: {json.dumps(value, ensure_ascii=False)}"

    return "\n".join([
        line(LABEL_PATIENT_MESSAGES, messages),
        line(LABEL_LATEST_MESSAGE, messages[-1]),
        line(LABEL_KNOWN_SYMPTOMS, known),
        line(LABEL_KNOWN_NEGATED, negated),
        line(LABEL_ASKED, asked or []),
        line(LABEL_TURN, 1),
        line(LABEL_SUGGESTED, []),
    ])


def test_asks_core_slot_first_for_symptom():
    provider = MockLLMProvider()
    out = provider.generate("sys", _prompt(symptoms=[{"text": "صداع", "negated": False}]))
    decision = parse_interview_decision(out.text)
    assert decision.finished is False
    assert decision.next_slot == "onset@صداع"


def test_dynamic_fever_specific_question_reached():
    provider = MockLLMProvider()
    # كل خانات OLDCARTS للحرارة طُرحت → يجب أن يصل لسؤال خاص بالحرارة.
    asked = [
        "onset@حرارة", "duration@حرارة", "severity@حرارة", "progression@حرارة",
        "location@حرارة", "quality@حرارة", "associated_symptoms@حرارة",
        "aggravating@حرارة", "relieving@حرارة",
    ]
    out = provider.generate("sys", _prompt(
        symptoms=[{"text": "حرارة", "negated": False}], asked=asked
    ))
    decision = parse_interview_decision(out.text)
    assert decision.next_slot == "temperature@حرارة"


def test_skips_asked_slot():
    provider = MockLLMProvider()
    out = provider.generate("sys", _prompt(
        symptoms=[{"text": "صداع", "negated": False}],
        asked=["onset@صداع"],
    ))
    decision = parse_interview_decision(out.text)
    assert decision.next_slot == "duration@صداع"


def test_finishes_when_all_asked():
    provider = MockLLMProvider()
    Sym = namedtuple("Sym", ["text", "negated"])
    targets = [t for t, _ in clinical.build_checklist([Sym("صداع", False)])]
    out = provider.generate("sys", _prompt(
        symptoms=[{"text": "صداع", "negated": False}], asked=targets
    ))
    decision = parse_interview_decision(out.text)
    assert decision.finished is True


def test_emits_the_unified_contract():
    """المزوّد الوهمي يُنتج عقد دور المقابلة (سجل + قرار السؤال التالي)، بلا
    أعراض — الأعراض تُستخرَج بمسار SymptomExtractor المستقلّ."""
    provider = MockLLMProvider()
    out = provider.generate("sys", _prompt(
        symptoms=[{"text": "صداع", "negated": False}]
    ))
    turn = parse_interview_turn(out.text)

    assert turn.symptoms == []
    assert turn.record.chief_complaint == "صداع"
    assert turn.decision.finished is False
    # لا يخترع قيماً طبية لم تُذكر.
    assert turn.record.severity is None
    assert turn.record.medications == []


def _symptom_extraction_builder() -> SymptomExtractionPromptBuilder:
    return SymptomExtractionPromptBuilder(mappings=[
        {"concept": "fever", "names": ["حرارة"]},
        {"concept": "headache", "names": ["صداع"]},
        {"concept": "cough", "names": ["سعال"]},
    ])


def test_detects_symptoms_and_negation_from_patient_text():
    """مسار استخراج الأعراض المستقلّ (CONCEPT_CATALOG) يستخرج من نص المريض مباشرةً."""
    builder = _symptom_extraction_builder()
    provider = MockLLMProvider()
    prompt = builder.extraction_prompt(["عندي حرارة وصداع ولا يوجد سعال"])

    out = provider.generate(builder.system_prompt(), prompt)
    symptoms = parse_symptoms(out.text, builder.concepts)

    found = {(s.text, s.negated) for s in symptoms}
    assert ("حرارة", False) in found
    assert ("صداع", False) in found
    assert ("سعال", True) in found
