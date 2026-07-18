"""اختبارات وحدة للمزوّد الوهمي (MockLLMProvider) — تساؤل ديناميكي."""

import json
from collections import namedtuple

from app.domain import clinical
from app.llm.mock_provider import MockLLMProvider
from app.parsing.interview_parser import parse_interview_decision


def _prompt(symptoms=None, asked=None, patient_messages=None):
    """يبني كتلة الحالة الموسومة كما ينتجها InterviewPromptBuilder v4.

    التغطية للتكرار عبر ``asked`` فقط (لا خريطة خانة→قيمة).
    """
    symptoms = symptoms or []
    extracted = [s["text"] for s in symptoms if not s.get("negated")]
    negated = [s["text"] for s in symptoms if s.get("negated")]

    def line(label, value):
        return f"{label}: {json.dumps(value, ensure_ascii=False)}"

    return "\n".join([
        line("PATIENT_MESSAGES", patient_messages or ["رسالة"]),
        line("EXTRACTED_SYMPTOMS", extracted),
        line("NEGATED_SYMPTOMS", negated),
        line("ASKED_SLOTS", asked or []),
        line("TURN_COUNT", 1),
        line("SUGGESTED_MISSING_SLOTS", []),
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
