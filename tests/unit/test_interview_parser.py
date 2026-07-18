"""اختبارات وحدة لمحلّل مخرجات المقابلة (parse_interview_decision)."""

import pytest

from app.exceptions import InterviewParsingError
from app.parsing.interview_parser import parse_interview_decision


def test_parses_question_decision():
    out = parse_interview_decision(
        '{"finished": false, "next_slot": "onset", "question": "منذ متى؟"}'
    )
    assert out.finished is False
    assert out.next_slot == "onset"
    assert out.question == "منذ متى؟"


def test_parses_finished_decision():
    out = parse_interview_decision('{"finished": true}')
    assert out.finished is True
    assert out.next_slot is None
    assert out.question is None


def test_strips_code_fences():
    out = parse_interview_decision(
        '```json\n{"finished": false, "next_slot": "severity", "question": "ما الشدة؟"}\n```'
    )
    assert out.next_slot == "severity"


def test_ignores_surrounding_text():
    out = parse_interview_decision(
        'إليك الرد: {"finished": true} انتهى'
    )
    assert out.finished is True


def test_rejects_empty():
    with pytest.raises(InterviewParsingError):
        parse_interview_decision("")


def test_rejects_invalid_json():
    with pytest.raises(InterviewParsingError):
        parse_interview_decision("{not json")


def test_rejects_missing_finished():
    with pytest.raises(InterviewParsingError):
        parse_interview_decision('{"next_slot": "onset", "question": "؟"}')


def test_rejects_question_without_slot():
    with pytest.raises(InterviewParsingError):
        parse_interview_decision('{"finished": false, "question": "؟"}')


def test_rejects_question_without_text():
    with pytest.raises(InterviewParsingError):
        parse_interview_decision('{"finished": false, "next_slot": "onset"}')
