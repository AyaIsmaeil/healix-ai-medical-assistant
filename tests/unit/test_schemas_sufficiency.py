import schemas.sufficiency as sufficiency_schema
from schemas.sufficiency import FALLBACK_QUESTION, SufficiencyAssessment

# --- valid parse -------------------------------------------------------------


def test_parses_a_sufficient_verdict_with_no_question():
    result = SufficiencyAssessment.model_validate({"is_sufficient": True})

    assert result.is_sufficient is True
    assert result.next_question is None


def test_parses_an_insufficient_verdict_with_a_question():
    result = SufficiencyAssessment.model_validate(
        {"is_sufficient": False, "next_question": "من متى بلش الوجع بالضبط؟"}
    )

    assert result.is_sufficient is False
    assert result.next_question == "من متى بلش الوجع بالضبط؟"


# --- coupling repair: sufficient + a question -----------------------------------


def test_a_stray_question_on_a_sufficient_verdict_is_dropped():
    result = SufficiencyAssessment.model_validate(
        {"is_sufficient": True, "next_question": "سؤال ما كان لازم يكون هون"}
    )

    assert result.next_question is None


def test_dropping_a_stray_question_is_audited(monkeypatch):
    logged = []
    monkeypatch.setattr(
        sufficiency_schema, "log_malformed_output", lambda **kw: logged.append(kw)
    )

    SufficiencyAssessment.model_validate(
        {"is_sufficient": True, "next_question": "سؤال زائد"}
    )

    assert len(logged) == 1
    assert logged[0]["reason"] == "next_question_present_while_sufficient"


# --- coupling repair: insufficient + a missing question -------------------------


def test_a_missing_question_on_an_insufficient_verdict_falls_back():
    result = SufficiencyAssessment.model_validate({"is_sufficient": False})

    assert result.next_question == FALLBACK_QUESTION


def test_a_blank_question_on_an_insufficient_verdict_falls_back():
    result = SufficiencyAssessment.model_validate(
        {"is_sufficient": False, "next_question": "   "}
    )

    assert result.next_question == FALLBACK_QUESTION


def test_falling_back_to_the_generic_question_is_audited(monkeypatch):
    logged = []
    monkeypatch.setattr(
        sufficiency_schema, "log_malformed_output", lambda **kw: logged.append(kw)
    )

    SufficiencyAssessment.model_validate({"is_sufficient": False})

    assert len(logged) == 1
    assert logged[0]["reason"] == "missing_next_question_while_insufficient"
