import pytest

from nodes._shared import (
    GENERIC_REFERRAL,
    latest_user_message,
    previous_assistant_message,
    support_line_text,
)
from support_lines import SupportLine


def _state(messages):
    return {"messages": messages}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


def test_latest_user_message_returns_the_only_user_message():
    state = _state([_user("مرحبا")])
    assert latest_user_message(state) == "مرحبا"


def test_latest_user_message_skips_a_trailing_assistant_message():
    state = _state([_user("الرسالة القديمة"), _assistant("سؤال متابعة")])
    assert latest_user_message(state) == "الرسالة القديمة"


def test_latest_user_message_picks_the_most_recent_user_message():
    state = _state([_user("أ"), _assistant("ب"), _user("ج")])
    assert latest_user_message(state) == "ج"


def test_latest_user_message_raises_when_no_user_message_exists():
    state = _state([_assistant("مرحبا كيف أساعدك؟")])
    with pytest.raises(ValueError, match="no user-role message"):
        latest_user_message(state)


# --- previous_assistant_message -------------------------------------------------


def test_previous_assistant_message_returns_none_on_a_first_turn():
    state = _state([_user("عندي صداع")])
    assert previous_assistant_message(state) is None


def test_previous_assistant_message_returns_the_immediately_preceding_one():
    state = _state([_user("عندي صداع"), _assistant("منذ متى بدأ؟"), _user("من الصبح")])
    assert previous_assistant_message(state) == "منذ متى بدأ؟"


def test_previous_assistant_message_does_not_reach_further_back_than_one_turn():
    # Two rounds of follow-up: only the SECOND assistant question is what
    # the latest reply is answering, not the first.
    state = _state(
        [
            _user("عندي صداع"),
            _assistant("سؤال أول"),
            _user("رد أول"),
            _assistant("سؤال ثاني"),
            _user("رد ثاني"),
        ]
    )
    assert previous_assistant_message(state) == "سؤال ثاني"


def test_previous_assistant_message_returns_none_if_the_prior_entry_is_not_assistant():
    # Defensive: two user messages back to back is not expected in this
    # project's design, but must not return something misleading — no
    # question was actually asked immediately before this reply.
    state = _state([_user("أ"), _user("ب")])
    assert previous_assistant_message(state) is None


# --- support_line_text -----------------------------------------------------------


def test_support_line_text_returns_verified_numbers_when_configured(monkeypatch):
    monkeypatch.setattr(
        "nodes._shared.verified_support_lines",
        lambda: (SupportLine(name="خط الدعم الوطني", phone="123-456", is_placeholder=False),),
    )

    text = support_line_text()

    assert "خط الدعم الوطني" in text
    assert "123-456" in text


def test_support_line_text_falls_back_to_the_generic_referral_when_none_configured(monkeypatch):
    monkeypatch.setattr("nodes._shared.verified_support_lines", lambda: ())

    assert support_line_text() == GENERIC_REFERRAL


def test_support_line_text_never_invents_a_number_when_none_are_configured(monkeypatch):
    monkeypatch.setattr("nodes._shared.verified_support_lines", lambda: ())

    assert not any(char.isdigit() for char in support_line_text())
