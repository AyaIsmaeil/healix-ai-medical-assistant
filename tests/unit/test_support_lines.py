import pytest

import support_lines
from support_lines import (
    SupportLine,
    _validate_no_placeholder_markers_on_real_entries,
    verified_support_lines,
)


def test_shipped_support_lines_are_all_placeholders():
    # The real, current state of this file: no verified numbers have been
    # supplied yet. This is what crisis_node.py must degrade gracefully
    # against until someone replaces the entries in SUPPORT_LINES.
    assert verified_support_lines() == ()


def test_verified_support_lines_returns_only_non_placeholder_entries(monkeypatch):
    monkeypatch.setattr(
        support_lines,
        "SUPPORT_LINES",
        (
            SupportLine(name="PLACEHOLDER_ORG", phone="PLACEHOLDER_NUMBER", is_placeholder=True),
            SupportLine(name="خط الدعم الوطني", phone="123-456", is_placeholder=False),
        ),
    )

    result = verified_support_lines()

    assert result == (SupportLine(name="خط الدعم الوطني", phone="123-456", is_placeholder=False),)


def test_verified_support_lines_empty_when_nothing_is_verified(monkeypatch):
    monkeypatch.setattr(
        support_lines,
        "SUPPORT_LINES",
        (SupportLine(name="PLACEHOLDER_ORG", phone="PLACEHOLDER_NUMBER", is_placeholder=True),),
    )

    assert verified_support_lines() == ()


# --- fail loud: a claimed-real entry that still reads as a placeholder --------


def test_rejects_an_entry_marked_real_whose_phone_still_says_placeholder():
    bad = (SupportLine(name="خط الدعم", phone="PLACEHOLDER_NUMBER", is_placeholder=False),)
    with pytest.raises(ValueError, match="PLACEHOLDER"):
        _validate_no_placeholder_markers_on_real_entries(bad)


def test_rejects_an_entry_marked_real_whose_name_still_says_placeholder():
    bad = (SupportLine(name="PLACEHOLDER_ORG", phone="123-456", is_placeholder=False),)
    with pytest.raises(ValueError, match="PLACEHOLDER"):
        _validate_no_placeholder_markers_on_real_entries(bad)


def test_accepts_a_genuinely_real_entry():
    good = (SupportLine(name="خط الدعم الوطني", phone="123-456", is_placeholder=False),)
    _validate_no_placeholder_markers_on_real_entries(good)  # must not raise


def test_accepts_a_placeholder_entry_that_says_placeholder():
    placeholder = (
        SupportLine(name="PLACEHOLDER_ORG", phone="PLACEHOLDER_NUMBER", is_placeholder=True),
    )
    _validate_no_placeholder_markers_on_real_entries(placeholder)  # must not raise
