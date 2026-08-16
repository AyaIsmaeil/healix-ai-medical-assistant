import pytest

from vocabulary.duration import (
    DURATION_FIXED_WORDS,
    DURATION_UNIT_DAYS,
    parse_duration_days,
)


# --- the reason this module exists: Arabic dual forms ------------------------


@pytest.mark.parametrize(
    ("text", "days"),
    [("يومين", 2), ("أسبوعين", 14), ("شهرين", 60)],
)
def test_parses_arabic_dual_forms(text, days):
    # These carry no digit, so a "number + unit" regex misses them entirely.
    assert parse_duration_days(text) == days


def test_dual_form_is_not_read_as_the_singular_unit():
    # "يومين" contains "يوم"; a unit-first match would return 1, not 2.
    assert parse_duration_days("عندي وجع من يومين") == 2


def test_dual_form_matches_inside_a_sentence():
    assert parse_duration_days("من أسبوعين وأنا تعبان") == 14


# --- counted durations --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "days"),
    [
        ("منذ 3 أيام", 3),
        ("من 5 ايام", 5),
        ("منذ 2 أسبوع", 14),
        ("من 3 أشهر", 90),
        ("منذ 1 سنة", 365),
        ("4 ايام", 4),
    ],
)
def test_parses_counted_durations(text, days):
    assert parse_duration_days(text) == days


def test_parses_arabic_indic_digits():
    # Relies on normalize() folding ٠-٩ before the regex runs.
    assert parse_duration_days("منذ ٣ أيام") == 3


# --- word boundaries: rules.red_flags._compile_keyword_matcher reused ----------


def test_counted_duration_does_not_match_a_longer_word_sharing_its_prefix():
    # "اسبوعية" (weekly — an adjective about FREQUENCY, not duration) starts
    # with the unit word "اسبوع". Verified directly (not assumed) that the
    # unbounded regex this replaces would have matched "3 اسبوع" as a bare
    # prefix and returned 21 (3 * 7) — a duration the patient never stated.
    assert parse_duration_days("عندي حالة اسبوعية بتصير من زمان") is None


def test_fixed_word_does_not_match_inside_an_unrelated_longer_word():
    # "يوم" (day, the FIXED dual form's own root) is a bare substring of
    # "يومياً" (daily — again frequency, not duration): "بيجيني الصداع 3
    # مرات يومياً" describes how OFTEN attacks recur, not that they started
    # 3 days ago. Naive substring containment on DURATION_FIXED_WORDS would
    # not fire here (يومين itself is absent), but the same collision class
    # applies once "يوم" alone is checked the same unbounded way — this
    # pins the boundary-aware matcher's behavior on the exact adjacent-
    # letter risk case now that it is regex-based rather than `in`-based.
    assert parse_duration_days("بيجيني الصداع 3 مرات يومياً") is None


def test_prefixed_dual_form_is_still_detected():
    # The dangerous direction, same reasoning as
    # test_arabic_prefixed_forms_of_chronic_keyword_lower_threshold in
    # rules/red_flags.py's own test suite: a strict \b boundary that
    # rejected an attached ال would silently miss a real duration.
    assert parse_duration_days("من الاسبوعين الماضيين صار عندي") == 14


# --- spelling tolerance --------------------------------------------------------


def test_tolerates_alef_and_taa_spelling_variants():
    # "اسبوعين" (bare alef) and "أسبوعين" (hamza) must behave identically.
    assert parse_duration_days("اسبوعين") == parse_duration_days("أسبوعين")


def test_lexicon_is_stored_normalized():
    from rules.crisis import normalize

    for word in list(DURATION_FIXED_WORDS) + list(DURATION_UNIT_DAYS):
        assert word == normalize(word), f"{word!r} is not stored normalized"


# --- absence is information, not zero -----------------------------------------


def test_returns_none_when_no_duration_stated():
    assert parse_duration_days("عندي صداع") is None


def test_returns_none_for_empty_text():
    assert parse_duration_days("") is None


def test_returns_none_rather_than_zero():
    # None means "not stated". Zero would be an invented clinical fact.
    result = parse_duration_days("ما في شي")
    assert result is None
    assert result != 0
