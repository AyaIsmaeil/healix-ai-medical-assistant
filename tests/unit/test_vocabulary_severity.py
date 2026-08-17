import pytest

from vocabulary.severity import (
    SEVERITY_WORDS,
    severity_from_score,
    severity_from_text,
)


# --- the documented 0-10 collapse ---------------------------------------------


@pytest.mark.parametrize("score", [0, 1, 2, 3])
def test_scores_0_to_3_collapse_to_mild(score):
    assert severity_from_score(score) == "mild"


@pytest.mark.parametrize("score", [4, 5, 6])
def test_scores_4_to_6_collapse_to_moderate(score):
    assert severity_from_score(score) == "moderate"


@pytest.mark.parametrize("score", [7, 8, 9, 10])
def test_scores_7_to_10_collapse_to_severe(score):
    assert severity_from_score(score) == "severe"


@pytest.mark.parametrize("score", [-1, 11, 100])
def test_out_of_range_score_raises_rather_than_guessing(score):
    with pytest.raises(ValueError):
        severity_from_score(score)


# --- inherited words land where the documented collapse says ------------------


@pytest.mark.parametrize(
    ("word", "level"),
    [
        ("خفيف", "mild"),      # old score 2
        ("بسيط", "mild"),      # old score 2
        ("متوسط", "moderate"), # old score 5
        ("شديد", "severe"),    # old score 8
        ("شديد جدا", "severe"),# old score 9
        ("لا يطاق", "severe"), # old score 10
    ],
)
def test_inherited_words_map_per_the_documented_collapse(word, level):
    assert severity_from_text(word) == level


@pytest.mark.parametrize(
    ("word", "old_score"),
    [("خفيف", 2), ("بسيط", 2), ("متوسط", 5), ("شديد", 8), ("شديد جدا", 9), ("لا يطاق", 10)],
)
def test_word_mapping_agrees_with_collapsing_its_old_numeric_score(word, old_score):
    # Guards the documented table: the word's level must equal what the
    # numeric collapse would produce from its original 0-10 score.
    assert severity_from_text(word) == severity_from_score(old_score)


def test_longer_phrase_is_not_shadowed_by_its_substring():
    # "شديد جدا" contains "شديد"; matching must not stop at the shorter word.
    assert severity_from_text("الوجع شديد جدا") == "severe"


# --- text extraction -----------------------------------------------------------


def test_numeric_rating_in_text():
    assert severity_from_text("الوجع 8 من 10") == "severe"


def test_numeric_rating_with_slash():
    assert severity_from_text("الوجع 2/10") == "mild"


def test_arabic_indic_digits_in_rating():
    # Relies on normalize() folding ٠-٩ first.
    assert severity_from_text("الوجع ٩ من 10") == "severe"


def test_explicit_number_wins_over_a_severity_word():
    # A patient who gives both has been more specific with the number.
    assert severity_from_text("وجع خفيف بس 9 من 10") == "severe"


def test_word_found_inside_a_sentence():
    assert severity_from_text("عندي وجع متوسط بالبطن") == "moderate"


def test_tolerates_spelling_variants():
    assert severity_from_text("شديد جداً") == severity_from_text("شديد جدا")


# --- word boundaries: rules.red_flags._compile_keyword_matcher reused ----------


def test_unrelated_word_containing_a_severity_word_is_not_detected():
    # "تخفيف" (relief/reduction, e.g. asking for PAIN RELIEF) contains
    # "خفيف" (mild) as a bare substring — same class of collision
    # rules/red_flags.py's own test suite documents for "سكر" inside
    # "سكرتير" (test_rules_red_flags.py). Verified directly (not assumed)
    # that the naive substring check this replaces would have matched:
    # normalize("خفيف") in normalize("محتاج تخفيف الألم بسرعة") is True.
    # A patient asking for relief has not graded severity as mild.
    assert severity_from_text("محتاج تخفيف الألم بسرعة") is None


def test_prefixed_forms_of_a_severity_word_are_still_detected():
    # The dangerous direction, same reasoning as
    # test_arabic_prefixed_forms_of_chronic_keyword_lower_threshold in
    # rules/red_flags.py's own test suite: a strict \b boundary that
    # rejected an attached ال would silently miss a real severity grading.
    assert severity_from_text("الألم الشديد ما بيروح") == "severe"


# --- absence is information ----------------------------------------------------


def test_returns_none_when_severity_not_stated():
    assert severity_from_text("عندي صداع من يومين") is None


def test_returns_none_for_empty_text():
    assert severity_from_text("") is None


# --- lexicon hygiene ------------------------------------------------------------


def test_lexicon_is_stored_normalized():
    from rules.crisis import normalize

    for word in SEVERITY_WORDS:
        assert word == normalize(word), f"{word!r} is not stored normalized"


def test_every_word_maps_to_a_valid_level():
    assert set(SEVERITY_WORDS.values()) <= {"mild", "moderate", "severe"}
