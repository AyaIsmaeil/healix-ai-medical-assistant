import re

import pytest

from rules.crisis import (
    CRISIS_PATTERNS,
    CrisisPattern,
    _validate_patterns_are_normalized,
    detect_crisis,
    normalize,
)

# All test strings below are ordinary, non-crisis Arabic — greetings,
# weather, symptom mentions, and everyday chatter. None of this exercises
# CRISIS_PATTERNS content (it's placeholder data); these tests only cover
# the normalization pipeline and the matching plumbing around it.


def test_normalize_strips_diacritics():
    assert normalize("السَّلَامُ عَلَيْكُمْ") == "السلام عليكم"


def test_normalize_unifies_alef_variants():
    assert normalize("إذا أحسست بألم آخر") == "اذا احسست بالم اخر"


def test_normalize_maps_taa_marbuta_to_haa():
    assert normalize("الصحة مهمة") == "الصحه مهمه"


def test_normalize_removes_tatweel():
    assert normalize("أ" + "ـ" * 3 + "نا بخير") == "انا بخير"


def test_normalize_collapses_chat_elongation():
    assert normalize("تمامممم") == "تمام"


def test_normalize_collapses_irregular_whitespace():
    assert normalize("الجو  حار    اليوم") == "الجو حار اليوم"


def test_normalize_strips_leading_and_trailing_whitespace():
    assert normalize("   بخير الحمد لله   ") == "بخير الحمد لله"


def test_normalize_makes_spelling_variants_equivalent():
    # ة vs ه — same word, two common spellings, must collapse identically.
    assert normalize("المدرسة") == normalize("المدرسه")


def test_normalize_folds_arabic_indic_digits():
    # An Arabic keyboard emits ٠-٩ by default; without folding, "٣٩ درجة"
    # carries a temperature nothing downstream can read.
    assert normalize("٣٩ درجة") == "39 درجه"


def test_normalize_folds_extended_arabic_indic_digits():
    # Persian/Urdu keyboards emit U+06F0-06F9, visually near-identical.
    assert normalize("۳۹ درجة") == "39 درجه"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("٠١٢٣٤٥٦٧٨٩", "0123456789"), ("۰۱۲۳۴۵۶۷۸۹", "0123456789")],
)
def test_normalize_folds_every_digit_in_both_ranges(raw, expected):
    assert normalize(raw) == expected


def test_normalize_does_not_collapse_repeated_digits():
    # Elongation collapse must skip digits — otherwise folding them to ASCII
    # is pointless because "999" would silently become "9".
    assert normalize("999") == "999"
    assert normalize("١١١") == "111"


def test_normalize_still_collapses_repeated_letters():
    # The digit exemption must not weaken letter elongation collapse.
    assert normalize("تمامممم") == "تمام"


def test_normalize_is_idempotent():
    once = normalize("أنـــا بخيــر الحمد لله")
    assert normalize(once) == once


def test_detect_crisis_no_match_on_ordinary_symptom_text():
    result = detect_crisis("عندي صداع وحرارة من يومين")

    assert result.matched is False
    assert result.categories == ()


def test_detect_crisis_matches_and_reports_category():
    # CRISIS_PATTERNS is placeholder data (see rules/crisis.py) — this
    # exercises the matching plumbing using the placeholder marker itself,
    # not real crisis content, since none exists yet.
    placeholder = CRISIS_PATTERNS[0]

    result = detect_crisis(placeholder.pattern.pattern)

    assert result.matched is True
    assert result.categories == (placeholder.category,)


def test_detect_crisis_reports_every_category_that_matched():
    # A message can plausibly hit more than one pattern; the result must
    # not silently drop down to whichever one happened to match first.
    first, second = CRISIS_PATTERNS[0], CRISIS_PATTERNS[1]
    message = f"{first.pattern.pattern} {second.pattern.pattern}"

    result = detect_crisis(message)

    assert result.matched is True
    assert result.categories == (first.category, second.category)


def test_detect_crisis_deduplicates_repeated_categories(monkeypatch):
    import rules.crisis as crisis_module

    repeated_category = (
        CrisisPattern("shared_category", re.compile("TEST_MARKER_ONE")),
        CrisisPattern("shared_category", re.compile("TEST_MARKER_TWO")),
    )
    monkeypatch.setattr(crisis_module, "CRISIS_PATTERNS", repeated_category)

    result = detect_crisis("TEST_MARKER_ONE TEST_MARKER_TWO")

    assert result.categories == ("shared_category",)


def test_validate_patterns_are_normalized_rejects_unnormalized_pattern():
    # "أنا بخير" ("I'm fine") — ordinary, benign text, chosen only because
    # it contains أ, which normalize() would fold to ا. This is not real
    # crisis content; it just needs to be a pattern normalize() changes.
    unnormalized = (CrisisPattern("example_category", re.compile("أنا بخير")),)

    with pytest.raises(ValueError, match="example_category"):
        _validate_patterns_are_normalized(unnormalized)


def test_validate_patterns_are_normalized_accepts_already_normalized_pattern():
    already_normalized = (CrisisPattern("example_category", re.compile("انا بخير")),)

    _validate_patterns_are_normalized(already_normalized)  # must not raise


def test_crisis_patterns_constant_passes_its_own_normalization_guard():
    # Guards against a future edit reintroducing an un-normalized pattern
    # into CRISIS_PATTERNS without anyone noticing until it silently fails
    # to match in production. (Also proven by the fact that importing
    # rules.crisis at all didn't raise — this makes it an explicit test.)
    _validate_patterns_are_normalized(CRISIS_PATTERNS)
