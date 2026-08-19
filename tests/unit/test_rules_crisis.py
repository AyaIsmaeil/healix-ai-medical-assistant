import re

import pytest

from rules.crisis import (
    CRISIS_PATTERNS,
    CrisisPattern,
    _validate_phrases_are_normalized,
    detect_crisis,
    normalize,
)

# All test strings below include ordinary, non-crisis Arabic unless noted.
# Crisis phrase tests use real Syrian colloquial already present in this
# project's fixtures (e.g. "بدي موت" in test_graph.py).


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


def test_detect_crisis_matches_suicidal_phrase_used_in_project_fixtures():
    result = detect_crisis("بدي موت")

    assert result.matched is True
    assert result.categories == ("suicidal_ideation",)


def test_detect_crisis_matches_hopelessness_phrase():
    result = detect_crisis("ما في فايده")

    assert result.matched is True
    assert result.categories == ("hopelessness_severe",)


def test_detect_crisis_reports_every_category_that_matched():
    # A message can plausibly hit more than one category; the result must
    # not silently drop down to whichever one happened to match first.
    result = detect_crisis("بدي موت، ما في فايده")

    assert result.matched is True
    assert result.categories == ("suicidal_ideation", "hopelessness_severe")


def test_detect_crisis_deduplicates_repeated_categories(monkeypatch):
    import rules.crisis as crisis_module

    repeated_category = (
        CrisisPattern("shared_category", re.compile("TEST_MARKER_ONE")),
        CrisisPattern("shared_category", re.compile("TEST_MARKER_TWO")),
    )
    monkeypatch.setattr(crisis_module, "CRISIS_PATTERNS", repeated_category)

    result = detect_crisis("TEST_MARKER_ONE TEST_MARKER_TWO")

    assert result.categories == ("shared_category",)


def test_validate_phrases_are_normalized_rejects_unnormalized_phrase():
    # "أنا بخير" ("I'm fine") — ordinary, benign text, chosen only because
    # it contains أ, which normalize() would fold to ا. This is not real
    # crisis content; it just needs to be a phrase normalize() changes.
    with pytest.raises(ValueError, match="انا بخير"):
        _validate_phrases_are_normalized(frozenset({"أنا بخير"}))


def test_validate_phrases_are_normalized_accepts_already_normalized_phrase():
    _validate_phrases_are_normalized(frozenset({"انا بخير"}))  # must not raise


def test_crisis_phrases_pass_normalization_guard():
    # Guards against a future edit reintroducing an un-normalized phrase
    # into _CRISIS_PHRASES without anyone noticing until it silently fails
    # to match in production.
    from rules import crisis as crisis_module

    for phrases in crisis_module._CRISIS_PHRASES.values():
        _validate_phrases_are_normalized(phrases)
