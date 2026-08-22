"""Tests for rag/health_education/preprocess.py's per-row cleaning logic.

All against clean_record() directly — no real AHD.xlsx read here (that
file is ~172MB, gitignored, and not something a unit test should depend
on existing). data/AHD_DATA_INSPECTION.md documents the real full-file
counts this logic was designed against.
"""

from __future__ import annotations

from rag.health_education.preprocess import clean_record, mask_pii


def test_a_normal_row_is_kept():
    result = clean_record("شو أسباب الصداع النصفي؟", "الصداع النصفي له عدة أسباب معروفة.", "أمراض عصبية")
    assert result.kept
    assert not result.quarantined_medication
    assert result.dropped_reason is None


def test_empty_question_is_dropped_as_malformed():
    result = clean_record("", "جواب موجود", "فئة")
    assert not result.kept
    assert result.dropped_reason == "malformed"


def test_empty_answer_is_dropped_as_malformed():
    result = clean_record("سؤال موجود", "", "فئة")
    assert not result.kept
    assert result.dropped_reason == "malformed"


def test_missing_category_is_dropped_as_malformed():
    result = clean_record("سؤال موجود", "جواب موجود", None)
    assert not result.kept
    assert result.dropped_reason == "malformed"


def test_junk_test_row_is_dropped():
    result = clean_record("test calll from lina 33", "أسئلة وأجوبة طبية - أمراض باطنية | الطبي", "أمراض باطنية")
    assert not result.kept
    assert result.dropped_reason == "junk"


def test_too_short_question_is_dropped_as_junk():
    result = clean_record("ok", "هاد جواب طويل بما فيه الكفاية.", "فئة")
    assert not result.kept
    assert result.dropped_reason == "junk"


def test_question_with_no_arabic_is_excluded():
    result = clean_record("WBC 7.5 LYM 2.4 HGB 12.0", "هاد ضمن الطبيعي.", "الطب العام")
    assert not result.kept
    assert result.dropped_reason == "non_arabic"


def test_medication_content_is_quarantined_not_dropped_and_not_kept():
    result = clean_record(
        "شو الجرعة المناسبة؟", "الجرعة اليومية هي قرص واحد مجم كل ٨ ساعات.", "الطب العام"
    )
    assert not result.kept
    assert result.quarantined_medication
    assert result.dropped_reason is None


def test_self_disclosed_name_is_masked_before_being_kept():
    result = clean_record("مرحبا اسمي محمد عمري 21 عندي دوخة", "راجع طبيب أعصاب.", "أمراض عصبية")
    assert result.kept
    assert result.pii_masked
    assert "محمد" not in result.question
    assert "[محجوب]" in result.question


def test_mask_pii_leaves_text_without_a_name_disclosure_unchanged():
    text = "عندي صداع من يومين"
    masked, was_masked = mask_pii(text)
    assert masked == text
    assert not was_masked


def test_very_long_answer_is_truncated_but_still_kept():
    long_answer = "جواب طويل جدا. " * 300  # well over 2000 chars
    result = clean_record("سؤال عادي عن موضوع صحي", long_answer, "فئة")
    assert result.kept
    assert len(result.answer) <= 2001  # _MAX_STORED_ANSWER_LEN + ellipsis
