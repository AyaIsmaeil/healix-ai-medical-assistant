"""اختبارات وحدة للمستخرج القاعدي (RuleBasedFeatureExtractor)."""

from app.domain.feature_extraction_rules import RuleBasedFeatureExtractor


def _extractor():
    return RuleBasedFeatureExtractor()


def test_extracts_age_from_explicit_phrase():
    result = _extractor().extract(["عمري 34 سنة وعندي صداع"])
    assert result.age == 34


def test_extracts_bare_age_answer():
    result = _extractor().extract(["23"])
    assert result.age == 23


def test_ignores_smoking_duration_years_false_positive():
    result = _extractor().extract(["أنا مدخن منذ 6 سنوات وعندي ضيق نفس"])
    assert result.age is None


def test_explicit_age_wins_over_duration_years():
    result = _extractor().extract(["مدخن منذ 6 سنوات", "عمري 23 سنة"])
    assert result.age == 23


def test_extracts_age_with_arabic_indic_digits():
    result = _extractor().extract(["عمري ٤٠ عام"])
    assert result.age == 40


def test_extracts_male_gender():
    result = _extractor().extract(["أنا رجل وعندي حرارة"])
    assert result.gender == "male"


def test_extracts_female_gender():
    result = _extractor().extract(["أنا امرأة وعندي صداع"])
    assert result.gender == "female"


def test_gender_none_when_no_hint():
    result = _extractor().extract(["عندي صداع منذ يومين"])
    assert result.gender is None


def test_extracts_smoking_positive():
    result = _extractor().extract(["أنا مدخن منذ سنوات"])
    assert result.smoking is True


def test_extracts_smoking_negative():
    result = _extractor().extract(["أنا ما بدخن"])
    assert result.smoking is False


def test_extracts_temperature():
    result = _extractor().extract(["حرارتي وصلت 39 درجة"])
    assert result.temperature == 39.0


def test_extracts_duration_dual_word():
    result = _extractor().extract(["عندي صداع منذ يومين"])
    assert result.duration_text == "يومين"
    assert result.duration_days == 2


def test_extracts_duration_explicit_count():
    result = _extractor().extract(["عندي حرارة منذ 3 أيام"])
    assert result.duration_days == 3


def test_extracts_severity_numeric():
    result = _extractor().extract(["شدة الألم 8 من 10"])
    assert result.severity == 8


def test_extracts_severity_qualitative_word():
    result = _extractor().extract(["الألم شديد جداً"])
    assert result.severity == 9


def test_unresolved_fields_lists_missing_only():
    extractor = _extractor()
    result = extractor.extract(["عمري 30 سنة"])
    unresolved = extractor.unresolved_fields(result)
    assert "age" not in unresolved
    assert set(unresolved) == {"gender", "smoking", "temperature", "duration", "severity"}


def test_unresolved_fields_empty_when_all_resolved():
    extractor = _extractor()
    result = extractor.extract([
        "عمري 30 سنة، أنا رجل، ما بدخن، حرارتي 39 درجة، منذ يومين، شدتها 8 من 10"
    ])
    assert extractor.unresolved_fields(result) == []


def test_no_values_extracted_from_unrelated_text():
    result = _extractor().extract(["مرحبا كيفك"])
    assert result.age is None
    assert result.gender is None
    assert result.smoking is None
    assert result.temperature is None
    assert result.duration_text is None
    assert result.severity is None
