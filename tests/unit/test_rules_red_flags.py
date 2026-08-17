import pytest

from rules.crisis import normalize
from rules.red_flags import (
    RED_FLAG_RULES,
    RedFlagRule,
    SymptomRequirement,
    _validate_rule_symptoms_are_canonical,
    check_red_flags,
)


def sx(name: str) -> dict[str, str]:
    return {"name": name}


def rule_ids(matches) -> set[str]:
    return {match.rule_id for match in matches}


# --- single-symptom trigger ------------------------------------------------


def test_single_symptom_trigger_fires_alone():
    matches = check_red_flags([sx("فقدان الوعي")])

    assert "loss_of_consciousness" in rule_ids(matches)


def test_single_symptom_trigger_does_not_fire_without_the_symptom():
    matches = check_red_flags([sx("سعال"), sx("التهاب الحلق")])

    assert "loss_of_consciousness" not in rule_ids(matches)


# --- combination trigger ----------------------------------------------------


def test_combination_trigger_fires_only_when_full_combination_present():
    only_required = check_red_flags([sx("ألم في الصدر")])
    full_combination = check_red_flags([sx("ألم في الصدر"), sx("ضيق تنفس")])

    assert "acs_chest_pain" not in rule_ids(only_required)
    assert "acs_chest_pain" in rule_ids(full_combination)


def test_combination_trigger_does_not_fire_on_the_any_of_symptom_alone():
    matches = check_red_flags([sx("ضيق تنفس")])

    assert "acs_chest_pain" not in rule_ids(matches)


def test_combination_trigger_reports_the_symptoms_that_matched():
    matches = check_red_flags([sx("ألم في الصدر"), sx("ضيق تنفس"), sx("سعال")])

    acs = next(m for m in matches if m.rule_id == "acs_chest_pain")
    # matched_symptoms reports normalized names (see check_red_flags docstring).
    assert acs.matched_symptoms == frozenset(
        {normalize("ألم في الصدر"), normalize("ضيق تنفس")}
    )


def test_any_of_only_rule_fires_on_a_single_member_of_the_group():
    # stroke_fast has no all_of requirement — any one FAST sign alone fires it.
    matches = check_red_flags([sx("تدلي في الوجه")])

    assert "stroke_fast" in rule_ids(matches)


# --- no false fire on unrelated symptoms ------------------------------------


def test_no_false_fire_on_unrelated_symptoms():
    matches = check_red_flags([sx("سعال"), sx("التهاب الحلق"), sx("عطاس")])

    assert matches == []


# --- empty input -------------------------------------------------------------


def test_empty_symptom_list_returns_no_matches():
    assert check_red_flags([]) == []


def test_empty_symptom_list_with_record_summary_returns_no_matches():
    assert check_red_flags([], "مريض سكري منذ 10 سنوات") == []


def test_malformed_entries_without_name_are_ignored_not_raised():
    matches = check_red_flags([{"duration": "يومين"}, sx("فقدان الوعي")])

    assert "loss_of_consciousness" in rule_ids(matches)


# --- chronic-condition threshold lowering -----------------------------------


def test_chronic_condition_lowers_threshold_for_matching_rule():
    without_history = check_red_flags([sx("ألم في الصدر")])
    with_diabetes_history = check_red_flags(
        [sx("ألم في الصدر")], "مريض سكري منذ 10 سنوات"
    )

    assert "acs_chest_pain" not in rule_ids(without_history)

    acs = next(m for m in with_diabetes_history if m.rule_id == "acs_chest_pain")
    assert acs.lowered_by_chronic_condition is True


def test_unrelated_record_summary_does_not_lower_threshold():
    matches = check_red_flags([sx("ألم في الصدر")], "لا توجد أمراض مزمنة معروفة")

    assert "acs_chest_pain" not in rule_ids(matches)


# --- chronic keyword matching: normalization --------------------------------


def test_diacritized_chronic_keyword_still_lowers_threshold():
    # "سكّري" carries a shadda; raw substring matching missed it entirely.
    matches = check_red_flags([sx("ألم في الصدر")], "مريض سكّري منذ سنوات")

    assert "acs_chest_pain" in rule_ids(matches)


def test_alef_and_taa_variant_chronic_keyword_still_lowers_threshold():
    # Record written "زراعه اعضاء" instead of "زراعة أعضاء" — ة/ه and أ/ا.
    matches = check_red_flags([sx("حمى")], "حالة بعد زراعه اعضاء")

    sepsis = next(m for m in matches if m.rule_id == "sepsis")
    assert sepsis.lowered_by_chronic_condition is True


# --- chronic keyword matching: word boundaries ------------------------------


def test_unrelated_word_containing_keyword_does_not_lower_threshold():
    # "عسكرية" (military) contains "سكري" as a bare substring. Treating a
    # military-service note as a diabetes history would silently lower a
    # cardiac threshold for the wrong patient.
    matches = check_red_flags([sx("ألم في الصدر")], "خدمة عسكرية سابقة")

    assert "acs_chest_pain" not in rule_ids(matches)


@pytest.mark.parametrize("summary", ["يعمل سكرتير", "كان سكران البارحة"])
def test_unrelated_words_containing_short_keyword_do_not_lower_threshold(summary):
    # "سكر" is a substring of both; neither is a diabetes history.
    matches = check_red_flags([sx("ألم في الصدر")], summary)

    assert "acs_chest_pain" not in rule_ids(matches)


@pytest.mark.parametrize(
    "summary",
    ["مريض سكري", "مريض السكري", "مصاب بالسكري", "ارتفاع في السكر", "سكر الدم مرتفع"],
)
def test_arabic_prefixed_forms_of_chronic_keyword_lower_threshold(summary):
    # ال / بال / etc. attach directly to the word; a strict \b boundary would
    # reject these, which is the dangerous direction (a missed red flag).
    matches = check_red_flags([sx("ألم في الصدر")], summary)

    assert "acs_chest_pain" in rule_ids(matches)


def test_rule_without_chronic_keywords_never_reports_chronic_lowering():
    stroke = next(r for r in RED_FLAG_RULES if r.id == "stroke_fast")

    assert stroke.mentions_chronic_condition("مريض سكري") is False


# --- chronic keyword matching: English (Laravel/DrugCentral-sourced) --------
#
# chronic_condition_keywords_en is checked independently of the Arabic set
# above — a different mechanism (_compile_english_keyword_matcher:
# lowercase + substring, not normalize() + word-boundary regex), sourced
# from real Laravel data, not translated from the Arabic keywords.


def test_real_ddi_condition_string_lowers_acs_chest_pain_threshold():
    # "Diabetes mellitus type 2" is a real, verified value from Laravel's
    # own lang/en/enums.php ddi_condition picker — chronic_diseases stores
    # exactly this kind of string, not a translation of "سكري".
    without_history = check_red_flags([sx("ألم في الصدر")])
    with_diabetes_history = check_red_flags(
        [sx("ألم في الصدر")], "Diabetes mellitus type 2"
    )

    assert "acs_chest_pain" not in rule_ids(without_history)

    acs = next(m for m in with_diabetes_history if m.rule_id == "acs_chest_pain")
    assert acs.lowered_by_chronic_condition is True


def test_plausible_free_text_lowers_sepsis_threshold():
    # Realistic free-text content (not from the picker — Laravel's picker
    # has no oncology/chemotherapy option at all, see CLAUDE.md > Known
    # limitations), the kind that would reach medical_record_summary via
    # diagnosis/treatment_plan/current_medications prose.
    matches = check_red_flags(
        [sx("حمى")], "patient on chemotherapy for breast neoplasm"
    )

    sepsis = next(m for m in matches if m.rule_id == "sepsis")
    assert sepsis.lowered_by_chronic_condition is True


def test_english_matcher_is_substring_not_word_boundary():
    # "allograft rejection" is DrugCentral's own real indication phrasing
    # for transplant immunosuppressants — "graft" sits mid-word (inside
    # "allo-graft"), not at a word start. A word-boundary-anchored regex
    # (like the Arabic side's) would miss this; the English matcher is
    # deliberately plain substring containment specifically so it doesn't.
    matches = check_red_flags([sx("حمى")], "history of allograft rejection")

    assert "sepsis" in rule_ids(matches)


def test_normalize_does_not_alter_english_text_unexpectedly():
    # Confirms the module docstring's own claim: normalize() is Arabic
    # diacritic/hamza/digit-fold logic only, with no case-folding — so it
    # is not relied on for the English matcher, which does its own
    # .lower() instead. This pins that claim down as tested behavior, not
    # just an assertion in a comment.
    text = "Diabetes Mellitus Type 2"
    assert normalize(text) == text


def test_english_keywords_are_case_insensitive():
    matches = check_red_flags([sx("ألم في الصدر")], "DIABETES MELLITUS TYPE 2")

    assert "acs_chest_pain" in rule_ids(matches)


def test_english_and_arabic_chronic_keywords_both_reachable_on_the_same_rule():
    # Same rule (sepsis), two independent language paths — neither one
    # is a fallback for the other, both are checked every time.
    arabic = check_red_flags([sx("حمى")], "مريض مصاب بسرطان")
    english = check_red_flags([sx("حمى")], "patient has cancer")

    assert "sepsis" in rule_ids(arabic)
    assert "sepsis" in rule_ids(english)


def test_base_combination_match_is_not_flagged_as_chronic_lowered():
    matches = check_red_flags(
        [sx("ألم في الصدر"), sx("ضيق تنفس")], "مريض سكري منذ 10 سنوات"
    )

    acs = next(m for m in matches if m.rule_id == "acs_chest_pain")
    assert acs.lowered_by_chronic_condition is False


# --- canonical vocabulary guard ---------------------------------------------


def test_guard_fires_on_rule_referencing_unknown_symptom():
    unknown_symptom_rule = (
        RedFlagRule(
            id="example_rule",
            category="example",
            reason_ar="نص توضيحي للاختبار فقط",
            source="test fixture",
            requirement=SymptomRequirement(all_of=frozenset({"عرض غير موجود في المفردات"})),
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        _validate_rule_symptoms_are_canonical(unknown_symptom_rule)

    assert "example_rule" in str(excinfo.value)
    assert "عرض غير موجود في المفردات" in str(excinfo.value)


def test_guard_fires_on_unknown_symptom_in_any_of():
    rule = (
        RedFlagRule(
            id="example_rule",
            category="example",
            reason_ar="نص توضيحي للاختبار فقط",
            source="test fixture",
            requirement=SymptomRequirement(
                all_of=frozenset({"حمى"}),
                any_of=frozenset({"عرض غير موجود في المفردات"}),
            ),
        ),
    )

    with pytest.raises(ValueError, match="example_rule"):
        _validate_rule_symptoms_are_canonical(rule)


def test_guard_fires_on_unknown_symptom_in_chronic_requirement():
    rule = (
        RedFlagRule(
            id="example_rule",
            category="example",
            reason_ar="نص توضيحي للاختبار فقط",
            source="test fixture",
            requirement=SymptomRequirement(all_of=frozenset({"حمى"})),
            chronic_condition_keywords=frozenset({"سكري"}),
            chronic_requirement=SymptomRequirement(
                all_of=frozenset({"عرض غير موجود في المفردات"})
            ),
        ),
    )

    with pytest.raises(ValueError, match="example_rule"):
        _validate_rule_symptoms_are_canonical(rule)


def test_guard_accepts_rule_using_only_canonical_symptoms():
    rule = (
        RedFlagRule(
            id="example_rule",
            category="example",
            reason_ar="نص توضيحي للاختبار فقط",
            source="test fixture",
            requirement=SymptomRequirement(all_of=frozenset({"حمى"})),
        ),
    )

    _validate_rule_symptoms_are_canonical(rule)  # must not raise


def test_guard_accepts_natural_spelling_variants_of_canonical_names():
    # Vocabulary is stored normalized; a rule may be authored with أ / ة.
    rule = (
        RedFlagRule(
            id="example_rule",
            category="example",
            reason_ar="نص توضيحي للاختبار فقط",
            source="test fixture",
            requirement=SymptomRequirement(all_of=frozenset({"ألم في الصدر"})),
        ),
    )

    _validate_rule_symptoms_are_canonical(rule)  # must not raise


def test_real_rule_set_passes_its_own_vocabulary_guard():
    # Explicit regression test: importing rules.red_flags already runs this,
    # but that makes the failure a collection error rather than a named
    # failing test.
    _validate_rule_symptoms_are_canonical(RED_FLAG_RULES)


# --- spelling tolerance in matching -----------------------------------------


def test_rule_fires_despite_spelling_variation_from_extraction():
    # extract_symptoms emitting ا where the rule says أ must still match —
    # this is the silent-miss failure mode the vocabulary work targets.
    matches = check_red_flags([sx("الم في الصدر"), sx("ضيق تنفس")])

    assert "acs_chest_pain" in rule_ids(matches)


def test_rule_fires_despite_diacritics_from_extraction():
    matches = check_red_flags([sx("فقدان الوَعي")])

    assert "loss_of_consciousness" in rule_ids(matches)


# --- ectopic_pregnancy -------------------------------------------------------


def test_ectopic_pregnancy_fires_with_bleeding():
    matches = check_red_flags([sx("ألم بطن"), sx("نزيف مهبلي")])

    assert "ectopic_pregnancy" in rule_ids(matches)


def test_ectopic_pregnancy_fires_with_delayed_period():
    matches = check_red_flags([sx("ألم بطن"), sx("تأخر الدورة الشهرية")])

    assert "ectopic_pregnancy" in rule_ids(matches)


def test_ectopic_pregnancy_does_not_fire_on_abdominal_pain_alone():
    matches = check_red_flags([sx("ألم بطن")])

    assert "ectopic_pregnancy" not in rule_ids(matches)


def test_ectopic_pregnancy_does_not_fire_on_the_any_of_symptom_alone():
    bleeding_only = check_red_flags([sx("نزيف مهبلي")])
    delayed_period_only = check_red_flags([sx("تأخر الدورة الشهرية")])

    assert "ectopic_pregnancy" not in rule_ids(bleeding_only)
    assert "ectopic_pregnancy" not in rule_ids(delayed_period_only)


def test_ectopic_pregnancy_reports_the_symptoms_that_matched():
    matches = check_red_flags([sx("ألم بطن"), sx("نزيف مهبلي")])

    match = next(m for m in matches if m.rule_id == "ectopic_pregnancy")
    assert match.matched_symptoms == frozenset({normalize("ألم بطن"), normalize("نزيف مهبلي")})


def test_ectopic_pregnancys_documented_over_inclusion_is_real_not_just_a_comment():
    # CLAUDE.md / the rule's own comment: this rule has no way to gate on
    # reproductive-age/female patient status — state has no structured
    # demographic fields, only free-text medical_record_summary — and
    # deliberately does not invent one (rules/crisis.py's "bias toward
    # false positives" principle, applied here). This test proves that is
    # real behavior, not just a claim in a comment: the rule fires
    # identically with no demographic context at all, and even when the
    # record summary explicitly states a context (male patient) where
    # ectopic pregnancy could not apply.
    no_context = check_red_flags([sx("ألم بطن"), sx("نزيف مهبلي")])
    explicit_male_context = check_red_flags(
        [sx("ألم بطن"), sx("نزيف مهبلي")], "مريض ذكر، عمره 45 سنة"
    )

    assert "ectopic_pregnancy" in rule_ids(no_context)
    assert "ectopic_pregnancy" in rule_ids(explicit_male_context)


# --- generic/specific symptom subsumption (_SUBSUMES) -----------------------
#
# Regression coverage for the gap where a rule written against a generic
# term (e.g. "حمى") silently never fired against a real extraction that
# correctly produced a more specific canonical sibling (e.g. "حمى مرتفعة
# مفاجئة") instead — see rules/red_flags.py's _SUBSUMES for the full
# audit and reasoning.


def test_bacterial_meningitis_fires_on_specific_fever_term_not_just_bare_fever():
    # Direct regression for tests/golden/cases.json's emergency_03: "حمى
    # عالية" (high fever) extracts to "حمى مرتفعة مفاجئة", not the bare
    # "حمى" the rule was originally written against.
    matches = check_red_flags([sx("حمى مرتفعة مفاجئة"), sx("تيبس الرقبة")])

    assert "bacterial_meningitis" in rule_ids(matches)


def test_bacterial_meningitis_still_fires_on_the_bare_generic_fever_term():
    # The original, already-working path must still work — subsumption is
    # additive, never a replacement for the exact match.
    matches = check_red_flags([sx("حمى"), sx("تيبس الرقبة")])

    assert "bacterial_meningitis" in rule_ids(matches)


def test_bacterial_meningitis_does_not_fire_on_specific_fever_term_alone():
    # Subsumption satisfies all_of via the specific sibling, but any_of is
    # still independently required — the fix must not bypass that gate.
    matches = check_red_flags([sx("حمى مرتفعة مفاجئة")])

    assert "bacterial_meningitis" not in rule_ids(matches)


def test_sepsis_fires_on_specific_mild_fever_term():
    matches = check_red_flags([sx("حمى خفيفة"), sx("تخليط ذهني مفاجئ")])

    assert "sepsis" in rule_ids(matches)


def test_sepsis_chronic_requirement_fires_on_specific_fever_term():
    # chronic_requirement.all_of={"حمى"} alone (no any_of) — same gap,
    # same fix, on the chronic-lowered branch specifically.
    matches = check_red_flags([sx("حمى مرتفعة مفاجئة")], "مريض مصاب بسرطان")

    sepsis = next(m for m in matches if m.rule_id == "sepsis")
    assert sepsis.lowered_by_chronic_condition is True


def test_ectopic_pregnancy_fires_on_lower_abdominal_pain_specifically():
    # Arguably the most clinically likely real-world phrasing for this
    # exact condition — classic ectopic presentation is lower
    # abdominal/pelvic pain — so this was the more concerning half of the
    # bug, not just the fever rules.
    matches = check_red_flags([sx("ألم أسفل البطن"), sx("نزيف مهبلي")])

    assert "ectopic_pregnancy" in rule_ids(matches)


def test_ectopic_pregnancy_fires_on_upper_abdominal_pain_specifically():
    matches = check_red_flags([sx("ألم أعلى البطن"), sx("تأخر الدورة الشهرية")])

    assert "ectopic_pregnancy" in rule_ids(matches)


def test_ectopic_pregnancy_fires_on_generalized_abdominal_pain_specifically():
    matches = check_red_flags([sx("ألم بطن معمم"), sx("نزيف مهبلي")])

    assert "ectopic_pregnancy" in rule_ids(matches)


def test_ectopic_pregnancy_still_fires_on_the_bare_generic_abdominal_pain_term():
    matches = check_red_flags([sx("ألم بطن"), sx("نزيف مهبلي")])

    assert "ectopic_pregnancy" in rule_ids(matches)


def test_ectopic_pregnancy_does_not_fire_on_specific_abdominal_pain_alone():
    # Same any_of-still-required check as meningitis above, for the
    # abdominal-pain side of the fix.
    matches = check_red_flags([sx("ألم أسفل البطن")])

    assert "ectopic_pregnancy" not in rule_ids(matches)


def test_specific_fever_term_reports_the_actual_matched_symptom_not_the_generic_one():
    # matched_symptoms should reflect what the patient actually said (see
    # _term_matches's own docstring), not the generic rule term they never
    # literally used.
    matches = check_red_flags([sx("حمى مرتفعة مفاجئة"), sx("تيبس الرقبة")])

    meningitis = next(m for m in matches if m.rule_id == "bacterial_meningitis")
    assert normalize("حمى مرتفعة مفاجئة") in meningitis.matched_symptoms
    assert normalize("حمى") not in meningitis.matched_symptoms


def test_unrelated_specific_symptom_does_not_spuriously_trigger_any_rule():
    # Confirms subsumption is scoped to the two explicit _SUBSUMES pairs,
    # not a general fuzzy match — a specific symptom with no generic
    # parent referenced by any rule must not trigger anything.
    matches = check_red_flags([sx("ألم أسفل الظهر")])  # Dysmenorrhea-only term

    assert matches == []
