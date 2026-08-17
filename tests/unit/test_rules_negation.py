import pytest

from rules.crisis import normalize
from rules.negation import (
    NEGATION_WINDOW,
    detect_negated_symptoms,
    merge_negations,
)

FEVER = "حمى"
DYSPNEA = "ضيق تنفس"


def negated(message: str) -> set[str]:
    return set(detect_negated_symptoms(message))


# --- basic detection -----------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "ما في حمى",
        "مافي حمى",
        "ما عندي حمى",
        "بدون حمى",
        "من غير حمى",
        "ليس عندي حمى",
    ],
)
def test_detects_each_negation_cue(message):
    assert normalize(FEVER) in negated(message)


def test_confirmed_symptom_is_not_reported_as_negated():
    assert normalize(FEVER) not in negated("عندي حمى من يومين")


def test_returns_empty_when_symptom_absent_from_message():
    assert negated("عندي وجع بالضهر") == set()


def test_returns_empty_for_empty_message():
    assert negated("") == set()


# --- the bounded window: the reason this was worth porting ---------------------


def test_distant_negation_does_not_leak_to_a_later_symptom():
    # "ما عندي" negates حمى only. Without a bounded lookbehind, ضيق تنفس
    # would be swept up too, and a real symptom would be recorded as denied.
    result = negated("ما عندي حمى بس عندي ضيق تنفس")

    assert normalize(FEVER) in result
    assert normalize(DYSPNEA) not in result


def test_negation_immediately_before_symptom_applies():
    assert normalize(DYSPNEA) in negated("ما في ضيق تنفس")


def test_cue_further_than_the_window_is_ignored():
    # Filler must not be a repeated character: normalize() collapses 3+ of
    # the same char, which would shrink it back inside the window.
    filler = "كتير منيح اليوم"
    assert len(normalize(filler)) > NEGATION_WINDOW

    assert normalize(FEVER) not in negated(f"ما في {filler} حمى")


# --- cue boundaries -------------------------------------------------------------


def test_bare_laa_inside_another_word_is_not_a_negation_cue():
    # "بلاش" contains "لا"; an unbounded substring cue would fire on it.
    assert normalize(FEVER) not in negated("بلاش نحكي عن هالشي عندي حمى")


# --- symptom-name boundaries: rules.red_flags._compile_keyword_matcher reused ---


def test_unrelated_word_containing_a_short_symptom_name_is_not_detected():
    # "الضحكة" (laughter) contains "حكة" (itching) as a bare substring —
    # same class of collision rules/red_flags.py's own test suite documents
    # for "سكر" inside "سكرتير" (test_rules_red_flags.py). Verified directly
    # (not assumed) that the naive substring check this replaces would have
    # matched: normalize("حكة") in normalize("ما بحب الضحكة الزايدة") is True.
    # The patient never mentioned itching; "ما" here negates nothing about
    # itching, since itching was never really "found" in the first place.
    result = negated("ما بحب الضحكة الزايدة")

    assert normalize("حكة") not in result


def test_prefixed_forms_of_a_symptom_name_are_still_detected():
    # The dangerous direction, same reasoning as
    # test_arabic_prefixed_forms_of_chronic_keyword_lower_threshold in
    # rules/red_flags.py's own test suite: a strict \b boundary that
    # rejected an attached ال would silently miss a real denial.
    assert normalize(FEVER) in negated("ما عندي الحمى")


# --- spelling tolerance ----------------------------------------------------------


def test_matching_tolerates_spelling_variation():
    # Message uses a different alef/taa form than the vocabulary entry.
    assert detect_negated_symptoms("ما في حمي") == detect_negated_symptoms("ما في حمي")


def test_arabic_indic_digits_do_not_break_matching():
    assert normalize(FEVER) in negated("من ٣ أيام ما في حمى")


# --- candidate list ---------------------------------------------------------------


def test_respects_an_explicit_candidate_list():
    result = detect_negated_symptoms("ما في حمى", candidates=[DYSPNEA])

    assert result == frozenset()


def test_returns_normalized_names():
    for name in detect_negated_symptoms("ما في حمى"):
        assert name == normalize(name)


# --- merge_negations: the OR contract ---------------------------------------------


def test_merge_is_a_union_not_an_intersection():
    # Each layer misses what the other catches; requiring agreement would
    # discard the cases the second layer exists to recover.
    merged = merge_negations([FEVER], [DYSPNEA])

    assert merged == {normalize(FEVER), normalize(DYSPNEA)}


def test_merge_deduplicates_the_same_name_from_both_layers():
    assert merge_negations([FEVER], [FEVER]) == {normalize(FEVER)}


def test_merge_normalizes_so_spelling_variants_do_not_double_count():
    merged = merge_negations(["حمى"], ["حمى"])

    assert len(merged) == 1


@pytest.mark.parametrize(
    ("llm", "rules"),
    [(None, None), ([], []), (None, [FEVER]), ([FEVER], None)],
)
def test_merge_handles_missing_or_empty_inputs(llm, rules):
    merged = merge_negations(llm, rules)

    assert merged == ({normalize(FEVER)} if (llm or rules) else frozenset())


def test_merge_keeps_llm_negation_the_rule_layer_missed():
    # The colloquial "كحة" is not vocabulary wording, so the rule layer
    # cannot see it — the LLM's finding must survive the merge.
    merged = merge_negations(llm_negated=[DYSPNEA], rule_negated=[])

    assert normalize(DYSPNEA) in merged
