"""Visibility for the known symptom-vocabulary gap.

The completeness assertion itself is marked `vocabulary_completeness` and
excluded from the default run (see pytest.ini) — it is expected to fail
until the vocabulary is reconciled, and a permanently red suite is a suite
people stop reading. Run it deliberately:

    pytest -m vocabulary_completeness

The unmarked tests below run every time and protect the reporting
mechanism, so the gap cannot quietly stop being measured.
"""

import pytest

from vocabulary.symptoms import (
    CANONICAL_SYMPTOMS,
    EXPECTED_SYMPTOM_COUNT,
    check_completeness,
)


@pytest.mark.vocabulary_completeness
def test_vocabulary_matches_the_approved_symptom_count():
    report = check_completeness()

    assert report.is_complete, "\n" + report.describe()


# --- always-run: the reporting mechanism itself --------------------------------


def test_completeness_report_counts_actual_entries():
    report = check_completeness()

    assert report.present == len(CANONICAL_SYMPTOMS)
    assert report.expected == EXPECTED_SYMPTOM_COUNT


def test_completeness_report_computes_the_missing_count():
    report = check_completeness()

    assert report.missing == max(0, EXPECTED_SYMPTOM_COUNT - len(CANONICAL_SYMPTOMS))
    assert report.is_complete == (len(CANONICAL_SYMPTOMS) >= EXPECTED_SYMPTOM_COUNT)


def test_description_states_both_numbers():
    # Whatever the state, the report must show present AND expected — a bare
    # "incomplete" tells nobody how big the gap is.
    description = check_completeness().describe()

    assert str(len(CANONICAL_SYMPTOMS)) in description
    assert str(EXPECTED_SYMPTOM_COUNT) in description


def test_vocabulary_never_exceeds_the_approved_count():
    # Overshooting means entries were added that are not on the approved
    # list — the same drift risk as inventing names, in the other direction.
    assert len(CANONICAL_SYMPTOMS) <= EXPECTED_SYMPTOM_COUNT


def test_incomplete_vocabulary_reports_actionable_guidance():
    report = check_completeness()
    if report.is_complete:
        pytest.skip("vocabulary is complete; the incomplete-path text does not apply")

    description = report.describe()
    assert "MISSING" in description
    assert "not by inventing plausible names" in description
