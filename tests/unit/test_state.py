import state
from state import merge_symptoms, merge_unmatched_mentions


def test_merge_symptoms_basic_dedup_and_order():
    existing = [{"name": "صداع"}]
    new = [{"name": "حمى"}, {"name": "صداع"}]

    result = merge_symptoms(existing, new)

    assert result == [{"name": "صداع"}, {"name": "حمى"}]


def test_merge_symptoms_fills_in_sparse_fields_without_erasing_existing():
    existing = [{"name": "غثيان", "duration": "يومين"}]
    new = [{"name": "دوخة"}, {"name": "غثيان", "severity": "شديد"}]

    result = merge_symptoms(existing, new)

    assert result == [
        {"name": "غثيان", "duration": "يومين", "severity": "شديد"},
        {"name": "دوخة"},
    ]


def test_merge_symptoms_new_entry_does_not_erase_existing_field_with_empty_value():
    existing = [{"name": "غثيان", "duration": "يومين"}]
    new = [{"name": "غثيان", "duration": "", "severity": None}]

    result = merge_symptoms(existing, new)

    assert result == [{"name": "غثيان", "duration": "يومين"}]


def test_merge_symptoms_does_not_mutate_inputs():
    existing = [{"name": "غثيان", "duration": "يومين"}]
    new = [{"name": "غثيان", "severity": "شديد"}]
    existing_snapshot = [dict(s) for s in existing]
    new_snapshot = [dict(s) for s in new]

    merge_symptoms(existing, new)

    assert existing == existing_snapshot
    assert new == new_snapshot


def test_merge_symptoms_handles_none_existing():
    result = merge_symptoms(None, [{"name": "صداع"}])

    assert result == [{"name": "صداع"}]


def test_merge_symptoms_handles_none_new():
    result = merge_symptoms([{"name": "صداع"}], None)

    assert result == [{"name": "صداع"}]


def test_merge_symptoms_handles_both_none():
    assert merge_symptoms(None, None) == []


def test_merge_symptoms_handles_empty_lists():
    assert merge_symptoms([], []) == []


def test_merge_symptoms_skips_and_logs_entry_missing_name(monkeypatch):
    logged = []
    monkeypatch.setattr(state, "log_malformed_output", lambda **kwargs: logged.append(kwargs))

    result = merge_symptoms([], [{"duration": "يومين"}, {"name": "صداع"}])

    assert result == [{"name": "صداع"}]
    assert len(logged) == 1
    assert logged[0]["node"] == "merge_symptoms"
    assert logged[0]["reason"] == "missing_name"
    assert logged[0]["payload"] == {"duration": "يومين"}


def test_merge_symptoms_skips_and_logs_entry_with_empty_name(monkeypatch):
    logged = []
    monkeypatch.setattr(state, "log_malformed_output", lambda **kwargs: logged.append(kwargs))

    result = merge_symptoms([], [{"name": ""}, {"name": "صداع"}])

    assert result == [{"name": "صداع"}]
    assert len(logged) == 1
    assert logged[0]["reason"] == "empty_name"
    assert logged[0]["payload"] == {"name": ""}


def test_merge_symptoms_does_not_raise_when_only_entry_is_malformed(monkeypatch):
    monkeypatch.setattr(state, "log_malformed_output", lambda **kwargs: None)

    result = merge_symptoms([], [{"duration": "يومين"}])

    assert result == []


# --- merge_unmatched_mentions ---------------------------------------------------


def test_merge_unmatched_mentions_accumulates_across_turns():
    existing = ["طنين بالأذن"]
    new = ["دوار عند الوقوف"]

    result = merge_unmatched_mentions(existing, new)

    assert result == ["طنين بالأذن", "دوار عند الوقوف"]


def test_merge_unmatched_mentions_dedupes_by_exact_string():
    existing = ["طنين بالأذن"]
    new = ["طنين بالأذن", "دوار عند الوقوف"]

    result = merge_unmatched_mentions(existing, new)

    assert result == ["طنين بالأذن", "دوار عند الوقوف"]


def test_merge_unmatched_mentions_does_not_fuzzy_match():
    # Deliberately no normalization here (CLAUDE.md: "no fuzzy merging
    # needed") — a near-identical phrase is kept as its own entry.
    existing = ["طنين بالاذن"]
    new = ["طنين بالأذن"]

    result = merge_unmatched_mentions(existing, new)

    assert result == ["طنين بالاذن", "طنين بالأذن"]


def test_merge_unmatched_mentions_preserves_first_seen_order():
    result = merge_unmatched_mentions(["ب", "أ"], ["ج", "أ"])

    assert result == ["ب", "أ", "ج"]


def test_merge_unmatched_mentions_handles_none_existing():
    assert merge_unmatched_mentions(None, ["طنين بالأذن"]) == ["طنين بالأذن"]


def test_merge_unmatched_mentions_handles_none_new():
    assert merge_unmatched_mentions(["طنين بالأذن"], None) == ["طنين بالأذن"]


def test_merge_unmatched_mentions_handles_both_none():
    assert merge_unmatched_mentions(None, None) == []


def test_merge_unmatched_mentions_handles_empty_lists():
    assert merge_unmatched_mentions([], []) == []


def test_merge_unmatched_mentions_does_not_mutate_inputs():
    existing = ["طنين بالأذن"]
    new = ["دوار عند الوقوف"]
    existing_snapshot = list(existing)
    new_snapshot = list(new)

    merge_unmatched_mentions(existing, new)

    assert existing == existing_snapshot
    assert new == new_snapshot
