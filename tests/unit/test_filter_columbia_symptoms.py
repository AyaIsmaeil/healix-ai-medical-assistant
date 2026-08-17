import csv
import json

import pytest

import llm_client
from llm_client import LLMConfigError, LLMUnavailable, LLMValidationError, _ProviderResponse, set_provider
from scripts.filter_columbia_symptoms import (
    Bucket,
    ClassificationError,
    ClassificationReport,
    ClusteringIncomplete,
    ClassifiedTerm,
    DuplicateGroup,
    _blocking_keys,
    _load_cluster_progress,
    _validate_batch_response,
    block_by_shared_tokens,
    chunk,
    classify_all,
    classify_batch,
    extract_candidate_terms,
    find_near_duplicates,
    find_near_duplicates_batched,
    load_classifications,
    main,
    pack_blocks,
    save_classifications,
    write_reports,
)


class FakeProvider:
    """Scripted provider: returns `responses` in order, or raises `errors`."""

    name = "fake"

    def __init__(self, *, responses=None, errors=None):
        self._responses = list(responses or [])
        self._errors = list(errors or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        if self._errors:
            error = self._errors.pop(0)
            if error is not None:
                raise error
        return self._responses.pop(0)


def classification_response(pairs: list[tuple[int, str, str]]) -> _ProviderResponse:
    """pairs: (index, bucket_value, reason)."""
    payload = {
        "classifications": [
            {"index": i, "bucket": b, "reason": r} for i, b, r in pairs
        ]
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def duplicate_response(groups: list[tuple[str, list[int]]]) -> _ProviderResponse:
    payload = {
        "groups": [
            {"canonical_suggestion": name, "member_indices": members}
            for name, members in groups
        ]
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("HEALIX_MODEL_FAST", "fake-fast-model")
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


# --- extract_candidate_terms --------------------------------------------------


def test_extract_candidate_terms_reads_the_header_row(tmp_path):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fever", "cough", "prognosis"])
        writer.writerow(["1", "0", "flu"])

    assert extract_candidate_terms(csv_path) == ["fever", "cough", "prognosis"]


def test_extract_candidate_terms_rejects_empty_header(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError):
        extract_candidate_terms(csv_path)


# --- chunk ---------------------------------------------------------------------


def test_chunk_splits_evenly():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_handles_a_remainder():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_single_batch_when_size_covers_everything():
    assert chunk([1, 2, 3], 10) == [[1, 2, 3]]


def test_chunk_empty_input_returns_no_batches():
    assert chunk([], 5) == []


def test_chunk_rejects_non_positive_size():
    with pytest.raises(ValueError):
        chunk([1, 2], 0)


# --- _validate_batch_response ---------------------------------------------------


def test_validate_accepts_exact_index_match():
    batch = [(0, "fever"), (1, "cough")]
    response = classification_response(
        [(0, "patient_reportable", "r"), (1, "patient_reportable", "r")]
    ).text
    from scripts.filter_columbia_symptoms import ClassificationBatch

    parsed = ClassificationBatch.model_validate_json(response)
    _validate_batch_response(batch, parsed)  # must not raise


def test_validate_rejects_missing_index():
    from scripts.filter_columbia_symptoms import ClassificationBatch

    batch = [(0, "fever"), (1, "cough")]
    parsed = ClassificationBatch.model_validate(
        {"classifications": [{"index": 0, "bucket": "patient_reportable", "reason": "r"}]}
    )
    with pytest.raises(ClassificationError, match=r"missing=\[1\]"):
        _validate_batch_response(batch, parsed)


def test_validate_rejects_hallucinated_extra_index():
    from scripts.filter_columbia_symptoms import ClassificationBatch

    batch = [(0, "fever")]
    parsed = ClassificationBatch.model_validate(
        {
            "classifications": [
                {"index": 0, "bucket": "patient_reportable", "reason": "r"},
                {"index": 99, "bucket": "noise", "reason": "r"},
            ]
        }
    )
    with pytest.raises(ClassificationError, match=r"extra=\[99\]"):
        _validate_batch_response(batch, parsed)


def test_validate_rejects_duplicate_index():
    from scripts.filter_columbia_symptoms import ClassificationBatch

    batch = [(0, "fever"), (1, "cough")]
    parsed = ClassificationBatch.model_validate(
        {
            "classifications": [
                {"index": 0, "bucket": "patient_reportable", "reason": "r"},
                {"index": 0, "bucket": "noise", "reason": "r"},
            ]
        }
    )
    with pytest.raises(ClassificationError):
        _validate_batch_response(batch, parsed)


# --- classify_batch: happy path -------------------------------------------------


def test_classify_batch_returns_classifications_in_one_call():
    provider = FakeProvider(
        responses=[
            classification_response(
                [(0, "patient_reportable", "common complaint"), (1, "exam_finding", "needs auscultation")]
            )
        ]
    )
    set_provider(provider)

    result = classify_batch([(0, "cough"), (1, "rale")])

    assert {c.index: c.bucket for c in result} == {0: Bucket.patient_reportable, 1: Bucket.exam_finding}
    assert len(provider.calls) == 1


def test_classify_batch_uses_quality_tier():
    provider = FakeProvider(
        responses=[classification_response([(0, "noise", "meta field")])]
    )
    set_provider(provider)

    classify_batch([(0, "prognosis")])

    assert provider.calls[0]["model"] == "fake-quality-model"


# --- classify_batch: bisection on incomplete response ---------------------------


def test_classify_batch_bisects_on_missing_index_and_recovers():
    # First call (full batch of 2) omits index 1 -> triggers bisection into
    # two size-1 sub-batches, each classified correctly on its own call.
    provider = FakeProvider(
        responses=[
            classification_response([(0, "patient_reportable", "r")]),  # index 1 missing
            classification_response([(0, "patient_reportable", "r")]),  # sub-batch [0]
            classification_response([(1, "exam_finding", "r")]),        # sub-batch [1]
        ]
    )
    set_provider(provider)

    result = classify_batch([(0, "fever"), (1, "rale")])

    assert {c.index: c.bucket for c in result} == {0: Bucket.patient_reportable, 1: Bucket.exam_finding}
    assert len(provider.calls) == 3


def test_classify_batch_bisects_on_llm_error():
    provider = FakeProvider(
        errors=[LLMValidationError("malformed"), None, None],
        responses=[
            classification_response([(0, "noise", "r")]),
            classification_response([(1, "noise", "r")]),
        ],
    )
    set_provider(provider)

    result = classify_batch([(0, "asymptomatic"), (1, "difficulty")])

    assert {c.index for c in result} == {0, 1}


def test_classify_batch_does_not_bisect_a_config_error():
    # A config error (bad model name, missing key, ...) cannot change
    # between attempts, so it must surface on the very first call — no
    # bisection into sub-batches, no retry, and it must be the original
    # LLMConfigError, not wrapped in ClassificationError.
    provider = FakeProvider(errors=[LLMConfigError("HEALIX_MODEL_QUALITY is not set")])
    set_provider(provider)

    with pytest.raises(LLMConfigError, match="HEALIX_MODEL_QUALITY"):
        classify_batch([(0, "fever"), (1, "rale")])

    assert len(provider.calls) == 1


def test_classify_batch_raises_naming_the_term_when_single_item_fails_persistently():
    provider = FakeProvider(
        errors=[LLMValidationError("still bad")],
    )
    set_provider(provider)

    with pytest.raises(ClassificationError, match="mystery term"):
        classify_batch([(7, "mystery term")])


def test_classify_batch_does_not_silently_drop_a_persistently_failing_term_in_a_larger_batch():
    # index 1 never resolves; index 0 always succeeds. The failure for 1
    # must surface as an exception, not vanish from the result set.
    provider = FakeProvider(
        responses=[
            classification_response([(0, "noise", "r")]),  # full batch: 1 missing
            classification_response([(0, "noise", "r")]),  # sub-batch [0]: fine
        ],
        errors=[None, None, LLMValidationError("bad"), LLMValidationError("bad")],
    )
    set_provider(provider)

    with pytest.raises(ClassificationError, match="broken term"):
        classify_batch([(0, "fine term"), (1, "broken term")])


# --- classify_all ---------------------------------------------------------------


def test_classify_all_covers_every_candidate_across_multiple_batches():
    candidates = ["fever", "rale", "hyponatremia", "cardiomegaly"]
    provider = FakeProvider(
        responses=[
            classification_response(
                [(0, "patient_reportable", "r"), (1, "exam_finding", "r")]
            ),
            classification_response(
                [(2, "lab_result", "r"), (3, "imaging_or_ecg", "r")]
            ),
        ]
    )
    set_provider(provider)

    result = classify_all(candidates, batch_size=2)

    assert set(result) == {0, 1, 2, 3}
    assert result[2].bucket is Bucket.lab_result


# --- find_near_duplicates --------------------------------------------------------


def test_find_near_duplicates_makes_no_call_for_empty_input():
    provider = FakeProvider()
    set_provider(provider)

    assert find_near_duplicates({}) == []
    assert provider.calls == []


def test_find_near_duplicates_returns_groups():
    provider = FakeProvider(
        responses=[
            duplicate_response(
                [("chest pain", [3, 5]), ("sweating", [10, 11, 12])]
            )
        ]
    )
    set_provider(provider)

    groups = find_near_duplicates(
        {3: "pain chest", 5: "chest discomfort", 10: "sweat", 11: "sweating increased", 12: "hyperhidrosis disorder"}
    )

    assert len(groups) == 2
    assert {3, 5} == set(groups[0].member_indices)


def test_find_near_duplicates_drops_hallucinated_indices():
    provider = FakeProvider(
        responses=[duplicate_response([("chest pain", [3, 5, 999])])]
    )
    set_provider(provider)

    groups = find_near_duplicates({3: "pain chest", 5: "chest discomfort"})

    assert groups[0].member_indices == [3, 5]


def test_find_near_duplicates_drops_groups_that_collapse_to_a_singleton():
    provider = FakeProvider(
        responses=[duplicate_response([("chest pain", [3, 999])])]
    )
    set_provider(provider)

    groups = find_near_duplicates({3: "pain chest"})

    assert groups == []


# --- write_reports ---------------------------------------------------------------


def _report(candidates, bucket_assignments, duplicate_groups=None):
    from scripts.filter_columbia_symptoms import ClassifiedTerm

    classifications = {
        i: ClassifiedTerm(index=i, bucket=bucket, reason=f"reason-{i}")
        for i, bucket in bucket_assignments.items()
    }
    return ClassificationReport(
        candidates=candidates,
        classifications=classifications,
        duplicate_groups=duplicate_groups or [],
    )


def test_write_reports_creates_one_file_per_bucket(tmp_path):
    report = _report(
        ["fever", "rale", "hyponatremia", "cardiomegaly", "suicidal", "homeless", "prognosis"],
        {
            0: Bucket.patient_reportable,
            1: Bucket.exam_finding,
            2: Bucket.lab_result,
            3: Bucket.imaging_or_ecg,
            4: Bucket.psychiatric_crisis,
            5: Bucket.social_or_status,
            6: Bucket.noise,
        },
    )

    write_reports(report, tmp_path)

    for bucket in Bucket:
        assert (tmp_path / f"{bucket.value}.md").exists()
    assert (tmp_path / "near_duplicates.md").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "summary.json").exists()


def test_bucket_file_with_zero_entries_is_still_written(tmp_path):
    report = _report(["fever"], {0: Bucket.patient_reportable})

    write_reports(report, tmp_path)

    content = (tmp_path / "imaging_or_ecg.md").read_text(encoding="utf-8")
    assert "(0 entries)" in content


def test_psychiatric_crisis_file_has_the_quarantine_explanation(tmp_path):
    report = _report(["suicidal"], {0: Bucket.psychiatric_crisis})

    write_reports(report, tmp_path)

    content = (tmp_path / "psychiatric_crisis.md").read_text(encoding="utf-8")
    assert "QUARANTINED" in content
    assert "crisis path" in content
    assert "suicidal" in content


def test_psychiatric_crisis_entries_do_not_leak_into_other_bucket_files(tmp_path):
    report = _report(
        ["headache", "suicidal", "chest pain"],
        {0: Bucket.patient_reportable, 1: Bucket.psychiatric_crisis, 2: Bucket.patient_reportable},
    )

    write_reports(report, tmp_path)

    for bucket in Bucket:
        if bucket is Bucket.psychiatric_crisis:
            continue
        content = (tmp_path / f"{bucket.value}.md").read_text(encoding="utf-8")
        assert "suicidal" not in content


def test_patient_reportable_file_annotates_duplicate_membership(tmp_path):
    report = _report(
        ["pain chest", "chest discomfort", "fever"],
        {0: Bucket.patient_reportable, 1: Bucket.patient_reportable, 2: Bucket.patient_reportable},
        duplicate_groups=[DuplicateGroup(canonical_suggestion="chest pain", member_indices=[0, 1])],
    )

    write_reports(report, tmp_path)

    content = (tmp_path / "patient_reportable.md").read_text(encoding="utf-8")
    assert 'near-duplicate: "chest pain"' in content.split("pain chest")[1].split("\n")[0]
    assert "near-duplicate" not in content.split("fever")[1]


def test_near_duplicates_report_lists_group_members(tmp_path):
    report = _report(
        ["pain chest", "chest discomfort"],
        {0: Bucket.patient_reportable, 1: Bucket.patient_reportable},
        duplicate_groups=[DuplicateGroup(canonical_suggestion="chest pain", member_indices=[0, 1])],
    )

    write_reports(report, tmp_path)

    content = (tmp_path / "near_duplicates.md").read_text(encoding="utf-8")
    assert "chest pain" in content
    assert "pain chest" in content
    assert "chest discomfort" in content


def test_summary_counts_match_the_classification(tmp_path):
    report = _report(
        ["a", "b", "c", "d"],
        {
            0: Bucket.patient_reportable,
            1: Bucket.patient_reportable,
            2: Bucket.noise,
            3: Bucket.psychiatric_crisis,
        },
    )

    write_reports(report, tmp_path)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["total"] == 4
    assert payload["counts"]["patient_reportable"] == 2
    assert payload["counts"]["noise"] == 1
    assert payload["counts"]["psychiatric_crisis"] == 1
    assert payload["counts"]["exam_finding"] == 0
    assert sum(payload["counts"].values()) == payload["total"]


def test_summary_json_and_markdown_agree_on_duplicate_group_count(tmp_path):
    report = _report(
        ["pain chest", "chest discomfort"],
        {0: Bucket.patient_reportable, 1: Bucket.patient_reportable},
        duplicate_groups=[DuplicateGroup(canonical_suggestion="chest pain", member_indices=[0, 1])],
    )

    write_reports(report, tmp_path)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["duplicate_group_count"] == 1
    assert "1" in (tmp_path / "summary.md").read_text(encoding="utf-8")


# --- main(): end-to-end wiring with --yes and --limit ----------------------------


def test_main_end_to_end_with_fake_provider(tmp_path, capsys):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["fever", "prognosis", "asymptomatic", "difficulty"])

    out_dir = tmp_path / "out"
    provider = FakeProvider(
        responses=[
            classification_response(
                [(0, "noise", "r"), (1, "noise", "r"), (2, "noise", "r"), (3, "noise", "r")]
            )
        ]
    )
    set_provider(provider)

    exit_code = main(
        ["--input", str(csv_path), "--out-dir", str(out_dir), "--yes"]
    )

    assert exit_code == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["total"] == 4
    assert payload["counts"]["noise"] == 4
    # All-noise input means nothing is patient_reportable, so no clustering
    # call should have been made.
    assert len(provider.calls) == 1
    assert "Reports written to" in capsys.readouterr().out


def test_main_respects_limit_for_a_cheap_dry_run(tmp_path):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["fever", "rale", "hyponatremia", "cardiomegaly"])

    provider = FakeProvider(responses=[classification_response([(0, "noise", "r")])])
    set_provider(provider)

    main(["--input", str(csv_path), "--out-dir", str(tmp_path / "out"), "--yes", "--limit", "1"])

    payload = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert payload["total"] == 1


def test_main_aborts_without_calling_the_provider_when_confirmation_declined(tmp_path, monkeypatch):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["fever"])

    provider = FakeProvider()
    set_provider(provider)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    exit_code = main(["--input", str(csv_path), "--out-dir", str(tmp_path / "out")])

    assert exit_code == 0
    assert provider.calls == []
    assert not (tmp_path / "out").exists()


# --- classification checkpointing -------------------------------------------------


def test_save_and_load_classifications_round_trips(tmp_path):
    candidates = ["fever", "rale"]
    classifications = {
        0: ClassifiedTerm(index=0, bucket=Bucket.patient_reportable, reason="r0"),
        1: ClassifiedTerm(index=1, bucket=Bucket.exam_finding, reason="r1"),
    }

    save_classifications(candidates, classifications, tmp_path)
    loaded = load_classifications(candidates, tmp_path)

    assert loaded == classifications
    assert (tmp_path / "classifications.json").exists()


def test_load_classifications_returns_none_when_no_checkpoint(tmp_path):
    assert load_classifications(["fever"], tmp_path) is None


def test_load_classifications_returns_none_on_candidate_mismatch(tmp_path):
    save_classifications(
        ["fever"], {0: ClassifiedTerm(index=0, bucket=Bucket.noise, reason="r")}, tmp_path
    )

    assert load_classifications(["cough"], tmp_path) is None


def test_load_classifications_returns_none_on_corrupt_checkpoint(tmp_path):
    (tmp_path / "classifications.json").write_text("not json", encoding="utf-8")

    assert load_classifications(["fever"], tmp_path) is None


# --- blocking for clustering batching ----------------------------------------------


def test_blocking_keys_share_an_exact_token():
    assert "chest" in _blocking_keys("pain chest")
    assert "chest" in _blocking_keys("chest discomfort")


def test_blocking_keys_share_a_prefix_for_morphological_variants():
    # "sweat" and "sweating" don't match as whole tokens, but the prefix
    # heuristic gives them a shared key — see the module note in
    # scripts/filter_columbia_symptoms.py on why this exists.
    assert _blocking_keys("sweat") & _blocking_keys("sweating increased")


def test_blocking_keys_do_not_connect_true_synonyms_with_no_lexical_overlap():
    # The disclosed limitation: "hyperhidrosis disorder" is the same concept
    # as "sweat" to a clinician, but shares no token or prefix with it.
    assert not (_blocking_keys("sweat") & _blocking_keys("hyperhidrosis disorder"))


def test_blocking_keys_drop_stopwords_and_short_tokens():
    keys = _blocking_keys("pain in the a")
    assert "in" not in keys
    assert "the" not in keys
    assert "a" not in keys


def test_block_by_shared_tokens_groups_related_terms_together():
    reportable = {0: "pain chest", 1: "chest discomfort", 2: "headache"}

    blocks = block_by_shared_tokens(reportable)

    block_of = {i: frozenset(b) for b in blocks for i in b}
    assert block_of[0] == block_of[1]
    assert block_of[2] != block_of[0]


def test_block_by_shared_tokens_unrelated_terms_get_singleton_blocks():
    reportable = {0: "headache", 1: "nausea"}

    blocks = block_by_shared_tokens(reportable)

    assert sorted(len(b) for b in blocks) == [1, 1]


def test_block_by_shared_tokens_is_deterministic():
    reportable = {3: "chest pain", 1: "pain chest", 5: "fever"}

    assert block_by_shared_tokens(reportable) == block_by_shared_tokens(reportable)


# --- pack_blocks ---------------------------------------------------------------------


def test_pack_blocks_never_splits_a_block_that_fits():
    batches = pack_blocks([[0, 1], [2, 3]], batch_size=4)

    for block in ([0, 1], [2, 3]):
        containing = [b for b in batches if set(block) <= set(b)]
        assert len(containing) == 1


def test_pack_blocks_respects_the_batch_size_cap():
    batches = pack_blocks([[0], [1], [2], [3]], batch_size=2)

    assert all(len(b) <= 2 for b in batches)
    assert sorted(sum(batches, [])) == [0, 1, 2, 3]


def test_pack_blocks_splits_a_block_larger_than_batch_size():
    batches = pack_blocks([[0, 1, 2, 3, 4]], batch_size=2)

    assert all(len(b) <= 2 for b in batches)
    assert sorted(sum(batches, [])) == [0, 1, 2, 3, 4]


def test_pack_blocks_empty_input_returns_no_batches():
    assert pack_blocks([], batch_size=5) == []


# --- find_near_duplicates_batched -----------------------------------------------------


def test_find_near_duplicates_batched_empty_input_makes_no_call(tmp_path):
    provider = FakeProvider()
    set_provider(provider)

    assert find_near_duplicates_batched({}, tmp_path) == []
    assert provider.calls == []


def test_find_near_duplicates_batched_groups_related_terms_in_one_call(tmp_path):
    provider = FakeProvider(responses=[duplicate_response([("chest pain", [0, 1])])])
    set_provider(provider)

    groups = find_near_duplicates_batched(
        {0: "pain chest", 1: "chest discomfort"}, tmp_path, batch_size=10
    )

    assert len(provider.calls) == 1
    assert {0, 1} == set(groups[0].member_indices)


def test_find_near_duplicates_batched_splits_unrelated_terms_into_separate_calls(tmp_path):
    # "headache" and "nausea" share no token/prefix, so at batch_size=1 each
    # must get its own clustering call.
    provider = FakeProvider(responses=[duplicate_response([]), duplicate_response([])])
    set_provider(provider)

    groups = find_near_duplicates_batched({0: "headache", 1: "nausea"}, tmp_path, batch_size=1)

    assert len(provider.calls) == 2
    assert groups == []


def test_find_near_duplicates_batched_checkpoints_after_each_batch(tmp_path):
    provider = FakeProvider(responses=[duplicate_response([]), duplicate_response([])])
    set_provider(provider)

    find_near_duplicates_batched({0: "headache", 1: "nausea"}, tmp_path, batch_size=1)

    done = _load_cluster_progress(tmp_path)
    assert len(done) == 2
    assert (tmp_path / "duplicate_groups.json").exists()


def test_find_near_duplicates_batched_raises_incomplete_and_keeps_completed_batches(tmp_path):
    # First clustering batch succeeds and checkpoints; second one fails.
    # The failure must not discard the first batch's result.
    provider = FakeProvider(
        responses=[duplicate_response([])],
        errors=[None, LLMUnavailable("timed out")],
    )
    set_provider(provider)

    with pytest.raises(ClusteringIncomplete):
        find_near_duplicates_batched({0: "headache", 1: "nausea"}, tmp_path, batch_size=1)

    done = _load_cluster_progress(tmp_path)
    assert len(done) == 1


def test_find_near_duplicates_batched_resumes_only_the_failed_batch(tmp_path):
    provider = FakeProvider(
        responses=[duplicate_response([])],
        errors=[None, LLMUnavailable("timed out")],
    )
    set_provider(provider)
    with pytest.raises(ClusteringIncomplete):
        find_near_duplicates_batched({0: "headache", 1: "nausea"}, tmp_path, batch_size=1)

    provider2 = FakeProvider(responses=[duplicate_response([])])
    set_provider(provider2)

    groups = find_near_duplicates_batched({0: "headache", 1: "nausea"}, tmp_path, batch_size=1)

    assert len(provider2.calls) == 1  # only the previously-failed batch reran
    assert groups == []


# --- main(): classification checkpoint resume ------------------------------------------


def test_main_resumes_from_classification_checkpoint_without_reclassifying(tmp_path):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["fever", "rale"])
    out_dir = tmp_path / "out"

    save_classifications(
        ["fever", "rale"],
        {
            0: ClassifiedTerm(index=0, bucket=Bucket.noise, reason="r"),
            1: ClassifiedTerm(index=1, bucket=Bucket.noise, reason="r"),
        },
        out_dir,
    )

    provider = FakeProvider()  # any call would raise IndexError: nothing queued
    set_provider(provider)

    exit_code = main(["--input", str(csv_path), "--out-dir", str(out_dir), "--yes"])

    assert exit_code == 0
    assert provider.calls == []
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["counts"]["noise"] == 2


def test_main_reclassifies_when_checkpoint_candidates_mismatch(tmp_path):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["fever", "rale"])
    out_dir = tmp_path / "out"

    save_classifications(
        ["something", "else"],
        {
            0: ClassifiedTerm(index=0, bucket=Bucket.noise, reason="r"),
            1: ClassifiedTerm(index=1, bucket=Bucket.noise, reason="r"),
        },
        out_dir,
    )

    provider = FakeProvider(
        responses=[classification_response([(0, "noise", "r"), (1, "noise", "r")])]
    )
    set_provider(provider)

    exit_code = main(["--input", str(csv_path), "--out-dir", str(out_dir), "--yes"])

    assert exit_code == 0
    assert len(provider.calls) == 1


# --- main(): a clustering failure must not discard classification work -----------------


def test_main_still_writes_reports_and_exits_nonzero_when_clustering_fails(tmp_path, capsys):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["headache", "nausea"])
    out_dir = tmp_path / "out"

    provider = FakeProvider(
        responses=[
            classification_response([(0, "patient_reportable", "r"), (1, "patient_reportable", "r")]),
            duplicate_response([]),
        ],
        errors=[None, None, LLMUnavailable("timed out")],
    )
    set_provider(provider)

    exit_code = main(
        ["--input", str(csv_path), "--out-dir", str(out_dir), "--yes", "--cluster-batch-size", "1"]
    )

    assert exit_code == 1
    assert (out_dir / "classifications.json").exists()
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["counts"]["patient_reportable"] == 2
    assert "clustering incomplete" in capsys.readouterr().out


def test_main_rerun_resumes_only_the_unfinished_clustering_batch(tmp_path):
    csv_path = tmp_path / "kb.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["headache", "nausea"])
    out_dir = tmp_path / "out"

    provider = FakeProvider(
        responses=[
            classification_response([(0, "patient_reportable", "r"), (1, "patient_reportable", "r")]),
            duplicate_response([]),
        ],
        errors=[None, None, LLMUnavailable("timed out")],
    )
    set_provider(provider)
    first_exit = main(
        ["--input", str(csv_path), "--out-dir", str(out_dir), "--yes", "--cluster-batch-size", "1"]
    )
    assert first_exit == 1

    provider2 = FakeProvider(responses=[duplicate_response([])])
    set_provider(provider2)
    second_exit = main(
        ["--input", str(csv_path), "--out-dir", str(out_dir), "--yes", "--cluster-batch-size", "1"]
    )

    assert second_exit == 0
    # No classification call (checkpoint used) and only the one batch that
    # never completed the first time re-ran.
    assert len(provider2.calls) == 1
