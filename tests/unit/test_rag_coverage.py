from rag.coverage import check_coverage, render_missing_markdown
from rag.schema import KnowledgeBaseEntry


def _entry(name, symptoms):
    return KnowledgeBaseEntry(
        name=name,
        name_ar=f"{name} (ar)",
        symptoms=symptoms,
        specialties=["طب عام"],
        source="src",
        translation_reviewed=True,
    )


# vocabulary/symptoms.py's real CANONICAL_SYMPTOMS includes "حمى" and
# "غثيان" (see rules/red_flags.py's acs_chest_pain / bacterial_meningitis
# groups) — used below as known-present symptoms without needing to fake
# the vocabulary module itself, so this test exercises the real one.


def test_check_coverage_buckets_matched_and_missing():
    entries = [
        _entry("Disease A", ["حمى", "عرض غير موجود"]),
        _entry("Disease B", ["غثيان"]),
    ]

    coverage = check_coverage(entries)

    assert coverage.matched == ("حمى", "غثيان")
    assert coverage.matched_count == 2
    assert coverage.missing == {"عرض غير موجود": ("Disease A",)}
    assert coverage.missing_count == 1


def test_check_coverage_collapses_a_symptom_needed_by_multiple_diseases_to_one_entry():
    entries = [
        _entry("Disease A", ["عرض جديد"]),
        _entry("Disease B", ["عرض جديد"]),
        _entry("Disease C", ["عرض جديد"]),
    ]

    coverage = check_coverage(entries)

    assert coverage.missing_count == 1
    assert coverage.missing["عرض جديد"] == ("Disease A", "Disease B", "Disease C")


def test_check_coverage_empty_entries_list():
    coverage = check_coverage([])
    assert coverage.matched_count == 0
    assert coverage.missing_count == 0


def test_render_missing_markdown_lists_each_symptom_and_its_diseases():
    entries = [_entry("Disease A", ["عرض جديد"]), _entry("Disease B", ["عرض جديد"])]
    coverage = check_coverage(entries)

    markdown = render_missing_markdown(coverage)

    assert "عرض جديد" in markdown
    assert "Disease A" in markdown
    assert "Disease B" in markdown


def test_render_missing_markdown_with_nothing_missing():
    coverage = check_coverage([_entry("Disease A", ["حمى"])])

    markdown = render_missing_markdown(coverage)

    assert "0 unique symptom" in markdown


# --- against the real shipped knowledge base ------------------------------------


def test_coverage_against_the_real_knowledge_base_runs_without_error():
    coverage = check_coverage()
    assert coverage.matched_count + coverage.missing_count > 0
    # Every disease name attached to a missing symptom must be a real
    # entry name, not something malformed by the bucketing logic.
    from rag.schema import load_all

    real_names = {e.name for e in load_all()}
    for diseases in coverage.missing.values():
        assert set(diseases) <= real_names
