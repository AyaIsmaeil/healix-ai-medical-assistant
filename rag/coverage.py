"""Coverage check: how many rag/knowledge_base/ symptom strings already
exist in vocabulary/symptoms.py, and which don't.

    python -m rag.coverage
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.schema import KNOWLEDGE_BASE_DIR, KnowledgeBaseEntry, load_all
from vocabulary.symptoms import CANONICAL_SYMPTOMS, is_canonical

MISSING_REPORT_PATH = KNOWLEDGE_BASE_DIR / "missing_from_vocabulary.md"


@dataclass(frozen=True)
class SymptomCoverage:
    # Unique symptom strings (exactly as written in the KB) that already
    # match the vocabulary, normalization-tolerant (vocabulary.is_canonical).
    matched: tuple[str, ...]
    # Unique symptom strings that don't -> every disease name that needed
    # each one, in the order first encountered.
    missing: dict[str, tuple[str, ...]]

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def missing_count(self) -> int:
        return len(self.missing)


def check_coverage(entries: list[KnowledgeBaseEntry] | None = None) -> SymptomCoverage:
    """Bucket every unique symptom string across `entries` as matched/missing.

    Counts unique strings, not per-entry occurrences — "تعب" needed by four
    diseases is one missing entry to add, not four.
    """
    entries = entries if entries is not None else load_all()

    matched: set[str] = set()
    missing: dict[str, list[str]] = {}
    for entry in entries:
        for symptom in entry.symptoms:
            if is_canonical(symptom):
                matched.add(symptom)
            else:
                missing.setdefault(symptom, []).append(entry.name)

    return SymptomCoverage(
        matched=tuple(sorted(matched)),
        missing={symptom: tuple(diseases) for symptom, diseases in sorted(missing.items())},
    )


def render_missing_markdown(coverage: SymptomCoverage) -> str:
    lines = [
        "# Symptoms referenced by rag/knowledge_base/ but absent from vocabulary/symptoms.py",
        "",
        f"{coverage.missing_count} unique symptom string(s) below. **Not added "
        "automatically** — see CLAUDE.md > Symptom vocabulary: a plausible but "
        "wrong wording silently fails to match later, which is worse than an "
        "absent one that fails loudly. For human review only.",
        "",
    ]
    for symptom, diseases in coverage.missing.items():
        lines.append(f"- **{symptom}** — needed by: {', '.join(diseases)}")
    return "\n".join(lines) + "\n"


def write_missing_report(coverage: SymptomCoverage, path=MISSING_REPORT_PATH) -> None:
    path.write_text(render_missing_markdown(coverage), encoding="utf-8")


if __name__ == "__main__":
    _entries = load_all()
    _coverage = check_coverage(_entries)
    write_missing_report(_coverage)
    # name_ar is a required, non-empty schema field (rag/schema.py) — a
    _missing_name_ar = [entry.name for entry in _entries if not entry.name_ar]
    print(f"KB entries loaded            : {len(_entries)}")
    print(f"Entries missing name_ar      : {len(_missing_name_ar)}")
    print(f"Symptoms matched vocabulary  : {_coverage.matched_count}")
    print(f"Symptoms missing (new)       : {_coverage.missing_count}")
    print(f"Current vocabulary total     : {len(CANONICAL_SYMPTOMS)}")
    print(f"Missing report written to    : {MISSING_REPORT_PATH}")
