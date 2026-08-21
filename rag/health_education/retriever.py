"""BM25 retrieval over rag/health_education/data/ahd_cleaned.jsonl.

Lazy process-wide singleton (same pattern as ml/model_loader.py): the
real corpus is only loaded/indexed on first actual use, not at import
time, so importing this module never pays that cost for a caller that
never reaches the educational path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from rules.crisis import normalize

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_CORPUS_PATH = _DATA_DIR / "ahd_cleaned.jsonl"

# BM25 scores are unbounded and corpus-dependent, not a probability or a
# calibrated relevance measure. This floor is a reasonable starting point
# (a hit must share more than one weak/common token with the query), NOT
# an empirically validated threshold — same "Unjustified / requires
# empirical tuning" labeling this project already applies to untuned
# parameters elsewhere (research/ML_AUDIT.md). Revisit once there is a
# real query log to tune against.
MIN_SCORE = 1.0


@dataclass(frozen=True)
class RetrievedDoc:
    question: str
    answer: str
    category: str
    score: float


def _tokenize(text: str) -> list[str]:
    return normalize(text).split()


class HealthEducationRetriever:
    """One BM25 index over one corpus (a list of {question, answer,
    category, source} dicts — rag.health_education.preprocess's own
    output shape). Not itself a singleton — get_retriever() below owns
    the process-wide instance; tests build a small one directly over a
    synthetic corpus instead of the real ~800k-row file."""

    def __init__(self, records: list[dict]):
        self._records = records
        self._bm25 = BM25Okapi([_tokenize(r["question"]) for r in records]) if records else None

    @classmethod
    def from_jsonl(cls, path: Path) -> "HealthEducationRetriever":
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Ranked hits, best first. Only positive-score hits are returned —
        a zero/negative BM25 score means no real term overlap at all, not
        a weak-but-real match."""
        if self._bm25 is None or not query.strip():
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedDoc(
                question=self._records[i]["question"],
                answer=self._records[i]["answer"],
                category=self._records[i]["category"],
                score=float(scores[i]),
            )
            for i in ranked_indices
            if scores[i] > 0
        ]


_retriever: HealthEducationRetriever | None = None


def get_retriever(corpus_path: Path = _DEFAULT_CORPUS_PATH) -> HealthEducationRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HealthEducationRetriever.from_jsonl(corpus_path)
    return _retriever


def set_retriever(retriever: HealthEducationRetriever | None) -> None:
    """Override the process-wide retriever. For tests; production
    lazy-loads the real corpus via get_retriever()."""
    global _retriever
    _retriever = retriever
