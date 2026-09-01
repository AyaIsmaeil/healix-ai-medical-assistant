"""Light Arabic stemming for retrieval (BM25 term matching only).

Implements the "light" affix-stripping stemmer from:

    Larkey, L. S., Ballesteros, L., & Connell, M. E. (2002).
    "Improving Stemming for Arabic Information Retrieval: Light
    Stemming and Co-occurrence Analysis." Proceedings of the 25th
    Annual International ACM SIGIR Conference on Research and
    Development in Information Retrieval (SIGIR'02), Tampere, Finland.
    https://ciir.cs.umass.edu/pubfiles/ir-249.pdf

That paper measured this exact class of stemmer (their best variant,
"light8", combined with stop-word removal) on the TREC-2001 Arabic
newswire corpus: +94.3% average precision over unstemmed retrieval, and
statistically significant (Wilcoxon, p<.05) improvements over lighter
variants and over normalization alone (their Table 2/Table 4). It does
NOT attempt root-finding or morphological analysis — it only strips a
small, fixed set of prefixes/suffixes chosen because they occur as real
affixes far more often than as the start/end of an affix-free word. The
paper found this simple approach outperformed a real morphological
analyzer (Khoja stemmer) on the same benchmark.

Applied here ONLY to `rag/health_education/retriever.py`'s BM25 term
matching (query and corpus tokenization) — never to `nodes/rag_retrieve.py`
or any other symptom-matching/triage path, and never to any text a
patient or doctor actually sees. A stemmed term is purely an internal
retrieval key.

Known, honest limitation (flagged rather than assumed away): this
algorithm was developed and measured on Modern Standard Arabic (MSA)
newswire text. This project's own corpus (rag/health_education/data/
ahd_cleaned.jsonl) contains real, colloquial Syrian-Arabic patient
questions, which the original paper never evaluated against. Before
this is trusted to help (not hurt) retrieval here, it must be measured
against this project's own corpus — see
scripts/evaluate_arabic_stemming.py.
"""

from __future__ import annotations

# Table 1, Larkey et al. (2002) — the "light8" prefix set (definite
# articles plus the "and" conjunction). Order matters: "و" is checked
# first with its own stricter length guard (paper section 4.2, step 1),
# since many real Arabic words begin with that letter without it being
# a conjunction prefix.
_WAW_PREFIX = "و"
_DEFINITE_ARTICLE_PREFIXES = ("بال", "كال", "فال", "وال", "ال")

# Table 1, Larkey et al. (2002) — the "light8" suffix set, applied
# right-to-left (longest-first ordering here achieves the same effect:
# a longer real suffix is stripped before a shorter substring of it
# could be mistaken for a different suffix).
_SUFFIXES: tuple[str, ...] = (
    "ية", "وا", "ون", "ين", "ات", "ان",
    "ها", "ه", "ة", "ي",
)

_MIN_REMAINING_AFTER_WAW = 3
_MIN_REMAINING_AFTER_PREFIX = 2
_MIN_REMAINING_AFTER_SUFFIX = 2


def light_stem(word: str) -> str:
    """Strip a small set of Arabic prefixes/suffixes from one
    already-normalized word (call rules.crisis.normalize() first — this
    function does not itself fold diacritics/alef variants/etc.).

    Never returns an empty string: if stripping would leave nothing
    usable, the pre-strip word (at that step) is kept instead.
    """
    if not word:
        return word

    stemmed = word

    if stemmed.startswith(_WAW_PREFIX) and len(stemmed) - 1 >= _MIN_REMAINING_AFTER_WAW:
        stemmed = stemmed[1:]

    for prefix in _DEFINITE_ARTICLE_PREFIXES:
        if stemmed.startswith(prefix) and len(stemmed) - len(prefix) >= _MIN_REMAINING_AFTER_PREFIX:
            stemmed = stemmed[len(prefix):]
            break

    for suffix in _SUFFIXES:
        if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= _MIN_REMAINING_AFTER_SUFFIX:
            stemmed = stemmed[: -len(suffix)]
            break

    return stemmed
