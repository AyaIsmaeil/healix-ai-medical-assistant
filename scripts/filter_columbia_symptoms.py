"""Classify the Columbia disease-symptom KB's ~400 feature-column strings
into review buckets — patient_reportable, exam_finding, lab_result,
imaging_or_ecg, psychiatric_crisis, social_or_status, noise.

*** This script writes REVIEW REPORTS ONLY. It never touches
vocabulary/symptoms.py. *** Every classified entry is a bucket
*suggestion* for a human to confirm or correct — nothing here is
auto-adopted into the canonical vocabulary. That is a separate, deliberate
step, and a plausible-looking auto-adoption would repeat exactly the
mistake vocabulary/symptoms.py's "do not invent entries" rule exists to
prevent, just one layer removed (an LLM inventing bucket membership
instead of a person inventing wording).

Input is a CSV whose HEADER ROW is the candidate list — this is how the
source KB encodes its ~400 feature columns (see e.g.
main_cleaned_fix.csv). The trailing "prognosis" column is not a real
symptom feature (it holds the diagnosis label), but it is deliberately
NOT filtered out here: it is a candidate like any other, and the
classifier is expected to bucket it as noise. Silently special-casing it
would be exactly the kind of "quietly decided for you" behavior this
script exists to avoid — every one of the ~400 strings must appear in
the report, full stop.

*** MAKES REAL, BILLABLE LLM CALLS by default. *** ~400 entries batched
means roughly a dozen classification calls, plus a variable number of
near-duplicate clustering calls over the patient_reportable subset — the
clustering step is itself batched (see "clustering batching" below), so
its call count depends on how many patient_reportable terms come back
from classification. All calls run at tier="quality" given the safety
relevance of the psychiatric_crisis bucket. Run deliberately:

    python scripts/filter_columbia_symptoms.py --input <path-to-csv> [--yes]

Use --limit for a cheap dry run against a handful of entries first.

*** Resumable. *** Classification results are written to
<out-dir>/classifications.json the moment classify_all() finishes —
before clustering starts. Clustering progress is written to
<out-dir>/duplicate_groups.json one batch at a time as each batch
finishes. Re-running the same --input/--out-dir loads whatever
checkpoints match the current candidate list and skips exactly the work
already done: a slow/failed clustering batch never costs a re-run of
classification, and a slow/failed later clustering batch never costs a
re-run of the batches before it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from llm_client import LLMConfigError, LLMError, call_llm  # noqa: E402

load_dotenv()

# Smaller than it needs to be for a well-behaved response, deliberately:
# an LLM is more likely to drop or merge entries in a very long list, and
# the bisection retry in classify_batch() means a smaller default costs
# little — most batches succeed first try.
DEFAULT_BATCH_SIZE = 30

# Below this, a batch that still fails index-completeness is not retried
# further — it is reported as a hard failure naming the exact term,
# rather than silently splitting down to nothing.
_MIN_BATCH_SIZE = 1

# Cap on how many patient_reportable terms go into a single clustering
# prompt. Smaller than DEFAULT_BATCH_SIZE isn't required by anything
# structural — clustering prompts are chosen deliberately smaller because
# the duplicate-finding task asks the model to reason about relationships
# *between* every pair of terms in the prompt (O(n^2) attention on the
# concept, not just an O(n) per-term label), which is exactly the shape
# that overwhelmed a local 4GB model at n=229 in one call.
DEFAULT_CLUSTER_BATCH_SIZE = 25


class Bucket(str, Enum):
    patient_reportable = "patient_reportable"
    exam_finding = "exam_finding"
    lab_result = "lab_result"
    imaging_or_ecg = "imaging_or_ecg"
    psychiatric_crisis = "psychiatric_crisis"
    social_or_status = "social_or_status"
    noise = "noise"


# --- LLM output schemas -----------------------------------------------------
#
# Tooling-local, not in schemas/: these validate a one-off classification
# call, not a graph node's output, and don't belong in the production
# symptom-extraction contract.


class ClassifiedTerm(BaseModel):
    index: int = Field(description="The candidate's index, exactly as given in the prompt.")
    bucket: Bucket
    reason: str = Field(description="One short clause justifying the bucket, for human review.")


class ClassificationBatch(BaseModel):
    classifications: list[ClassifiedTerm]


class DuplicateGroup(BaseModel):
    canonical_suggestion: str = Field(
        description="Suggested plain-language canonical phrasing. A hint for the "
        "reviewer only — never adopted automatically."
    )
    member_indices: list[int] = Field(
        description="Original candidate indices (from the full numbered list, not "
        "re-numbered) that describe the same patient-reportable concept."
    )


class DuplicateClustering(BaseModel):
    groups: list[DuplicateGroup] = Field(default_factory=list)


class ClassificationError(RuntimeError):
    """A batch could not be classified completely, even after bisection."""


# --- candidate extraction ---------------------------------------------------


def extract_candidate_terms(csv_path: Path) -> list[str]:
    """Read the header row of the Columbia CSV as the candidate term list."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    if not header:
        raise ValueError(f"{csv_path}: header row is empty.")
    return header


def chunk(items: list, size: int) -> list[list]:
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


# --- classification ----------------------------------------------------------


def _build_classification_prompt(batch: list[tuple[int, str]]) -> str:
    bucket_list = "\n".join(f"  - {b.value}" for b in Bucket)
    numbered = "\n".join(f"{index}: {term}" for index, term in batch)
    return f"""You are classifying medical vocabulary terms drawn from a disease-symptom
knowledge base, for a downstream system that only accepts symptoms a
PATIENT could state unaided, in their own words.

Assign each numbered term to exactly one of these buckets:
{bucket_list}

Bucket definitions:
  - patient_reportable: a patient could describe this unaided in plain
    language (e.g. headache, fever, nausea, cough, chest pain).
  - exam_finding: requires a clinician's physical examination or a bedside
    instrument to detect (e.g. rale, egophony, Murphy's sign, jugular
    venous distention). A patient cannot report this themselves.
  - lab_result: requires a laboratory test (e.g. hyponatremia,
    transaminitis, hematocrit decreased).
  - imaging_or_ecg: requires imaging or an ECG/EEG tracing to detect (e.g.
    cardiomegaly, ST segment elevation, lung nodule).
  - psychiatric_crisis: indicates active suicidal or homicidal ideation,
    intent, or comparable acute crisis (e.g. suicidal, feeling suicidal,
    homicidal thoughts, feeling hopeless). This bucket is safety-critical:
    when genuinely uncertain between this and another bucket, prefer this
    one — a missed crisis term is far worse than an over-cautious one.
  - social_or_status: describes a demographic, social, or status fact
    rather than a clinical symptom (e.g. homelessness, nonsmoker,
    transsexual).
  - noise: not a reportable clinical concept at all — a dataset artifact,
    a meta-field, or too vague/abstract to mean anything on its own (e.g.
    prognosis, asymptomatic, difficulty, no status change).

Terms to classify (index: term):
{numbered}

Return one classification per term, using its exact given index. Do not
skip any index and do not invent indices that were not given."""


def _validate_batch_response(
    batch: list[tuple[int, str]], response: ClassificationBatch
) -> None:
    expected = {index for index, _ in batch}
    returned = {c.index for c in response.classifications}
    if returned != expected:
        missing = expected - returned
        extra = returned - expected
        raise ClassificationError(
            f"index mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    if len(response.classifications) != len(batch):
        raise ClassificationError(
            f"duplicate indices in response for batch of {len(batch)}"
        )


def classify_batch(batch: list[tuple[int, str]]) -> list[ClassifiedTerm]:
    """Classify one batch, bisecting and retrying on any incomplete response.

    A response is accepted only if it names every index in the batch
    exactly once — a batch that comes back short, padded, or malformed is
    never partially trusted. Bisecting isolates which specific term is
    causing trouble rather than discarding the whole batch's classifications
    every time. At size 1, a persistent failure is a real error naming the
    exact term, never a silent drop.

    LLMConfigError is the exception to all of that: it means required
    configuration (API key, model name, a provider/schema combination the
    provider refuses) is missing or unusable. That cannot change between
    attempts or between batch sizes, so bisecting it would just repeat the
    identical failure once per sub-batch instead of surfacing the one clear
    error immediately. It propagates unbisected and unretried.
    """
    prompt = _build_classification_prompt(batch)
    term_label = batch[0][1] if len(batch) == 1 else f"batch of {len(batch)}"

    try:
        result = call_llm(
            prompt,
            schema=ClassificationBatch,
            tier="quality",
            prompt_name="filter_columbia_symptoms:classify",
        )
        assert isinstance(result, ClassificationBatch)
        _validate_batch_response(batch, result)
        return result.classifications
    except LLMConfigError:
        raise
    except (LLMError, ClassificationError) as exc:
        if len(batch) <= _MIN_BATCH_SIZE:
            raise ClassificationError(
                f"could not classify {term_label!r} after retry: {exc}"
            ) from exc
        mid = len(batch) // 2
        return classify_batch(batch[:mid]) + classify_batch(batch[mid:])


def classify_all(
    candidates: list[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[int, ClassifiedTerm]:
    """Classify every candidate. Raises on any term that never resolves.

    Returns index -> ClassifiedTerm for all len(candidates) entries — a
    result missing even one index is a bug in this function, not something
    calling code should have to guard against.
    """
    indexed = list(enumerate(candidates))
    results: dict[int, ClassifiedTerm] = {}
    for batch in chunk(indexed, batch_size):
        for classified in classify_batch(batch):
            results[classified.index] = classified

    missing = set(range(len(candidates))) - set(results)
    if missing:
        raise ClassificationError(f"never classified: {sorted(missing)}")
    return results


# --- classification checkpointing ---------------------------------------------
#
# Written the moment classify_all() returns, before the clustering call that
# timed out in production and took the (unpersisted) classification results
# down with it. A checkpoint is only ever trusted for the *exact* candidate
# list it was written for — a different --input or --limit produces a
# different candidate list, and silently reusing a stale checkpoint against
# it would attribute someone else's classifications to these terms. Mismatch
# is not an error, just a cache miss: classify_all() runs fresh.

_CLASSIFICATION_CHECKPOINT_NAME = "classifications.json"


def _classification_checkpoint_path(out_dir: Path) -> Path:
    return out_dir / _CLASSIFICATION_CHECKPOINT_NAME


def save_classifications(
    candidates: list[str], classifications: dict[int, ClassifiedTerm], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidates": candidates,
        "classifications": {
            str(index): {"bucket": c.bucket.value, "reason": c.reason}
            for index, c in sorted(classifications.items())
        },
    }
    _classification_checkpoint_path(out_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_classifications(
    candidates: list[str], out_dir: Path
) -> dict[int, ClassifiedTerm] | None:
    """Load a checkpoint if one exists and its candidate list matches exactly.

    Returns None on a missing file, a corrupt file, or a candidate-list
    mismatch — every case means "nothing usable to resume from", not an
    error, so classify_all() can run unconditionally in all of them.
    """
    path = _classification_checkpoint_path(out_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("candidates") != candidates:
        return None
    return {
        int(index): ClassifiedTerm(index=int(index), bucket=Bucket(v["bucket"]), reason=v["reason"])
        for index, v in payload["classifications"].items()
    }


# --- near-duplicate clustering ------------------------------------------------


def _build_duplicate_prompt(reportable: list[tuple[int, str]]) -> str:
    numbered = "\n".join(f"{index}: {term}" for index, term in reportable)
    return f"""Below is a numbered list of patient-reportable symptom phrases, drawn
from a disease-symptom knowledge base. The index numbers are each
phrase's ORIGINAL catalog index — they are not sequential, and gaps
between them are expected and meaningless.

Several of these phrases describe the same underlying reportable concept
in different wording (e.g. "pain chest", "chest discomfort", and "chest
tightness" all describe chest pain/discomfort; "sweat", "sweating
increased", and "hyperhidrosis disorder" all describe sweating). These
need to collapse to one canonical entry in a downstream vocabulary.

Find every such group of 2 or more phrases. For each group, suggest a
short canonical phrasing (a hint for a human reviewer — not a final
decision) and list the original indices of every member. Do not include
a phrase in a group by itself, and do not force phrases together that
describe genuinely different concepts merely because they share a word.

Phrases:
{numbered}"""


def find_near_duplicates(reportable: dict[int, str]) -> list[DuplicateGroup]:
    """One clustering call over the patient_reportable subset.

    Returns [] if there is nothing to cluster (including an empty input) —
    that is a valid, uninteresting result, not an error.
    """
    if not reportable:
        return []

    items = sorted(reportable.items())
    prompt = _build_duplicate_prompt(items)
    result = call_llm(
        prompt,
        schema=DuplicateClustering,
        tier="quality",
        prompt_name="filter_columbia_symptoms:duplicates",
    )
    assert isinstance(result, DuplicateClustering)

    valid_indices = set(reportable)
    groups = []
    for group in result.groups:
        members = [i for i in group.member_indices if i in valid_indices]
        # A hallucinated index outside the reportable set, or a group that
        # collapsed to <2 real members after filtering, is dropped rather
        # than reported as a duplicate pairing that does not actually hold.
        if len(members) >= 2:
            groups.append(DuplicateGroup(canonical_suggestion=group.canonical_suggestion, member_indices=members))
    return groups


# --- clustering batching (blocking) --------------------------------------------
#
# find_near_duplicates() above sent every patient_reportable term to the model
# in one prompt. At n=229 that is what timed out: the clustering task asks the
# model to compare every term against every other term, and that reasoning
# load — not just prompt length — is what a local 4GB model can't do in one
# shot within any reasonable budget.
#
# Naive fixed-size chunking (reuse chunk() from above) would "fix" the
# timeout and break correctness at the same time: two near-duplicate phrases
# placed in different chunks are never shown to the model together, so a
# real duplicate pair is silently lost with no error to notice it by — worse
# than the timeout it would replace.
#
# APPROACH CHOSEN: token-overlap blocking (a standard record-linkage
# technique) via union-find over an inverted index, with one refinement.
# Each patient_reportable phrase is normalized (lowercased, non-alphanumeric
# split, English stopwords dropped) into a token set; any two phrases
# sharing at least one *token or token-prefix* are unioned into the same
# block. The prefix half (first _PREFIX_LEN chars of any longer token) is a
# crude stemmer — cheap and deterministic, no morphology library — that
# exists specifically so "sweat" and "sweating increased" block together
# (both produce the key "sweat"), not just exact-token matches like "pain
# chest" / "chest discomfort" (share "chest" outright). Every clustering
# prompt is then built from whole blocks only, never a fraction of one, so
# two phrases judged related can never be separated into different
# clustering calls. Blocks are greedily packed up to
# DEFAULT_CLUSTER_BATCH_SIZE to keep the call count down.
#
# LIMITATIONS (report these, don't hide them):
#   1. Lexical (+ crude prefix stemming) only, not semantic. This catches
#      "pain chest" / "chest discomfort" (share "chest") and "sweat" /
#      "sweating increased" (share prefix "sweat"), but NOT a synonym pair
#      with zero lexical overlap even after stemming, e.g. "sweat" and
#      "hyperhidrosis disorder" — genuinely different words for the same
#      concept. Such a pair lands in two different blocks and, unless those
#      blocks happen to be packed into the same batch by coincidence, the
#      model never sees them together and cannot group them. This is a real
#      recall gap versus one single-context call — the deliberate trade for
#      a batch small enough to actually finish on a local model.
#   2. A block larger than DEFAULT_CLUSTER_BATCH_SIZE (many phrases sharing
#      one very common token) is itself split across multiple calls via
#      plain chunk() — the one case where even two phrases sharing a token
#      can still end up in different batches. Rare for this vocabulary
#      (tokens are specific clinical words, not e.g. "pain" alone once
#      stopwords and short tokens are removed), but not impossible.
#   3. Blocking is recomputed from `reportable` on every run. Given the same
#      candidate list and the same DEFAULT_CLUSTER_BATCH_SIZE it is fully
#      deterministic (stable sort, sorted input order), which is what makes
#      the per-batch checkpoint below safe to resume against — but changing
#      --cluster-batch-size between runs invalidates prior clustering
#      checkpoints' batch shapes and forces those batches to be redone.

_CLUSTER_CHECKPOINT_NAME = "duplicate_groups.json"

_BLOCKING_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with", "without", "for", "not"}
)

# Crude stemming for the blocking signal only (never shown to the model,
# never adopted as a canonical name) — see the "clustering batching" module
# note above for exactly what this does and doesn't catch.
_PREFIX_LEN = 5


def _blocking_keys(term: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", term.lower())
    tokens = {w for w in words if len(w) > 1 and w not in _BLOCKING_STOPWORDS}
    keys = set(tokens)
    keys.update(token[:_PREFIX_LEN] for token in tokens if len(token) > _PREFIX_LEN)
    return keys


def block_by_shared_tokens(reportable: dict[int, str]) -> list[list[int]]:
    """Group candidate indices so any two sharing a normalized token (or
    token prefix — see _blocking_keys) land in the same block, via
    union-find over an inverted index.

    See the "clustering batching" module note above for why this exists and
    what it does and doesn't catch. Order is deterministic for a given
    `reportable` (iterated in sorted-index order), which pack_blocks() and
    the clustering checkpoint below both depend on.
    """
    parent = {index: index for index in reportable}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    token_index: dict[str, list[int]] = {}
    for index, term in sorted(reportable.items()):
        for key in _blocking_keys(term):
            token_index.setdefault(key, []).append(index)

    for indices in token_index.values():
        for other in indices[1:]:
            union(indices[0], other)

    blocks: dict[int, list[int]] = {}
    for index in sorted(reportable):
        blocks.setdefault(find(index), []).append(index)
    return list(blocks.values())


def pack_blocks(blocks: list[list[int]], batch_size: int) -> list[list[int]]:
    """Greedily pack blocks into batches of at most `batch_size` indices.

    A block is split (via chunk()) only when it alone exceeds batch_size —
    see limitation #2 in the module note above. Otherwise every batch is a
    union of whole blocks, largest-first, so batch count stays low without
    ever separating two indices that block_by_shared_tokens judged related.
    """
    whole: list[list[int]] = []
    oversized_split: list[list[int]] = []
    for block in blocks:
        if len(block) > batch_size:
            oversized_split.extend(chunk(block, batch_size))
        else:
            whole.append(block)

    whole.sort(key=len, reverse=True)
    batches: list[list[int]] = []
    for block in whole:
        for existing in batches:
            if len(existing) + len(block) <= batch_size:
                existing.extend(block)
                break
        else:
            batches.append(list(block))
    batches.extend(oversized_split)
    return batches


class ClusteringIncomplete(RuntimeError):
    """Clustering stopped partway through. Carries whatever groups the
    batches completed before the failure already found, so a caller can
    still write a report instead of discarding them along with the error.
    """

    def __init__(self, message: str, groups: list[DuplicateGroup]) -> None:
        super().__init__(message)
        self.groups = groups


def _cluster_checkpoint_path(out_dir: Path) -> Path:
    return out_dir / _CLUSTER_CHECKPOINT_NAME


def _save_cluster_progress(done: dict[tuple[int, ...], list[DuplicateGroup]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batches": [
            {"indices": list(key), "groups": [g.model_dump() for g in groups]}
            for key, groups in done.items()
        ]
    }
    _cluster_checkpoint_path(out_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_cluster_progress(out_dir: Path) -> dict[tuple[int, ...], list[DuplicateGroup]]:
    path = _cluster_checkpoint_path(out_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    done: dict[tuple[int, ...], list[DuplicateGroup]] = {}
    for entry in payload.get("batches", []):
        key = tuple(entry["indices"])
        done[key] = [DuplicateGroup(**g) for g in entry["groups"]]
    return done


def find_near_duplicates_batched(
    reportable: dict[int, str],
    out_dir: Path,
    batch_size: int = DEFAULT_CLUSTER_BATCH_SIZE,
) -> list[DuplicateGroup]:
    """Cluster the patient_reportable subset in batches, checkpointing after
    each one so a failure partway through never loses the batches that
    already succeeded — the same guarantee save_classifications() gives the
    classification phase, applied one level down.

    Raises ClusteringIncomplete (carrying every group found so far) on the
    first batch that raises LLMError, rather than losing that progress to
    an unhandled exception. Batches already in the checkpoint are skipped
    outright — re-running after a partial failure resumes, it does not
    restart.
    """
    if not reportable:
        return []

    batches = pack_blocks(block_by_shared_tokens(reportable), batch_size)
    done = _load_cluster_progress(out_dir)

    all_groups: list[DuplicateGroup] = []
    for indices in batches:
        key = tuple(sorted(indices))
        if key in done:
            all_groups.extend(done[key])
            continue

        subset = {i: reportable[i] for i in indices}
        try:
            groups = find_near_duplicates(subset)
        except LLMError as exc:
            raise ClusteringIncomplete(
                f"clustering stopped after {len(done)}/{len(batches)} batches: {exc}",
                groups=all_groups,
            ) from exc

        done[key] = groups
        _save_cluster_progress(done, out_dir)
        all_groups.extend(groups)

    return all_groups


# --- report writing -----------------------------------------------------------

_QUARANTINE_HEADER = """# psychiatric_crisis — QUARANTINED, not part of the symptom vocabulary
#
# These terms are excluded from vocabulary/symptoms.py entirely and never
# will be added there. They indicate acute suicidal/homicidal ideation or
# intent, which this service routes through the crisis path (rules/crisis.py,
# CLAUDE.md > Non-negotiable safety rules #9) — not through symptom
# extraction or RAG. A term like this reaching extract_symptoms/RAG would
# mean an acute crisis got treated as a routine triage input instead of
# stopping analysis and surfacing the crisis marker.
#
# Kept here, separately, so a mental-health-qualified reviewer can decide
# which (if any) inform rules/crisis.py's own pattern list — see the
# PLACEHOLDER DATA warning in that file. This list is a candidate source
# for that review, not itself reviewed or approved content.
#
"""


@dataclass
class ClassificationReport:
    candidates: list[str]
    classifications: dict[int, ClassifiedTerm]
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)

    def by_bucket(self) -> dict[Bucket, list[int]]:
        buckets: dict[Bucket, list[int]] = {b: [] for b in Bucket}
        for index in sorted(self.classifications):
            buckets[self.classifications[index].bucket].append(index)
        return buckets

    def counts(self) -> Counter:
        return Counter(c.bucket.value for c in self.classifications.values())


def _duplicate_group_for(index: int, groups: list[DuplicateGroup]) -> DuplicateGroup | None:
    return next((g for g in groups if index in g.member_indices), None)


def write_reports(report: ClassificationReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_bucket = report.by_bucket()

    for bucket, indices in by_bucket.items():
        lines = []
        if bucket is Bucket.psychiatric_crisis:
            lines.append(_QUARANTINE_HEADER)
        lines.append(f"# {bucket.value} ({len(indices)} entries)\n")

        for index in indices:
            term = report.candidates[index]
            reason = report.classifications[index].reason
            line = f"- [{index}] {term} — {reason}"
            if bucket is Bucket.patient_reportable:
                group = _duplicate_group_for(index, report.duplicate_groups)
                if group is not None:
                    line += f"  [near-duplicate: \"{group.canonical_suggestion}\"]"
            lines.append(line)

        (out_dir / f"{bucket.value}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    _write_duplicate_report(report, out_dir)
    _write_summary(report, out_dir)


def _write_duplicate_report(report: ClassificationReport, out_dir: Path) -> None:
    lines = [
        f"# Near-duplicate candidate groups within patient_reportable "
        f"({len(report.duplicate_groups)} groups)\n"
    ]
    for n, group in enumerate(report.duplicate_groups, start=1):
        lines.append(f"## Group {n} — suggested canonical: \"{group.canonical_suggestion}\"")
        for index in group.member_indices:
            lines.append(f"- [{index}] {report.candidates[index]}")
        lines.append("")

    (out_dir / "near_duplicates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(report: ClassificationReport, out_dir: Path) -> None:
    counts = report.counts()
    total = len(report.classifications)

    lines = [f"# Classification summary ({total} total candidates)\n"]
    for bucket in Bucket:
        lines.append(f"- {bucket.value}: {counts.get(bucket.value, 0)}")
    lines.append(f"\nNear-duplicate groups within patient_reportable: {len(report.duplicate_groups)}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "total": total,
        "counts": {b.value: counts.get(b.value, 0) for b in Bucket},
        "duplicate_group_count": len(report.duplicate_groups),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Columbia KB CSV path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/columbia_symptom_classification"),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--cluster-batch-size",
        type=int,
        default=DEFAULT_CLUSTER_BATCH_SIZE,
        help="Max patient_reportable terms per near-duplicate clustering call",
    )
    parser.add_argument("--limit", type=int, default=None, help="Classify only the first N candidates (for a cheap dry run)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the cost confirmation prompt")
    args = parser.parse_args(argv)

    candidates = extract_candidate_terms(args.input)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    checkpointed = load_classifications(candidates, args.out_dir)
    n_batches = len(chunk(candidates, args.batch_size))
    print(f"Candidates to classify : {len(candidates)}")
    if checkpointed is not None:
        print(f"Classification calls   : 0 (resuming from checkpoint at {args.out_dir}/{_CLASSIFICATION_CHECKPOINT_NAME})")
    else:
        print(f"Classification calls   : ~{n_batches} (batch size {args.batch_size}, tier=quality)")
    print(
        "Plus a variable number of near-duplicate clustering calls over the "
        f"patient_reportable subset, batched at up to {args.cluster_batch_size} terms each."
    )
    print()

    if not args.yes:
        print("This makes REAL, BILLABLE Gemini API calls.")
        if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Aborted. No API calls made.")
            return 0
        print()

    if checkpointed is not None:
        print(f"Loaded {len(checkpointed)} classifications from checkpoint — skipping re-classification.")
        classifications = checkpointed
    else:
        print("Classifying...")
        classifications = classify_all(candidates, batch_size=args.batch_size)
        save_classifications(candidates, classifications, args.out_dir)
        print(f"Classification checkpoint written to {args.out_dir}/{_CLASSIFICATION_CHECKPOINT_NAME}")

    reportable = {
        index: candidates[index]
        for index in sorted(classifications)
        if classifications[index].bucket is Bucket.patient_reportable
    }
    print(f"Clustering near-duplicates among {len(reportable)} patient_reportable entries...")

    clustering_complete = True
    try:
        duplicate_groups = find_near_duplicates_batched(
            reportable, args.out_dir, batch_size=args.cluster_batch_size
        )
    except ClusteringIncomplete as exc:
        clustering_complete = False
        duplicate_groups = exc.groups
        print(f"\nWARNING: {exc}")
        print(
            "Classification results are unaffected (already checkpointed above). "
            "Completed clustering batches were checkpointed too — re-run the same "
            "command to resume clustering from where it stopped."
        )

    report = ClassificationReport(
        candidates=candidates, classifications=classifications, duplicate_groups=duplicate_groups
    )
    write_reports(report, args.out_dir)

    print()
    print(f"Reports written to {args.out_dir}/")
    print()
    counts = report.counts()
    for bucket in Bucket:
        print(f"  {bucket.value:20} {counts.get(bucket.value, 0)}")
    print(f"  {'total':20} {len(classifications)}")
    print(f"\n  near-duplicate groups: {len(duplicate_groups)}")
    if not clustering_complete:
        print("  (clustering incomplete — see warning above)")

    return 0 if clustering_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
