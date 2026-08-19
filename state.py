"""HealixState: the LangGraph state shared across all nodes.

See CLAUDE.md > State and > Graph flow for the source of truth on what
each field means and how it moves through the graph.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from audit.logger import log_malformed_output

# dict[str, Any], not dict[str, str]: every field is a string EXCEPT
# duration_days (int | None — nodes.extract_symptoms._with_duration_days,
# vocabulary.duration.parse_duration_days), same "one non-string field
# breaks the narrower alias" reasoning as CandidateDisease below.
Symptom = dict[str, Any]

# One red-flag hit — {"id": rule_id-or-"llm", "reason": reason_ar / the
# LLM's own reasoning}. A dict pairing the two fields at construction,
# never two same-length lists a caller has to zip — see
# nodes.check_red_flags.check_red_flags, the only place these are built.
RedFlag = dict[str, str]

# One rag_retrieve match — {"name": ..., "name_ar": ..., "match_score": ...,
# "matched_symptoms": [...], "missing_symptoms": [...], "negated_symptoms":
# [...], "specialties": [...]}. dict[str, Any], not dict[str, str] like
# Symptom/RedFlag above: match_score is a float and the rest are lists, not
# strings. specialties and name_ar are both carried straight from the KB
# entry (rag.schema.KnowledgeBaseEntry) — see nodes.rag_retrieve.rag_retrieve,
# the only place these are built, and nodes.diagnose.diagnose, which carries
# both into diagnosis["differential"] entries for nodes.route_specialty
# (specialties) and nodes.generate_reports (name_ar — the patient register's
# only source of a disease name; name itself is English/Latin, doctor-register
# only, see rag/schema.py's own field description).
CandidateDisease = dict[str, Any]


def _valid_name(symptom: Symptom) -> str | None:
    """Return symptom["name"], or None (and audit-log it) if it's missing/empty.

    LLM extraction output is not schema-validated before it reaches this
    reducer (schemas/ validates the raw LLM call, not what a graph merge
    step receives) — a malformed entry here must not raise mid-conversation.
    """
    name = symptom.get("name")
    if name in (None, ""):
        reason = "missing_name" if "name" not in symptom else "empty_name"
        log_malformed_output(node="merge_symptoms", reason=reason, payload=dict(symptom))
        return None
    return name


def merge_symptoms(existing: list[Symptom] | None, new: list[Symptom] | None) -> list[Symptom]:
    """Accumulating reducer for `symptoms` / `negated_symptoms`.

    Symptoms persist and merge across turns instead of being overwritten
    each time extract_symptoms runs (CLAUDE.md > State), deduplicated by
    the "name" key. Merging a symptom already present fills in whatever
    fields the new entry has rather than replacing it outright — a sparse
    later mention (e.g. just a name) never erases detail such as duration
    or severity gathered on an earlier turn. None/"" values on the new
    side are skipped for that reason. Entries missing a usable "name" are
    skipped and audit-logged rather than failing the turn.
    """
    merged: dict[str, Symptom] = {}
    order: list[str] = []

    for symptom in existing or []:
        name = _valid_name(symptom)
        if name is None:
            continue
        order.append(name)
        merged[name] = dict(symptom)

    for symptom in new or []:
        name = _valid_name(symptom)
        if name is None:
            continue
        if name not in merged:
            order.append(name)
            merged[name] = dict(symptom)
            continue
        current = merged[name]
        for key, value in symptom.items():
            if value not in (None, ""):
                current[key] = value

    return [merged[name] for name in order]


def merge_unmatched_mentions(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Accumulating reducer for `unmatched_mentions` — list, deduped by exact string.

    Unlike merge_symptoms, entries here are plain phrases, not records with
    optional sub-fields to fill in — so there is nothing to merge field by
    field, and exact-string dedup (no normalization, no fuzzy matching) is
    enough, preserving first-seen order across turns. See
    schemas.symptoms.SymptomExtraction.unmatched_mentions for what produces
    these: a symptom-like phrase the patient used that extract_symptoms
    recognized but could not map to vocabulary/symptoms.py, and refused to
    force a near-match name onto rather than drop.
    """
    seen: dict[str, None] = {}
    for mention in existing or []:
        seen.setdefault(mention, None)
    for mention in new or []:
        seen.setdefault(mention, None)
    return list(seen)


Severity = Literal["low", "moderate", "high", "emergency"]

# Structured, Laravel-sourced only — never inferred from conversation
# text. See the patient_sex field comment below and
# nodes/rag_retrieve.py's module docstring ("Sex-specific gating") for
# why conversational inference was deliberately not built.
PatientSex = Literal["male", "female"]

# Which terminal node produced the current/most recent turn's response —
# see CLAUDE.md > Graph flow and api.contracts.ChatResponse.stage, which
# imports this same alias rather than redefining it (the two must never
# drift: this is the value a terminal node sets in state, that field is
# how it reaches Laravel).
Stage = Literal["followup", "crisis", "emergency", "diagnosis"]

# Which terminal SAFETY outcome this thread has already reached, if any —
# see the thread_outcome field comment below and nodes/reiterate_terminal_outcome.py's
# module docstring for the full reasoning (CLAUDE.md > Non-negotiable
# safety rule 13). Deliberately NOT the same type as Severity: a thread
# reaching "diagnosis_complete" was explicitly considered and rejected as
# a value here (see the reiterate_terminal_outcome module docstring) —
# this field is scoped to the two SAFETY-terminal outcomes only.
ThreadOutcome = Literal["crisis", "emergency"]


class HealixState(TypedDict):
    # Set on first turn only, by load_record.
    thread_id: str
    medical_record_summary: str  # filtered summary from Laravel, never a raw record dump
    # Structured, from api.contracts.ChatRequest.patient_sex — Laravel's
    # own account data, not inferred from conversation text (see
    # PatientSex's own comment above). Optional: Laravel may not always
    # send it, same as medical_record_summary. Consumed by
    # nodes.rag_retrieve to gate sex-specific rag/knowledge_base/ entries
    # (rag.schema.KnowledgeBaseEntry.applicable_sex) — see that node's
    # module docstring for the full gating/exclusion/follow-up design.
    patient_sex: PatientSex | None

    # Each entry conforms to api.contracts.Message ({"role": ..., "content":
    # ...}) — kept as a plain dict here, not that Pydantic model, for the
    # same reason Symptom below is: a reducer must tolerate a malformed
    # entry without raising mid-conversation (see merge_symptoms).
    messages: Annotated[list[dict[str, str]], operator.add]

    symptoms: Annotated[list[Symptom], merge_symptoms]
    # Carries diagnostic weight equal to confirmed symptoms — not a lesser signal.
    negated_symptoms: Annotated[list[Symptom], merge_symptoms]
    # Symptom-like phrases extract_symptoms recognized but could not map to
    # the vocabulary — never silently dropped just because they don't have
    # a canonical name yet. Must be surfaced in the doctor report (not
    # built yet) so nothing the patient said gets lost between here and
    # there. See schemas.symptoms.SymptomExtraction.unmatched_mentions.
    unmatched_mentions: Annotated[list[str], merge_unmatched_mentions]

    # How many follow-up questions have been asked so far this differential
    # cycle (CLAUDE.md > Graph flow: the assess_sufficiency <-> ask_followup
    # loop). Two writers, deliberately: nodes.assess_sufficiency (a symptom-
    # detail follow-up) and nodes.rag_retrieve (a sex-clarification
    # follow-up, see that node's module docstring). Both increment by
    # exactly one, at the moment THEY decide a follow-up question is
    # needed, and both check the SAME nodes.assess_sufficiency.
    # MAX_FOLLOW_UP_QUESTIONS ceiling before incrementing — one shared
    # budget, not two independent counters. This is a deliberate, narrow
    # expansion of what used to be a strict single-writer rule, not a
    # reintroduction of the drift risk that made state["red_flags"] need
    # fixing (see git history / CLAUDE.md > State): that risk was
    # UNCOORDINATED increments against no shared limit; here both writers
    # are coordinated by the same constant and the same short-circuit
    # discipline (check the count before ever incrementing it), so the
    # ceiling's meaning — "at most this many follow-up-style questions per
    # differential cycle" — holds regardless of which node's need
    # triggered any given increment. ask_followup itself still never
    # touches this field.
    turn_count: int

    is_crisis: bool
    # Union of rule-based and LLM detections (OR, never AND). Each entry
    # is a RedFlag — {"id": ..., "reason": ...} — never split across two
    # parallel lists: the LLM's own entry (id="llm") has no rule_id to
    # pair against a same-index reason list, which is exactly the kind of
    # asymmetry that makes two-list pairing unsafe here. Doctor-facing,
    # not patient-facing: emergency_node's message never uses "reason",
    # on purpose (CLAUDE.md > Non-negotiable safety rule 1 — a matched
    # rule's reason_ar names a clinical concern, which reads as
    # diagnosis-adjacent). Surfaced in reports["doctor"] via
    # nodes/generate_reports.py so a matched rule's rationale isn't lost.
    # See nodes.check_red_flags.check_red_flags, the only place these are
    # built, and rules.red_flags.RedFlagMatch.reason_ar for the source.
    red_flags: list[RedFlag]
    # Only new clinical information may change this — never a patient objection
    # (safety rule 5). Partially wired: nodes.emergency_node sets this to
    # "emergency" unconditionally the moment it fires — a red flag firing
    # IS an emergency-level severity judgment by definition, so this is a
    # free, unambiguous mapping, not a new one requiring its own design.
    # nodes.crisis_node deliberately does NOT set this — crisis is a
    # different axis entirely (acute psychological/self-harm risk, not a
    # point on a physical-symptom-severity scale), and safety rule 9
    # requires stopping symptom analysis entirely on that path, which is
    # exactly the analysis this field would need to be derived from. The
    # low/moderate/high grading for the NORMAL (non-emergency) diagnostic
    # path remains unwired — no node computes it yet — see CLAUDE.md >
    # Known limitations for why that's a separate, tracked gap rather than
    # bolted on here.
    severity: Severity | None

    # Which terminal SAFETY outcome ("crisis" or "emergency") this thread
    # has already reached, if any — set once by crisis_node/emergency_node
    # the moment either fires, and read by graph.py's routing after
    # check_red_flags on every later turn (safety rule 13). UNLIKE stage
    # (below), never cleared by nodes/reset_stage.py — sticky for the rest
    # of the thread's life, on purpose: this is a genuinely different
    # concept from stage (which node produced THIS turn's reply) and from
    # severity (above, a graded clinical-severity judgment) — see
    # nodes/reiterate_terminal_outcome.py's module docstring for the full
    # three-way comparison. No reducer: a plain field, last-write-wins —
    # whichever terminal safety node fires MOST RECENTLY is what's
    # recorded. That's deliberate, not an oversight: crisis_check always
    # runs first every turn regardless of thread_outcome (a genuine new
    # crisis signal must always be able to fire crisis_node, even on a
    # thread that already reached "emergency"), and check_red_flags's
    # verdict is checked before thread_outcome in graph.py's routing (a
    # genuine new red flag must likewise still reach emergency_node even
    # on a thread that already reached "crisis") — either can overwrite
    # the other if a later turn's real detection legitimately fires again.
    thread_outcome: ThreadOutcome | None

    # assess_sufficiency's verdict for this turn: whether the accumulated
    # symptom picture is enough to attempt a differential, or whether one
    # more follow-up question would meaningfully narrow it. Recomputed fresh
    # each call, same reasoning as red_flags above (no reducer) — it is
    # today's judgment against the full accumulated state, not something to
    # merge with a stale prior verdict. graph.py's conditional edge reads
    # this to route between ask_followup and rag_retrieve; the node itself
    is_sufficient: bool
    # The question to ask next, Syrian colloquial Arabic. Consumed by
    # ask_followup to actually send it to the patient; the deciding node
    # only sets its text, never appends it to state["messages"] itself.
    # Two writers: assess_sufficiency (None when is_sufficient=True, a
    # real question otherwise) and nodes.rag_retrieve, which can overwrite
    # this LATER in the same turn with a sex-clarification question (see
    # that node's module docstring) even after assess_sufficiency set it
    # to None — graph.py's routing after rag_retrieve reads whichever
    # value is current at that point, so this is not a race, just a
    # later write in the same turn's sequence winning, same as any other
    # no-reducer field.
    next_question: str | None
    # True when nodes.assess_sufficiency.MAX_FOLLOW_UP_QUESTIONS was
    # reached and either writer (assess_sufficiency or nodes.rag_retrieve —
    # see turn_count's comment above) forced progress despite genuinely
    # incomplete information, rather than asking another question. Reaches
    # generate_reports so a differential made under this flag is presented
    # with that caveat, not silently as if the picture were complete.
    information_limited: bool

    # rag_retrieve's ranked output (nodes/rag_retrieve.py) — each entry a
    # state.CandidateDisease, not a bare name: safety rule 6 constrains
    # diagnose() to select only from these, and both diagnose() and the
    # doctor report need the match_score/matched/missing/negated detail
    # behind each ranking, not just which names cleared the floor.
    # Recomputed fresh each call, same reasoning as red_flags/is_sufficient
    # above (no reducer) — a fresh retrieval against the full accumulated
    # symptom set, not merged with a stale prior list. A sex-restricted KB
    # entry (rag.schema.KnowledgeBaseEntry.applicable_sex) that would
    # otherwise clear the match floor is EXCLUDED here, not merely
    # downranked, whenever patient_sex contradicts it or is unconfirmed —
    # see nodes/rag_retrieve.py's module docstring for the full gating
    # design and why this path's tolerance for a wrong/ambiguous candidate
    # is deliberately lower than rules/red_flags.py's ectopic_pregnancy
    # rule's accepted over-inclusion.
    candidate_diseases: list[CandidateDisease]
    diagnosis: dict | None  # ranked possibilities with uncertainty, or insufficient_information

    specialty: str | None
    # The Laravel-facing counterpart of `specialty` above: `specialty`
    # translated through nodes.route_specialty.SPECIALTY_MAP into one of
    # Laravel's real specializations.name_ar values (or a generic
    # referral phrase when none applies — see that module's own
    # docstring, and CLAUDE.md > Known limitations for the full
    # KB-vs-Laravel specialty comparison this exists to bridge).
    # api/main.py sends THIS field, not `specialty`, as
    # api.contracts.ChatResponse.specialty — `specialty` itself stays the
    # original, clinically-accurate KB string and is what
    # nodes/generate_reports.py's patient/doctor reports actually show.
    specialty_laravel: str | None
    reports: dict | None  # {"patient": ..., "doctor": ...}

    # Absent/None until a terminal node runs (CLAUDE.md > Graph flow) — a
    # graph invocation that hasn't reached one yet (e.g. mid-turn) simply
    # hasn't set this.
    stage: Stage | None
