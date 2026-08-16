"""rag_retrieve: matches the patient's confirmed symptoms against
rag/knowledge_base/*.json by set overlap, producing a ranked list of
candidate diseases (CLAUDE.md > Graph flow: runs after assess_sufficiency
judges the picture sufficient, before diagnose).

No LLM call — this is retrieval, not generation. CLAUDE.md > Non-negotiable
safety rule 7 requires the eventual diagnosis to cite a COMPUTED match
score (matched symptoms / reference symptoms), never an LLM-reported
confidence — this node is where that computation actually happens; every
score downstream is quoting a number produced here, not a judgment call.

Matching against state["unmatched_mentions"] is structurally impossible,
not just skipped: rag/knowledge_base/ entries only ever reference
vocabulary/symptoms.py's canonical names (CLAUDE.md > Vocabulary
provenance and growth — every KB symptom is reconciled to a canonical
entry before being added), and an unmatched mention is by definition a
phrase that didn't map to one. It stays in state for the doctor report,
untouched here.

Both sides of the comparison are normalized (rules.crisis.normalize)
before matching, never compared in authored spelling — CLAUDE.md >
Symptom vocabulary > "Compare against the normalized form, never the
authored spelling" applies here exactly as it does everywhere else:
rag/knowledge_base/*.json stores each disease's symptoms in natural,
human-authored spelling (e.g. "ألم في الصدر", with hamza), while
state["symptoms"]'s names are already the NORMALIZED form (schemas.
symptoms.SymptomName's enum is generated from the normalized vocabulary
set). Comparing either side raw would silently miss matches whenever the
two spellings differ by hamza/ta-marbuta/diacritics — the exact failure
mode that section of CLAUDE.md exists to prevent, and it applies to any
new code comparing symptom names, not just the vocabulary layer itself.

--- Negation: a defining symptom the patient denied is counter-evidence,
not just an absence (CLAUDE.md > State: negated_symptoms carries
diagnostic weight equal to confirmed symptoms). Each one directly cancels
out one confirmed match in the score, via net_matched =
len(matched) - len(negated_hits), floored at zero — a candidate with
several OTHER confirmed matches ("a compensating match elsewhere")
survives this with a lower but still positive score; one whose only
matches are also its negated ones does not clear MIN_MATCHED_SYMPTOMS
and is excluded outright, never appearing in the output at all.

--- MIN_MATCHED_SYMPTOMS (see below) is an ABSOLUTE floor, not a
percentage of each disease's symptom list, and applies to net_matched
(post-negation), not the raw match count. Found necessary empirically,
not assumed: rag/knowledge_base/hypertension.json lists only two
symptoms ("صداع", "دوخة" — headache, dizziness), both individually
common and nonspecific (the entry's own "note": "Most cases are
asymptomatic"). A single matched symptom against that two-symptom list
already scores 0.5 under the plain matched/total formula — a bare
headache alone would read as a strong hypertension match despite being
one of the least specific complaints in the entire knowledge base. A
percentage-only floor cannot fix this (0.5 already clears most
reasonable percentage thresholds); an absolute minimum count does,
without unfairly penalizing longer symptom lists like influenza's seven
just for being longer. Two was chosen as the smallest count that stops
this specific counterexample while still being clearable by every other
current KB entry (every entry has at least two symptoms) — not a
generically "safe-sounding" round number.

--- A candidate list can legitimately end up empty: if nothing clears
MIN_MATCHED_SYMPTOMS, that means insufficient match against the current
knowledge base, not a retrieval failure — diagnose() (CLAUDE.md >
Non-negotiable safety rule 6) treats an empty list as
insufficient_information being a valid, expected output, exactly as
CLAUDE.md already documents for that node.

--- KNOWN LIMITATION, not a bug (CLAUDE.md > Known limitations): a
match_score is only meaningfully comparable WITHIN one disease's own
match, not ACROSS different diseases, while rag/knowledge_base/ stays
this small and this uneven in symptom-list length. Do not read "higher
score = more confident" across two different candidate_diseases entries.
This is a property of the reference data's current size and evenness,
not something MIN_MATCHED_SYMPTOMS (above) fixes or was meant to fix:
that guards a single score from being misleadingly high in isolation;
this is about comparing two already-correct scores to each other.
Resolves as the knowledge base grows broader and more even, not via a
code change here.

--- Sex-specific gating. rag.schema.KnowledgeBaseEntry.applicable_sex
marks the 3 KB entries that are anatomically restricted to one sex
(dysmenorrhea, polycystic_ovary_syndrome, vaginal_candidiasis) —
reviewed the full KB for any others while wiring this in;
urinary_tract_infection, iron_deficiency_anaemia, and mumps are all
more-common-in-one-sex, not exclusive-to-one-sex, so none of them
qualify (see applicable_sex's own field description for why that
distinction matters), and no other entry in the current 37 does either.

Motivated by a real reproduced bug: a patient describing upper abdominal
pain + nausea, no sex given, got Dysmenorrhea as the TOP-ranked
candidate — symptom-name overlap alone has no way to know that
"abdominal pain + nausea" describing a male patient cannot be menstrual
pain. Gating happens only AFTER an entry already clears
MIN_MATCHED_SYMPTOMS (see _match_entry) — sex is irrelevant to a
candidate that wasn't going to appear anyway. Three cases from there:

1. state["patient_sex"] contradicts entry.applicable_sex -> excluded
   outright, same as failing the floor. Never appears in
   candidate_diseases at all.
2. state["patient_sex"] matches (or the entry has no applicable_sex) ->
   ordinary candidate, unaffected.
3. state["patient_sex"] is None/absent AND the entry would otherwise
   qualify -> NEITHER included NOR silently excluded. Per the accepted
   design (Problem 2, Option A + D fallback): treated as insufficient
   information specifically for this ambiguity, triggering a follow-up
   question through the SAME next_question/turn_count/ask_followup
   mechanism assess_sufficiency's own insufficient verdict uses — not a
   new mechanism, not a guess in either direction. See
   _SEX_CLARIFICATION_QUESTION and rag_retrieve()'s own logic below.

--- Why this path excludes/asks rather than following the
ectopic_pregnancy red-flag rule's already-established "accept
over-inclusion" precedent (rules/red_flags.py, deliberately NOT touched
by this change): the two paths have different failure costs. A
false-positive ectopic_pregnancy firing costs one unnecessary ER
referral — cheap, and the missed-true-case alternative is a
life-threatening emergency, so over-inclusion is the right bias there
(that rule's own comment makes this tradeoff explicit). A false positive
HERE is different in kind, not just degree: it is a candidate appearing
in a ranked, evidence-presented DIFFERENTIAL (safety rule 1's "ranked
possibilities" standard) — being ranked and shown at all confers a kind
of credibility a red-flag referral never claims. A clinically-impossible
candidate in that list can actively misdirect a patient or a doctor, not
just cost an extra visit. Different cost structures justify different
tolerances — this is a deliberate divergence from the ectopic_pregnancy
precedent, not an inconsistency with it.

--- What this does NOT solve, flagged rather than silently left: there
is no mechanism anywhere that captures the patient's ANSWER to the sex-
clarification follow-up question back into state["patient_sex"] — that
field's only real source is Laravel (api.contracts.ChatRequest.patient_sex,
CLAUDE.md > Laravel <-> service contract), structured and reliable, not
conversational inference. If Laravel never sends it, this follow-up
question will keep firing every time this exact ambiguity recurs on a
thread, without ever actually resolving it from the conversation — bounded
by the same MAX_FOLLOW_UP_QUESTIONS ceiling assess_sufficiency's own loop
respects (see rag_retrieve() below), so this degrades safely once spent
(the ambiguous candidate stays excluded, information_limited is set),
never an infinite loop, but genuinely never learned from chat alone
either. Extracting patient_sex from free text was deliberately not built:
it would be exactly the kind of "infer a safety-adjacent demographic from
conversation" mechanism the ectopic_pregnancy rule's own comment already
declined to improvise — doing it here while still declining it there
would be an inconsistent double standard, not a considered exception.

--- turn_count (CLAUDE.md > State) gains a SECOND writer here, alongside
nodes.assess_sufficiency. Deliberate, not a reintroduction of the drift
risk CLAUDE.md's original single-writer rule guarded against: both
writers represent the exact same conceptual event — "a follow-up
question was asked this turn" — from two different gaps (assess_
sufficiency: the symptom picture itself; this node: an unconfirmed
patient_sex blocking a sex-restricted candidate) — and both increment
the exact same shared budget, checked against the exact same
MAX_FOLLOW_UP_QUESTIONS ceiling, before ever incrementing. One shared
counter with coordinated writers, not two independent ones.

Coordinated is not just a description here, it's a structural
guarantee: the two writers' increment branches CANNOT both fire in the
same turn. rag_retrieve only ever runs when assess_sufficiency already
returned is_sufficient=True this same turn — and assess_sufficiency's
own turn_count increment happens ONLY on its is_sufficient=False branch
(nodes/assess_sufficiency.py), which routes the turn to ask_followup
and ends it there, never reaching this node at all. So whichever writer
increments turn_count on a given turn, the other node did not even run
that turn — proven by graph.py's own conditional edges (_route_after_
assess_sufficiency, _route_after_rag_retrieve), not just asserted here.
See nodes/assess_sufficiency.py's own module docstring for the mirror
of this same argument from its side, and
tests/unit/test_graph.py::test_a_turn_where_assess_sufficiency_is_insufficient_never_lets_rag_retrieve_increment_turn_count
for the test-level proof.

Once the ceiling is already spent, this node does not keep asking: the
ambiguous candidate stays excluded (case 3 above never included it to
begin with — only the QUESTION stops firing) and
state["information_limited"] is set True, the same honest caveat
assess_sufficiency's own ceiling forces elsewhere.
"""

from __future__ import annotations

from typing import Any

from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS
from rag.schema import KnowledgeBaseEntry, load_all
from rules.crisis import normalize
from rules.red_flags import _SUBSUMES
from state import CandidateDisease, HealixState, PatientSex, Symptom

# See module docstring's "MIN_MATCHED_SYMPTOMS" section for why this is an
# absolute count, not a fraction, and why 2 specifically.
MIN_MATCHED_SYMPTOMS = 2

# Reused, not reinvented: the exact same generic-term -> specific-siblings
# mapping rules/red_flags.py's SymptomRequirement.satisfied_by already
# applies to red-flag rules, hand-authored and reviewed there (see that
# module's own comment for the full audit — which pairs were included and
# which were deliberately left out, e.g. "تقيؤ" -> "تقيؤ دم"). The gap this
# closes is structurally identical, just at a different call site: a KB
# entry listing the generic "حمى" would silently never match an extraction
# that correctly picked the specific "حمى مرتفعة مفاجئة" — set intersection
# alone has no concept that one canonical term is a more specific case of
# another. Confirmed scope (audited, not guessed): 12 rag/knowledge_base/
# entries use bare "حمى" (acute_gastroenteritis, acute_otitis_media,
# chickenpox, hand_foot_and_mouth_disease, infectious_mononucleosis,
# influenza, measles, mumps, streptococcal_pharyngitis, tonsillitis,
# typhoid_fever, urinary_tract_infection), and 2 use bare "ألم بطن"
# (acute_gastroenteritis, typhoid_fever) — every one of them silently
# under-scored a patient who described the specific form, and some
# combinations tipped below MIN_MATCHED_SYMPTOMS's absolute floor,
# excluding the candidate outright rather than merely down-ranking it.
#
# Normalized once here, mirroring rules/red_flags.py's own _SUBSUMES_N
# construction — imported the reviewed DATA (_SUBSUMES), not that
# module's private normalized cache, so this module's own normalization
# stays self-contained rather than reaching into another module's
# implementation detail.
_SUBSUMES_N: dict[str, frozenset[str]] = {
    normalize(generic): frozenset(normalize(specific) for specific in specifics)
    for generic, specifics in _SUBSUMES.items()
}


def _subsumed_evidence(kb_symptoms: frozenset[str], present: set[str]) -> dict[str, str]:
    """kb_symptom -> the patient's actual reported term satisfying it in
    `present`, for every kb_symptom with real evidence either directly or
    via a _SUBSUMES_N specific sibling. A kb_symptom with no evidence in
    `present` at all is simply absent from the returned mapping.

    Reports the patient's own specific term as the value, not the KB's
    generic wording — same "matched_symptoms shows what was really
    extracted" precedent as rules.red_flags._term_matches's own
    docstring, applied here so a doctor report shows "حمى مرتفعة مفاجئة"
    when that is what the patient actually said, not a generic "حمى"
    they never used.
    """
    evidence: dict[str, str] = {}
    for kb_symptom in kb_symptoms:
        if kb_symptom in present:
            evidence[kb_symptom] = kb_symptom
            continue
        siblings = _SUBSUMES_N.get(kb_symptom, frozenset()) & present
        if siblings:
            evidence[kb_symptom] = next(iter(siblings))
    return evidence

# Code-generated, no LLM call — same "hand-authored, reviewed, no risk of
# drift" reasoning as emergency_node's fixed message. Deliberately
# disease-silent: naming which candidate prompted this would lean toward
# announcing a diagnosis before diagnose() has even run (safety rule 6).
_SEX_CLARIFICATION_QUESTION = (
    "قبل ما نكمل، ممكن تحكيلي إذا كنت رجل أو امرأة؟ هيك بقدر أعطيك تقييم أدق."
)


def _normalized_names(symptoms: list[Symptom]) -> set[str]:
    return {normalize(symptom["name"]) for symptom in symptoms if symptom.get("name")}


def _match_entry(
    entry: KnowledgeBaseEntry,
    confirmed: set[str],
    negated: set[str],
    patient_sex: PatientSex | None,
) -> tuple[CandidateDisease | None, bool]:
    """One disease's match detail, or (None, needs_sex_clarification).

    The second value is True only when this entry would otherwise
    qualify (clears the floor below) but is sex-restricted
    (rag.schema.KnowledgeBaseEntry.applicable_sex) and patient_sex is
    unknown — see module docstring's "Sex-specific gating" section. A
    confirmed sex MISMATCH returns (None, False): that candidate is
    definitively excluded, not ambiguous, so it must not also trigger a
    follow-up question.
    """
    kb_symptoms = frozenset(normalize(name) for name in entry.symptoms)

    # confirmed_evidence/negated_evidence map kb_symptom -> the patient's
    # actual term satisfying it, direct or via _SUBSUMES_N — computed
    # independently of each other (not elif/continue), same overlap-
    # tolerant shape the original plain-intersection version had: a
    # kb_symptom the patient both confirmed and later denied (a real,
    # documented scenario — CLAUDE.md > State) can still land in both,
    # netting out via net_matched below exactly as it did before.
    confirmed_evidence = _subsumed_evidence(kb_symptoms, confirmed)
    if not confirmed_evidence:
        return None, False  # zero positive evidence — never a candidate, regardless of negation

    negated_evidence = _subsumed_evidence(kb_symptoms, negated)
    net_matched = max(0, len(confirmed_evidence) - len(negated_evidence))
    if net_matched < MIN_MATCHED_SYMPTOMS:
        return None, False

    if entry.applicable_sex is not None:
        if patient_sex is None:
            return None, True  # would qualify, but sex-restricted and unconfirmed
        if patient_sex != entry.applicable_sex:
            return None, False  # confirmed mismatch — excluded outright, not ambiguous

    missing = kb_symptoms - confirmed_evidence.keys() - negated_evidence.keys()

    return {
        "name": entry.name,
        "name_ar": entry.name_ar,
        "match_score": round(net_matched / len(kb_symptoms), 2),
        "matched_symptoms": sorted(confirmed_evidence.values()),
        "missing_symptoms": sorted(missing),
        "negated_symptoms": sorted(negated_evidence.values()),
        # Carried forward from the KB entry as-is (not deduplicated or
        # reordered here) so nodes.route_specialty can use it without a
        # second, independent lookup against rag/knowledge_base/ — same
        # "downstream needs this detail" reasoning as matched/missing/
        # negated_symptoms above. name_ar (above) is carried forward for
        # the identical reason, one hop further downstream: nodes.
        # generate_reports's patient register needs it and must never
        # re-derive or translate a name itself (rag/schema.py: name_ar is
        # authored per entry, not mechanically derivable from name).
        "specialties": list(entry.specialties),
    }, False


def rag_retrieve(state: HealixState) -> dict[str, Any]:
    """Rank candidate diseases by symptom overlap against the knowledge base."""
    confirmed = _normalized_names(state.get("symptoms", []))
    negated = _normalized_names(state.get("negated_symptoms", []))
    patient_sex = state.get("patient_sex")

    candidates: list[CandidateDisease] = []
    needs_sex_clarification = False
    for entry in load_all():
        match, ambiguous = _match_entry(entry, confirmed, negated, patient_sex)
        if match is not None:
            candidates.append(match)
        if ambiguous:
            needs_sex_clarification = True

    candidates.sort(key=lambda candidate: candidate["match_score"], reverse=True)

    if needs_sex_clarification:
        turn_count = state.get("turn_count", 0)
        if turn_count < MAX_FOLLOW_UP_QUESTIONS:
            # Reuses ask_followup's own contract (next_question + a
            # turn_count increment against the shared ceiling) rather
            # than inventing a second follow-up mechanism — see module
            # docstring's turn_count section.
            return {
                "candidate_diseases": candidates,
                "next_question": _SEX_CLARIFICATION_QUESTION,
                "turn_count": turn_count + 1,
            }
        # Ceiling already spent — stop asking. The ambiguous candidate(s)
        # were never included in `candidates` to begin with (see
        # _match_entry), so no further exclusion is needed here; only
        # honesty about the gap is (same caveat assess_sufficiency's own
        # ceiling forces).
        return {"candidate_diseases": candidates, "information_limited": True}

    return {"candidate_diseases": candidates}
