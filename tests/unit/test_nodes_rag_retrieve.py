import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.assess_sufficiency import MAX_FOLLOW_UP_QUESTIONS
from nodes.extract_symptoms import extract_symptoms
from nodes.rag_retrieve import _SEX_CLARIFICATION_QUESTION, MIN_MATCHED_SYMPTOMS, rag_retrieve


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("rag_retrieve must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _symptom(name):
    return {"name": name}


def _state(*, symptoms=(), negated_symptoms=(), patient_sex=None, turn_count=0):
    return {
        "symptoms": [_symptom(n) for n in symptoms],
        "negated_symptoms": [_symptom(n) for n in negated_symptoms],
        "patient_sex": patient_sex,
        "turn_count": turn_count,
    }


# --- clear top match: influenza's exact symptom set ------------------------------
#
# rag/knowledge_base/influenza.json's own symptoms, normalized (two of the
# seven require it: "ألم عضلي" -> "الم عضلي", "سيلان أنف" -> "سيلان انف" —
# CLAUDE.md > Symptom vocabulary > Compare against the normalized form).
#
# Not asserted as the ONLY candidate: as the knowledge base grows, other
# viral-syndrome entries sharing a generic symptom or two (fever, cough,
# sore throat) legitimately clear MIN_MATCHED_SYMPTOMS as lower-ranked
# partial matches — genuinely correct behavior for a differential list,
# not a regression. What must hold regardless of knowledge-base size is
# narrower: Influenza itself is a perfect (1.0) match and ranks first.

INFLUENZA_SYMPTOMS = ["حمى", "سعال", "الم عضلي", "صداع", "تعب", "التهاب حلق", "سيلان انف"]


def test_influenzas_exact_symptom_set_is_a_perfect_top_ranked_match():
    result = rag_retrieve(_state(symptoms=INFLUENZA_SYMPTOMS))

    top = result["candidate_diseases"][0]
    assert top == {
        "name": "Influenza",
        "name_ar": "الإنفلونزا",
        "match_score": 1.0,
        "matched_symptoms": sorted(INFLUENZA_SYMPTOMS),
        "missing_symptoms": [],
        "negated_symptoms": [],
        "specialties": ["طب عام"],
    }


# --- negation downranks, and can exclude outright -------------------------------
#
# rag/knowledge_base/urinary_tract_infection.json: حرقة عند التبول, تبول
# متكرر, ألم أسفل البطن, حمى (4 symptoms) — normalized forms verified
# directly against schemas.symptoms.SYMPTOM_NAMES before use.


def test_a_negated_defining_symptom_downranks_relative_to_it_simply_being_missing():
    # Same 3 confirmed symptoms in both cases — only whether the 4th
    # (حمى) is explicitly denied or just never mentioned differs.
    negated_result = rag_retrieve(
        _state(
            symptoms=["حرقه عند التبول", "تبول متكرر", "الم اسفل البطن"],
            negated_symptoms=["حمى"],
        )
    )
    missing_result = rag_retrieve(
        _state(symptoms=["حرقه عند التبول", "تبول متكرر", "الم اسفل البطن"])
    )

    negated_score = negated_result["candidate_diseases"][0]["match_score"]
    missing_score = missing_result["candidate_diseases"][0]["match_score"]

    assert negated_score == 0.5
    assert missing_score == 0.75
    assert negated_score < missing_score


def test_negated_symptom_appears_in_its_own_field_not_missing():
    result = rag_retrieve(
        _state(
            symptoms=["حرقه عند التبول", "تبول متكرر", "الم اسفل البطن"],
            negated_symptoms=["حمى"],
        )
    )

    entry = result["candidate_diseases"][0]
    assert entry["negated_symptoms"] == ["حمى"]
    assert "حمى" not in entry["missing_symptoms"]
    assert entry["missing_symptoms"] == []


def test_enough_negation_excludes_the_candidate_outright():
    # 2 confirmed, 2 negated -> net_matched = 0, below MIN_MATCHED_SYMPTOMS.
    result = rag_retrieve(
        _state(
            symptoms=["حرقه عند التبول", "تبول متكرر"],
            negated_symptoms=["الم اسفل البطن", "حمى"],
        )
    )

    assert result == {"candidate_diseases": []}


# --- subsumption: a specific term satisfies a generic KB requirement -------------
#
# rules/red_flags.py's _SUBSUMES maps a generic canonical term to its more
# specific siblings (e.g. "حمى" -> {"حمى مرتفعة مفاجئة", "حمى خفيفة"}).
# nodes/rag_retrieve.py reuses that same mapping (_subsumed_evidence) so a
# KB entry that lists only the generic term still counts a patient's more
# specific report as evidence, rather than silently missing it under plain
# set-overlap. acute_otitis_media.json's own 3-symptom list — 'ألم أذن',
# 'حمى' (generic), 'فقدان سمع' — is used for both tests below since it is
# short enough to make the floor-crossing effect unambiguous.


def test_a_specific_fever_term_counts_as_evidence_for_a_kb_entrys_generic_fever():
    # The patient reports the SPECIFIC term "حمى مرتفعة مفاجئة", never the
    # bare "حمى" acute_otitis_media.json actually lists. Plain set-overlap
    # would treat these as two unrelated strings and score this 2/3; with
    # subsumption the generic requirement is satisfied and this is a full,
    # perfect match — proving the specific term is genuinely counted, not
    # just tolerated.
    result = rag_retrieve(
        _state(symptoms=["الم اذن", "حمى مرتفعه مفاجئه", "فقدان سمع"])
    )

    top = result["candidate_diseases"][0]
    assert top["name"] == "Acute Otitis Media"
    assert top["match_score"] == 1.0
    assert top["missing_symptoms"] == []
    # Reports the patient's own specific wording, not the KB's generic
    # "حمى" — same "matched_symptoms shows what was really said" precedent
    # rules.red_flags._term_matches already established.
    assert "حمى مرتفعه مفاجئه" in top["matched_symptoms"]
    assert "حمى" not in top["matched_symptoms"]


def test_subsumption_crosses_the_min_matched_symptoms_floor_not_just_downranks():
    # Only 2 of acute_otitis_media's 3 symptoms confirmed, and one of the
    # two is the SPECIFIC fever term rather than the KB's generic "حمى".
    # Without subsumption, "حمى مرتفعه مفاجئه" would match nothing on this
    # entry's list at all -> net_matched=1 ("فقدان سمع" alone), below
    # MIN_MATCHED_SYMPTOMS(2) -> excluded outright, same failure mode
    # test_a_single_vague_symptom_produces_no_candidates documents above.
    # With subsumption, it satisfies the generic "حمى" requirement -> 2
    # matched -> clears the floor and appears as a real (partial) candidate
    # instead of being silently dropped.
    result = rag_retrieve(_state(symptoms=["حمى مرتفعه مفاجئه", "فقدان سمع"]))

    names = [c["name"] for c in result["candidate_diseases"]]
    assert "Acute Otitis Media" in names

    entry = result["candidate_diseases"][names.index("Acute Otitis Media")]
    assert entry["match_score"] == round(2 / 3, 2)
    assert entry["missing_symptoms"] == ["الم اذن"]


# --- no reasonable match: a valid, expected outcome ------------------------------


def test_a_single_vague_symptom_produces_no_candidates():
    # "صداع" alone only overlaps hypertension's 2-symptom list by one
    # symptom — one match, below MIN_MATCHED_SYMPTOMS(2). This is the
    # exact counterexample MIN_MATCHED_SYMPTOMS exists for (see
    # nodes/rag_retrieve.py's module docstring).
    result = rag_retrieve(_state(symptoms=["صداع"]))

    assert result == {"candidate_diseases": []}


def test_no_symptoms_at_all_produces_no_candidates():
    result = rag_retrieve(_state())

    assert result == {"candidate_diseases": []}


# --- ranking: multiple candidates sorted by score, descending -------------------


def test_multiple_candidates_are_ranked_by_match_score_descending():
    # Asthma's full 4-symptom list (1.0) plus half of UTI's (0.5) — the
    # two sets share no symptoms, so this is genuinely two independent
    # candidates, not one entry double-counted.
    #
    # Not asserted as the ONLY two candidates: same "knowledge base
    # growth legitimately adds lower-ranked partial matches" reasoning as
    # the Influenza test above — Acute Bronchitis and Community-Acquired
    # Pneumonia (added in a later session) both share "سعال" plus one
    # more symptom with this list (ضيق بالصدر / ضيق تنفس respectively),
    # clearing MIN_MATCHED_SYMPTOMS at 0.4 each. What must hold
    # regardless of knowledge-base size is narrower: Asthma and UTI stay
    # the top two, in this exact order and score.
    result = rag_retrieve(
        _state(
            symptoms=[
                "ضيق تنفس",
                "ازيز صدر",
                "سعال",
                "ضيق بالصدر",
                "حرقه عند التبول",
                "تبول متكرر",
            ]
        )
    )

    names_and_scores = [(c["name"], c["match_score"]) for c in result["candidate_diseases"]]
    assert names_and_scores[:2] == [("Asthma", 1.0), ("Urinary Tract Infection", 0.5)]


# --- sex-specific gating: real KB entry (Dysmenorrhea), real rag_retrieve run --
#
# rag/knowledge_base/dysmenorrhea.json: ["ألم أسفل البطن", "ألم أسفل
# الظهر", "غثيان"] (location-granularity pass — was bare "ألم بطن"
# before Problem 1's fix landed), applicable_sex="female" — normalized
# forms verified directly against rules.crisis.normalize before use,
# same discipline as the UTI tests above.

DYSMENORRHEA_SYMPTOMS = ["الم اسفل البطن", "الم اسفل الظهر", "غثيان"]


def test_sex_mismatch_excludes_the_candidate_outright():
    result = rag_retrieve(_state(symptoms=DYSMENORRHEA_SYMPTOMS, patient_sex="male"))

    names = [c["name"] for c in result["candidate_diseases"]]
    assert "Dysmenorrhea" not in names


def test_sex_match_includes_the_candidate_normally():
    result = rag_retrieve(_state(symptoms=DYSMENORRHEA_SYMPTOMS, patient_sex="female"))

    top = result["candidate_diseases"][0]
    assert top["name"] == "Dysmenorrhea"
    assert top["match_score"] == 1.0


def test_a_non_sex_restricted_entry_is_unaffected_by_patient_sex():
    # Influenza has no applicable_sex — patient_sex being set at all must
    # not accidentally filter it.
    result = rag_retrieve(_state(symptoms=INFLUENZA_SYMPTOMS, patient_sex="male"))

    assert result["candidate_diseases"][0]["name"] == "Influenza"


def test_absent_patient_sex_with_an_otherwise_qualifying_candidate_triggers_a_followup():
    # This is the exact reproduced bug scenario's mechanism, isolated:
    # Dysmenorrhea's symptom set alone, with no patient_sex given at all.
    result = rag_retrieve(_state(symptoms=DYSMENORRHEA_SYMPTOMS))

    names = [c["name"] for c in result["candidate_diseases"]]
    assert "Dysmenorrhea" not in names  # neither included nor silently guessed away
    assert result["next_question"] == _SEX_CLARIFICATION_QUESTION
    assert result["turn_count"] == 1
    assert set(result) == {"candidate_diseases", "next_question", "turn_count"}


def test_followup_trigger_increments_the_existing_turn_count():
    result = rag_retrieve(_state(symptoms=DYSMENORRHEA_SYMPTOMS, turn_count=2))

    assert result["turn_count"] == 3


def test_ceiling_reached_stops_asking_and_flags_information_limited():
    # turn_count already at the shared ceiling (nodes.assess_sufficiency.
    # MAX_FOLLOW_UP_QUESTIONS) — must not ask a 7th question. The
    # candidate stays excluded (it was never included to begin with) and
    # information_limited flags the gap honestly instead.
    result = rag_retrieve(
        _state(symptoms=DYSMENORRHEA_SYMPTOMS, turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )

    names = [c["name"] for c in result["candidate_diseases"]]
    assert "Dysmenorrhea" not in names
    assert "next_question" not in result
    assert result["information_limited"] is True


def test_ceiling_reached_does_not_call_the_llm_either():
    set_provider(_ExplodingProvider())

    rag_retrieve(
        _state(symptoms=DYSMENORRHEA_SYMPTOMS, turn_count=MAX_FOLLOW_UP_QUESTIONS)
    )  # must not raise


# --- node contract: partial state dict, no LLM call ------------------------------


def test_rag_retrieve_returns_only_candidate_diseases_key():
    result = rag_retrieve(_state(symptoms=INFLUENZA_SYMPTOMS))

    assert set(result) == {"candidate_diseases"}


def test_rag_retrieve_does_not_call_the_llm():
    set_provider(_ExplodingProvider())

    rag_retrieve(_state(symptoms=INFLUENZA_SYMPTOMS))  # must not raise


# --- real end-to-end: extract_symptoms -> rag_retrieve --------------------------


def _extraction_response(symptoms: list[str]) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


class _FakeProvider:
    name = "fake"

    def __init__(self, *, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def test_real_end_to_end_migraine_pattern_message_retrieves_migraine():
    # The exact textbook migraine picture used earlier this session
    # (unilateral throbbing headache + nausea + photophobia + phonophobia)
    # — schemas.symptoms.SymptomExtraction's name enum holds the
    # NORMALIZED form, so the fake extraction response must too.
    migraine_symptoms = [
        "صداع نابض من جهه واحده",
        "غثيان",
        "حساسيه للضوء",
        "حساسيه للصوت",
    ]
    set_provider(_FakeProvider(responses=[_extraction_response(migraine_symptoms)]))

    state = {
        "messages": [
            {
                "role": "user",
                "content": "عندي صداع نابض من جهة وحدة، وغثيان، وحساسية من الضوء والصوت",
            }
        ]
    }
    state.update(extract_symptoms(state))

    result = rag_retrieve(state)

    assert result["candidate_diseases"][0]["name"] == "Migraine"
    assert result["candidate_diseases"][0]["name_ar"] == "الشقيقة"
    assert result["candidate_diseases"][0]["match_score"] == 1.0


def test_real_end_to_end_regression_upper_abdominal_pain_no_longer_surfaces_dysmenorrhea():
    # Exact reproduction of this session's manual-testing bug: a patient
    # describing upper abdominal pain + nausea, no sex given, originally
    # got Dysmenorrhea as the TOP-ranked candidate (0.67, ahead of Acute
    # Gastroenteritis/Hepatitis A at 0.40 each) purely from generic "ألم
    # بطن" symptom-name overlap.
    #
    # The extraction response below is the REAL one, RE-VERIFIED via
    # scripts/try_extract_manually.py against "عندي ألم قوي براس معدتي
    # من فوق، مع غثيان" AFTER Problem 1's location-granularity fix
    # landed: the model now correctly selects the new, more specific
    # "ألم أعلى البطن" (upper abdominal pain) instead of the old bare
    # "ألم بطن" — the fix isn't just a KB-side relabeling, the model
    # actually uses the more precise term once it exists as an option.
    #
    # Dysmenorrhea stays excluded regardless of the _SUBSUMES fix below:
    # its list says "ألم أسفل البطن" (LOWER), a different sibling under
    # the same "ألم بطن" parent, not a sibling of "ألم أعلى البطن"
    # (UPPER) itself — the two never satisfy each other, so this
    # exclusion is unaffected and still fires on the location mismatch
    # alone (net_matched=1: only "غثيان" survives, below
    # MIN_MATCHED_SYMPTOMS).
    #
    # Acute Gastroenteritis is DELIBERATELY left on the bare generic
    # "ألم بطن" in its own KB entry (its cited WHO source never specified
    # a location — see the Problem 1 mapping report). Before the
    # _SUBSUMES wiring (CLAUDE.md > State > candidate_diseases), that
    # meant "ألم أعلى البطن" and "ألم بطن" were different canonical
    # strings and plain set-overlap matching silently never connected
    # them — this test originally pinned THAT gap as correct behavior.
    # It was the gap, not the fix: rules.red_flags._SUBSUMES already
    # documents "ألم أعلى البطن" as a specific sibling of the generic
    # "ألم بطن", and nodes.rag_retrieve now applies that same mapping
    # (_subsumed_evidence), so the generic entry correctly recognizes the
    # patient's more specific term as satisfying evidence. Acute
    # Gastroenteritis therefore now ALSO matches: "ألم بطن" (via
    # subsumption) + "غثيان" (direct) = 2, clearing MIN_MATCHED_SYMPTOMS.
    # Both it and Hepatitis A land on an identical match_score (2/5 =
    # 0.4 each — rag/knowledge_base/acute_gastroenteritis.json and
    # hepatitis_a.json both list exactly 5 symptoms), so the tie is
    # broken by load_all()'s stable, alphabetical-by-filename load order
    # ("acute_gastroenteritis.json" sorts before "hepatitis_a.json"),
    # putting Acute Gastroenteritis ahead of Hepatitis A. No
    # sex-clarification follow-up fires: none of the remaining candidates
    # have an applicable_sex.
    #
    # A later session added Peptic Ulcer Disease ("ألم أعلى البطن",
    # "انتفاخ", "غثيان", "فقدان شهية" — 4 symptoms total) to the
    # knowledge base. It matches this exact patient too — "ألم أعلى
    # البطن" + "غثيان" both direct, 2/4 = 0.5 — and legitimately outranks
    # both 0.4 matches: a patient reporting isolated upper abdominal pain
    # with nausea is a genuinely plausible peptic-ulcer presentation, at
    # least as plausible as gastroenteritis or hepatitis A here. This is
    # the same knowledge-base-growth effect the Influenza test documents
    # above, not a new bug — and the exact match-score-comparability
    # limitation CLAUDE.md > Known limitations already documents (PUD's
    # higher score comes from a shorter 4-symptom list, not from being a
    # more confident match).
    set_provider(
        _FakeProvider(
            responses=[
                _extraction_response(["الم اعلى البطن", "غثيان"]),
            ]
        )
    )

    state = {
        "messages": [
            {"role": "user", "content": "عندي ألم قوي براس معدتي من فوق، مع غثيان"}
        ]
    }
    state.update(extract_symptoms(state))

    result = rag_retrieve(state)

    names = [c["name"] for c in result["candidate_diseases"]]
    assert "Dysmenorrhea" not in names
    assert names == ["Peptic Ulcer Disease", "Acute Gastroenteritis", "Hepatitis A"]
    assert result["candidate_diseases"][0]["match_score"] == 0.5
    assert result["candidate_diseases"][1]["match_score"] == 0.4
    assert result["candidate_diseases"][2]["match_score"] == 0.4
    assert "next_question" not in result
