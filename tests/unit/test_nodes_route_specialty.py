import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.diagnose import diagnose
from nodes.extract_symptoms import extract_symptoms
from nodes.rag_retrieve import rag_retrieve
from nodes.route_specialty import (
    GENERAL_PRACTICE,
    GENERAL_REFERRAL_PHRASE,
    LARAVEL_SPECIALTIES,
    SPECIALTY_MAP,
    route_specialty,
)


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("route_specialty must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _differential_entry(name, match_score=1.0, specialties=("طب عام",)):
    return {
        "name": name,
        "match_score": match_score,
        "certainty": "high",
        "matched_symptoms": [],
        "missing_symptoms": [],
        "negated_symptoms": [],
        "specialties": list(specialties),
    }


def _state(*, status="differential", differential=(), diagnosis=None):
    if diagnosis is None:
        diagnosis = {"status": status, "differential": list(differential), "reasoning": None}
    return {"diagnosis": diagnosis}


# --- single clear specialty from a top candidate --------------------------------


def test_single_clear_specialty_from_the_top_candidate():
    result = route_specialty(
        _state(differential=[_differential_entry("Migraine", specialties=["عصبية"])])
    )

    assert result == {"specialty": "عصبية", "specialty_laravel": "الأمراض العصبية"}


# --- insufficient_information: general practice, never a guess ------------------
#
# GENERAL_PRACTICE now resolves to Laravel's real "General Medicine" row
# (SPECIALTY_MAP was extended to route it there deliberately — see that
# map's own comment), not the plain GENERAL_REFERRAL_PHRASE text these
# tests used to expect: a patient with insufficient information gets
# matched to a real bookable doctor, not just a generic notice.


def test_insufficient_information_routes_to_general_practice():
    result = route_specialty(_state(status="insufficient_information", differential=[]))

    assert result == {
        "specialty": GENERAL_PRACTICE,
        "specialty_laravel": GENERAL_PRACTICE,
    }


def test_missing_diagnosis_entirely_routes_to_general_practice():
    # Not expected in practice (diagnose always sets it), but must not
    # crash on a state that hasn't reached diagnose yet — same "must not
    # crash on an unreached state" guarantee every other routing-adjacent
    # node in this project has.
    assert route_specialty({}) == {
        "specialty": GENERAL_PRACTICE,
        "specialty_laravel": GENERAL_PRACTICE,
    }


def test_differential_status_with_an_empty_list_routes_to_general_practice():
    # Defensive: diagnose()'s own consistency validator should prevent
    # this combination, but route_specialty must not assume that and
    # index into an empty list.
    result = route_specialty(_state(status="differential", differential=[]))

    assert result == {
        "specialty": GENERAL_PRACTICE,
        "specialty_laravel": GENERAL_PRACTICE,
    }


# --- multi-specialty differential: top candidate's own list, not an aggregate ---


def test_top_candidates_own_multiple_specialties_are_joined():
    # One disease legitimately spanning two specialties (e.g. BPPV in the
    # real knowledge base) — both are safe to report, since they're the
    # SAME clinical entity's own established relevance, not competing
    # possibilities.
    result = route_specialty(
        _state(
            differential=[
                _differential_entry("BPPV", specialties=["عصبية", "أنف وأذن وحنجرة"])
            ]
        )
    )

    # specialty (clinical, unchanged) keeps both distinct KB strings.
    assert result["specialty"] == "عصبية، أنف وأذن وحنجرة"
    # specialty_laravel: عصبية and أنف وأذن وحنجرة now map to TWO
    # different real Laravel specialties (ENT gained its own row via
    # 2026_08_17_000002_backfill_specialization_codes_and_missing_specialties.php
    # — see nodes/route_specialty.py's own LARAVEL_SPECIALTIES comment),
    # so BPPV genuinely sitting on the neuro-otology boundary now shows
    # both real specialties instead of collapsing onto neurology alone.
    assert result["specialty_laravel"] == "الأمراض العصبية، أنف وأذن وحنجرة"


def test_a_lower_ranked_candidates_specialty_does_not_leak_into_the_result():
    # The core of the "top candidate only, not an aggregate" decision:
    # a second, lower-ranked candidate's specialty must not appear in
    # the output just because it's also in the differential.
    result = route_specialty(
        _state(
            differential=[
                _differential_entry("Migraine", match_score=1.0, specialties=["عصبية"]),
                _differential_entry(
                    "Some Other Disease", match_score=0.5, specialties=["باطنية"]
                ),
            ]
        )
    )

    assert result == {"specialty": "عصبية", "specialty_laravel": "الأمراض العصبية"}
    assert "باطنية" not in result["specialty"]


def test_a_candidate_with_no_specialties_at_all_falls_back_to_general_practice():
    # Defensive: rag/knowledge_base/*.json's own schema requires at least
    # one specialty per entry, so this should not occur with real data —
    # but an empty list must not produce an empty/blank specialty string.
    result = route_specialty(
        _state(differential=[_differential_entry("Something", specialties=[])])
    )

    assert result == {
        "specialty": GENERAL_PRACTICE,
        "specialty_laravel": GENERAL_PRACTICE,
    }


# --- SPECIALTY_MAP: the Laravel bridge --------------------------------------------
#
# Every one of the 19 unique specialty strings rag/knowledge_base/*.json
# entries use, as of this session's audit (37 existing entries + the 12
# proposed in the same batch — CLAUDE.md > Known limitations has the full
# comparison against Laravel's real specializations table). Kept here as
# an explicit, static list rather than derived from the live KB directory
# so this structural guarantee holds even before/independent of any given
# KB file existing on disk.
_ALL_KB_SPECIALTY_STRINGS = (
    "أطفال",
    "جلدية",
    "نسائية",
    "عظمية",
    "عصبية",
    "عينية",
    "قلبية",
    "مسالك بولية",
    "باطنية",
    "معدية",
    "أنف وأذن وحنجرة",
    "هضمية",
    "روماتيزم",
    "غدد صماء",
    "صدرية",
    "علاج طبيعي",
    "دموية",
    "حساسية",
    GENERAL_PRACTICE,  # طب عام — the one deliberate exception, see below
)


def test_every_kb_specialty_string_is_accounted_for():
    # GENERAL_PRACTICE is now a normal SPECIALTY_MAP key too (see that
    # map's own comment) — no longer the one excluded exception.
    assert set(_ALL_KB_SPECIALTY_STRINGS) == set(SPECIALTY_MAP)


def test_every_specialty_map_value_is_a_real_laravel_specialty():
    # The structural guarantee itself: SPECIALTY_MAP can never point
    # somewhere a future doctor-matching lookup against Laravel's real
    # specializations table would silently fail to find. Also enforced at
    # import time (nodes/route_specialty.py's own module-level assert) —
    # this pins it down as a tested contract too, not just an assertion
    # nobody's watching. Covers GENERAL_PRACTICE too now — it maps to a
    # real row same as everything else.
    for kb_string in _ALL_KB_SPECIALTY_STRINGS:
        assert SPECIALTY_MAP[kb_string] in LARAVEL_SPECIALTIES, (
            f"{kb_string!r} maps to a value outside Laravel's real 21 specialties"
        )


def test_an_unrecognized_specialty_string_still_falls_back_to_the_referral_phrase():
    # GENERAL_REFERRAL_PHRASE is no longer reached by any of the 19
    # currently-known KB specialty strings (all of them, including
    # GENERAL_PRACTICE, now map to a real Laravel specialty), but it must
    # still exist as a defensive fallback for a future KB entry using a
    # specialty string nobody has added to SPECIALTY_MAP yet.
    result = route_specialty(
        _state(
            differential=[_differential_entry("Something New", specialties=["تخصص غير معروف"])]
        )
    )

    assert result["specialty_laravel"] == GENERAL_REFERRAL_PHRASE


def test_general_practice_now_maps_to_the_real_general_medicine_specialty():
    # Deliberate product decision (reversing an earlier, narrower design):
    # Laravel gained a real "General Medicine" row
    # (2026_08_17_000002_backfill_specialization_codes_and_missing_specialties.php),
    # so GENERAL_PRACTICE now resolves to it directly — a patient with no
    # more specific specialty gets matched to a real bookable doctor
    # instead of only a generic referral phrase.
    assert GENERAL_PRACTICE in SPECIALTY_MAP
    assert SPECIALTY_MAP[GENERAL_PRACTICE] == GENERAL_PRACTICE
    assert GENERAL_PRACTICE in LARAVEL_SPECIALTIES

    result = route_specialty(
        _state(differential=[_differential_entry("Influenza", specialties=[GENERAL_PRACTICE])])
    )

    assert result["specialty_laravel"] == GENERAL_PRACTICE


def test_general_practice_is_shown_alongside_a_real_co_listed_specialty():
    # rag/knowledge_base/acute_gastroenteritis.json's real shape:
    # ["هضمية", "طب عام"]. Both are now real, distinct Laravel
    # specialties (Gastroenterology and General Medicine), so both come
    # through — GENERAL_PRACTICE is no longer suppressed in favor of a
    # more specific co-listed specialty, since it is itself a real,
    # separately bookable option now, not just a fallback phrase.
    result = route_specialty(
        _state(
            differential=[
                _differential_entry(
                    "Acute Gastroenteritis", specialties=["هضمية", GENERAL_PRACTICE]
                )
            ]
        )
    )

    assert result["specialty_laravel"] == "أمراض الجهاز الهضمي، طب عام"


def test_two_kb_specialties_mapping_to_the_same_laravel_value_are_deduped():
    # rag/knowledge_base/acute_musculoskeletal_strain.json and
    # tendinitis.json's real shape: ["عظمية", "علاج طبيعي"] — both map to
    # جراحة العظام. Without dedup this would show a visibly redundant
    # "جراحة العظام، جراحة العظام".
    result = route_specialty(
        _state(
            differential=[
                _differential_entry("Tendinitis", specialties=["عظمية", "علاج طبيعي"])
            ]
        )
    )

    assert result["specialty_laravel"] == "جراحة العظام"


# --- node contract: partial state dict, no LLM call ------------------------------


def test_route_specialty_returns_only_the_specialty_key():
    result = route_specialty(
        _state(differential=[_differential_entry("Migraine", specialties=["عصبية"])])
    )

    assert set(result) == {"specialty", "specialty_laravel"}


def test_route_specialty_does_not_call_the_llm():
    set_provider(_ExplodingProvider())

    route_specialty(
        _state(differential=[_differential_entry("Migraine", specialties=["عصبية"])])
    )  # must not raise


# --- real end-to-end: extract_symptoms -> rag_retrieve -> diagnose -> route_specialty


def _extraction_response(symptoms: list[str]) -> _ProviderResponse:
    payload = {
        "symptoms": [{"name": name} for name in symptoms],
        "negated_symptoms": [],
        "unmatched_mentions": [],
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


def _diagnosis_response(status, differential=(), reasoning=None) -> _ProviderResponse:
    payload = {"status": status, "differential": list(differential), "reasoning": reasoning}
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


class _FakeProvider:
    name = "fake"

    def __init__(self, *, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def test_real_end_to_end_migraine_pattern_message_routes_to_neurology():
    # The same textbook migraine picture used throughout this session
    # (unilateral throbbing headache + nausea + photophobia +
    # phonophobia), chained through the real, unmocked
    # extract_symptoms -> rag_retrieve -> diagnose -> route_specialty
    # pipeline. Migraine's real knowledge-base entry specialty is
    # ["عصبية"] (neurology) — verified directly against the shipped KB
    # file before writing this assertion.
    migraine_symptoms = [
        "صداع نابض من جهه واحده",
        "غثيان",
        "حساسيه للضوء",
        "حساسيه للصوت",
    ]
    set_provider(
        _FakeProvider(
            responses=[
                _extraction_response(migraine_symptoms),
                _diagnosis_response("differential", differential=["Migraine"], reasoning="تطابق كامل"),
            ]
        )
    )

    state = {
        "messages": [
            {
                "role": "user",
                "content": "عندي صداع نابض من جهة وحدة، وغثيان، وحساسية من الضوء والصوت",
            }
        ]
    }
    state.update(extract_symptoms(state))
    state.update(rag_retrieve(state))
    state.update(diagnose(state))

    result = route_specialty(state)

    assert result == {"specialty": "عصبية", "specialty_laravel": "الأمراض العصبية"}
