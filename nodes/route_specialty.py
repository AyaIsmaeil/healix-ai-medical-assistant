"""route_specialty: derives state["specialty"] from diagnose's ranked
differential (CLAUDE.md > Graph flow: runs after diagnose, before
generate_reports — wired in graph.py as diagnose -> route_specialty ->
generate_reports).

No LLM call: rag/knowledge_base/*.json entries already carry a
specialties field. It's propagated unchanged through
state["candidate_diseases"] (nodes/rag_retrieve.py) into
state["diagnosis"]["differential"][i]["specialties"] (nodes/diagnose.py)
specifically so this node needs neither a second, independent
rag/knowledge_base/ lookup nor a fresh LLM judgment that could
contradict already-computed data — a second LLM call here would be
exactly the "silent substitution / silently degraded result" pattern
CLAUDE.md > LLM tiers already refuses elsewhere in this project (no
cross-provider fallback, no re-deriving a computed score), applied to
specialty routing instead of a score.

--- insufficient_information -> طب عام (general practice), never a
guess. With no differential to route from, a general-practice referral
is the only honest recommendation — same reasoning as diagnose() itself
refusing to force a pick (CLAUDE.md > Non-negotiable safety rule 6).

--- Multi-specialty decision, reasoned as asked rather than assumed:
this reports the TOP-RANKED candidate's OWN specialties only — never an
aggregate across every candidate in the differential. Two reasons:

  1. state["specialty"] is a single str | None, in both state.py and
     api.contracts.ChatResponse.specialty — an already-fixed field in
     the Laravel <-> service contract, not something to redefine as a
     list without a strong reason (CLAUDE.md: contracts.py is "the
     authoritative definition of the HTTP boundary"). This node's job,
     per its own name, is ROUTING — resolving to one clear
     recommendation Laravel's booking flow can act on, not surfacing
     every possibility the differential contains.
  2. A single disease's OWN specialties list (e.g.
     benign_paroxysmal_positional_vertigo.json -> ["عصبية", "أنف وأذن
     وحنجرة"]) is safe to join: that is one already-identified clinical
     entity's genuine, source-established multi-specialty relevance, not
     two competing possibilities. Joining specialties ACROSS different
     candidates would be different in kind — those are competing,
     uncertain alternatives (safety rule 1: ranked possibilities with
     uncertainty, not a single answer), and neither this service nor
     Laravel has any defined way for a downstream booking flow to parse
     a string like "عصبية، باطنية" as "either is fine" versus a single
     (wrong) combined specialty name. Diluting the one clear signal this
     field is supposed to carry is worse than only surfacing the
     top-ranked candidate's own.

Nothing is actually lost by this choice: the full differential, with
every candidate's own specialties, stays exactly where diagnose() put
it in state["diagnosis"] — generate_reports draws on lower-ranked
candidates' specialties for an "also consider" note when relevant.
This field just doesn't encode that; it encodes the one routing
recommendation.

--- Two specialty fields, not one: state["specialty"] (this node's
original output, unchanged by anything below) is the KB's own
clinically-accurate string(s) — generate_reports.py reads THIS one for
both the patient- and doctor-facing report text, so a report still
correctly says "معدية" or "طب عام" when that's the real clinical
category. state["specialty_laravel"] is new: rag/knowledge_base/*.json's
19 unique specialty strings were authored independently of Laravel's own
`specializations` table and, per this session's audit, don't match it —
not one is a byte-exact string match to any of Laravel's 11 real rows
(see CLAUDE.md > Known limitations for the full comparison). Sending the
raw KB string as ChatResponse.specialty would mean any future
doctor-matching lookup against that table silently never matches.
specialty_laravel is what api/main.py actually sends as
ChatResponse.specialty; specialty stays internal-and-reports-only. See
SPECIALTY_MAP and _resolve_laravel_specialty below for the translation
itself.
"""

from __future__ import annotations

from typing import Any

from state import HealixState

# CLAUDE.md's own task description names this exact string — kept as a
# named constant rather than repeated inline, matching every other fixed
# fallback value in this project (e.g. crisis_node's generic referral).
GENERAL_PRACTICE = "طب عام"

# --- Laravel specialty bridge -------------------------------------------------
#
# rag/knowledge_base/*.json's 19 unique specialty strings were authored
# against clinical convention, independent of Laravel's own
# `specializations` table — nobody had compared the two until this
# session's audit. Compared directly against the LIVE database (`php
# artisan tinker`, not just seeder/migration source, since those told two
# different stories: the seeder has 11 rows, a later migration's
# translation dictionary has 20 entries but only backfills name_ar on
# rows that already exist — 9 of those 20 are dead code, translations
# for specializations that were never actually seeded). Zero of the 19 KB
# strings are a byte-exact match to any of the 11 real name_ar values.
# See CLAUDE.md > Known limitations for the full comparison table and the
# reasoning behind every mapping below — this is not guessed cold.
#
# LARAVEL_SPECIALTIES is the live database's real name_ar column, as of
# this writing (11 rows: database/seeders/SpecializationsTableSeeder.php
# in the Laravel repo) — kept here, not re-derived, because this service
# has no live connection to that database; a real change on the Laravel
# side needs a matching edit here, deliberately not automated.
LARAVEL_SPECIALTIES: frozenset[str] = frozenset(
    {
        "الأمراض الجلدية",  # Dermatology
        "أمراض القلب",  # Cardiology
        "جراحة العظام",  # Orthopedics
        "طب الأطفال",  # Pediatrics
        "الأمراض العصبية",  # Neurology
        "الأورام",  # Oncology
        "طب العيون",  # Ophthalmology
        "الجراحة العامة",  # General Surgery
        "الطب النفسي",  # Psychiatry
        "أمراض النساء والولادة",  # Gynecology
        "المسالك البولية",  # Urology
    }
)

# Translates every KB specialty string EXCEPT GENERAL_PRACTICE into one
# of LARAVEL_SPECIALTIES. GENERAL_PRACTICE is deliberately excluded, not
# an oversight — see _resolve_laravel_specialty's own docstring for why
# it gets different handling entirely.
SPECIALTY_MAP: dict[str, str] = {
    # Wording-only: Laravel already has the specialty, phrased as a full
    # clinical noun phrase where the KB uses a bare adjective/noun.
    "أطفال": "طب الأطفال",
    "جلدية": "الأمراض الجلدية",
    "نسائية": "أمراض النساء والولادة",
    "عظمية": "جراحة العظام",
    "عصبية": "الأمراض العصبية",
    "عينية": "طب العيون",
    "قلبية": "أمراض القلب",
    "مسالك بولية": "المسالك البولية",
    # No Laravel equivalent at all, even among the 9 dead-code
    # translations in that unused migration dictionary — mapped to the
    # single closest real specialty. Each choice is a real clinical
    # judgment call, not arbitrary; see CLAUDE.md > Known limitations for
    # the reasoning behind every one, including the two flagged as
    # genuinely imperfect (هضمية -> surgery for medically-managed GI
    # conditions; the whole set concentrating onto قلبية/جراحة العظام
    # because Laravel has no general-medicine specialty to spread onto).
    "باطنية": "أمراض القلب",
    "معدية": "طب الأطفال",
    "أنف وأذن وحنجرة": "الأمراض العصبية",
    "هضمية": "الجراحة العامة",
    "روماتيزم": "جراحة العظام",
    "غدد صماء": "أمراض القلب",
    "صدرية": "أمراض القلب",
    "علاج طبيعي": "جراحة العظام",
    "دموية": "أمراض القلب",
    "حساسية": "الأمراض الجلدية",
}

assert set(SPECIALTY_MAP.values()) <= LARAVEL_SPECIALTIES, (
    "SPECIALTY_MAP has a value outside LARAVEL_SPECIALTIES — every "
    "mapped value must be one of Laravel's real specializations.name_ar "
    "rows, or a future doctor-matching lookup against that table would "
    "silently never match."
)

# Shown in place of a forced specialty when a turn's KB specialties, once
# GENERAL_PRACTICE is excluded, leave nothing behind — see
# _resolve_laravel_specialty's own docstring for the full reasoning.
# api.contracts.ChatResponse.specialty is a plain `str | None`, not a
# Literal of LARAVEL_SPECIALTIES, so this is valid against the type
# contract; it is a deliberate, documented exception to "always one of
# the 11 real values" (CLAUDE.md > Known limitations), not a bug.
GENERAL_REFERRAL_PHRASE = "استشر طبيب أو راجع أقرب مركز رعاية أولية"


def _resolve_laravel_specialty(kb_specialties: list[str]) -> str:
    """Translate this turn's KB-native specialty string(s) into the value
    ChatResponse.specialty should actually carry (api/main.py reads
    state["specialty_laravel"], this function's return value, for
    exactly that field — see state.py's own comment on that field).

    GENERAL_PRACTICE ("طب عام") is deliberately excluded from
    SPECIALTY_MAP rather than forced onto one of Laravel's 11 real
    specialties: every other no-equivalent KB string has at least one
    defensible "closest" real specialty (see SPECIALTY_MAP's own
    comment), but طب عام doesn't — Laravel has no general-practice or
    family-medicine specialty at all, and forcing it onto e.g. Pediatrics
    would misroute the (typically adult) patients this path serves
    toward a children's specialist, which risks undermining trust in the
    system more than an unmapped fallback would. When none of a turn's
    KB specialties survive the map — either the list was just
    [GENERAL_PRACTICE] (e.g. Influenza's own sole specialty, or
    route_specialty's own insufficient_information fallback), or every
    entry in it was — this returns GENERAL_REFERRAL_PHRASE instead of a
    specialty name.

    A specialty that IS co-listed alongside GENERAL_PRACTICE (e.g.
    rag/knowledge_base/acute_gastroenteritis.json's ["هضمية", "طب عام"])
    still maps normally: GENERAL_PRACTICE is simply dropped from the
    joined result, not replacing it with the referral phrase — the
    referral phrase is only for the case where NOTHING more specific is
    available.

    Deduplicates the mapped result, preserving order: two different KB
    specialties can map onto the SAME Laravel value (real, not
    hypothetical — rag/knowledge_base/acute_musculoskeletal_strain.json
    and tendinitis.json both list ["عظمية", "علاج طبيعي"], which both map
    to "جراحة العظام"), and joining that unreduced would show a visibly
    redundant "جراحة العظام، جراحة العظام".
    """
    mapped = [SPECIALTY_MAP[s] for s in kb_specialties if s in SPECIALTY_MAP]
    if not mapped:
        return GENERAL_REFERRAL_PHRASE

    seen: set[str] = set()
    deduped = [value for value in mapped if not (value in seen or seen.add(value))]
    return "، ".join(deduped)


def route_specialty(state: HealixState) -> dict[str, Any]:
    """Recommend a specialty from diagnose's top-ranked differential candidate.

    Returns BOTH specialty (the original, clinically-accurate KB
    string(s) — unchanged behavior, what generate_reports shows) and
    specialty_laravel (the new SPECIALTY_MAP-translated value —
    what api/main.py sends as ChatResponse.specialty). See state.py's
    own comment on specialty_laravel for why these are two separate
    fields rather than one being overwritten in place.
    """
    diagnosis = state.get("diagnosis") or {}
    differential = diagnosis.get("differential") or []

    if diagnosis.get("status") != "differential" or not differential:
        return {
            "specialty": GENERAL_PRACTICE,
            "specialty_laravel": _resolve_laravel_specialty([GENERAL_PRACTICE]),
        }

    # Already sorted by match_score, descending (nodes.diagnose) — the
    # first entry is reliably the top-ranked candidate, not something
    # this node needs to re-sort or re-derive.
    top_candidate = differential[0]
    specialties = top_candidate.get("specialties") or [GENERAL_PRACTICE]

    return {
        "specialty": "، ".join(specialties),
        "specialty_laravel": _resolve_laravel_specialty(specialties),
    }
