"""route_specialty: derives state["specialty"] from diagnose's ranked
differential. No LLM call — rag/knowledge_base entries already carry a
specialties field, propagated through unchanged.

insufficient_information -> طب عام (general practice), never a guess.

Reports the TOP-RANKED candidate's own specialties only, never an
aggregate across the whole differential: state["specialty"] is a single
str, and joining specialties ACROSS competing candidates would produce
an unparseable, misleading combined string. The full differential (with
every candidate's own specialties) stays in state["diagnosis"] for
generate_reports to draw on.

Two specialty fields: `specialty` is the KB's own clinically-accurate
string (what generate_reports shows). `specialty_laravel` is
SPECIALTY_MAP-translated into one of Laravel's real specializations —
the KB's 19 specialty strings don't match Laravel's table byte-for-byte,
so sending the raw KB string as ChatResponse.specialty would silently
never match a doctor-lookup query.
"""

from __future__ import annotations

from typing import Any

from state import HealixState

GENERAL_PRACTICE = "طب عام"

# rag/knowledge_base's 19 specialty strings don't byte-match Laravel's
# specializations table. LARAVEL_SPECIALTIES mirrors the live database's
# real name_ar column — this service has no live DB connection, so a real
# Laravel-side change needs a matching manual edit here.
#
# 21 rows total: the original 11
# (healix-backend/database/seeders/SpecializationsTableSeeder.php) plus
# the 10 added later by
# healix-backend/database/migrations/2026_08_17_000002_backfill_specialization_codes_and_missing_specialties.php
# ("the specialties the AI service ... can recommend but that were never
# seeded"). Every string below was copied character-for-character from
# that Laravel source — SpecializationResolver::resolve() does an exact
# (case-folded, not diacritic-folded) string match against name_ar, so a
# transcription slip here (a missing hamza, a stray diacritic) would
# silently never match a real row.
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
        "أمراض الجهاز الهضمي",  # Gastroenterology
        "طب عام",  # General Medicine
        "أمراض الدم",  # Hematology
        "الأمراض المُعدية",  # Infectious Disease
        "أنف وأذن وحنجرة",  # Otolaryngology (ENT)
        "أمراض الصدر والجهاز التنفسي",  # Pulmonology
        "الروماتيزم والمفاصل",  # Rheumatology
        "طب الأسرة",  # Family Medicine
        "الغدد الصماء",  # Endocrinology
        "أمراض النساء والتوليد",  # Obstetrics and Gynecology (own row, distinct from Gynecology above)
    }
)

# Translates every KB specialty EXCEPT GENERAL_PRACTICE (handled
# separately below) into one of LARAVEL_SPECIALTIES. Now that Laravel has
# grown to 21 real specialties (see LARAVEL_SPECIALTIES's own comment),
# most KB specialty strings map to a genuine matching row rather than an
# approximated "closest" one — only علاج طبيعي (Physical Therapy) and
# حساسية (Allergy) still have no real Laravel row and keep their original
# closest-specialty fallback.
SPECIALTY_MAP: dict[str, str] = {
    # Wording-only: same specialty, KB uses a bare adjective/noun.
    "أطفال": "طب الأطفال",
    "جلدية": "الأمراض الجلدية",
    "نسائية": "أمراض النساء والولادة",
    "عظمية": "جراحة العظام",
    "عصبية": "الأمراض العصبية",
    "عينية": "طب العيون",
    "قلبية": "أمراض القلب",
    "مسالك بولية": "المسالك البولية",
    # Now a real, direct Laravel specialty (added by the migration above) —
    # previously approximated onto a "closest" specialty before that row existed.
    "معدية": "الأمراض المُعدية",
    "أنف وأذن وحنجرة": "أنف وأذن وحنجرة",
    "هضمية": "أمراض الجهاز الهضمي",
    "روماتيزم": "الروماتيزم والمفاصل",
    "غدد صماء": "الغدد الصماء",
    "صدرية": "أمراض الصدر والجهاز التنفسي",
    "دموية": "أمراض الدم",
    # Internal Medicine still has no exact Laravel row; "طب عام" (General
    # Medicine, added by the same migration) is a materially better fit
    # than the old Cardiology approximation.
    "باطنية": "طب عام",
    # Still no direct Laravel row for either — closest-specialty fallback unchanged.
    "علاج طبيعي": "جراحة العظام",
    "حساسية": "الأمراض الجلدية",
    # GENERAL_PRACTICE itself -> Laravel's real "General Medicine" row
    # (added by the same migration as the entries above). Deliberate
    # product decision, not the earlier accidental self-mapping this
    # module used to have: an insufficient-information turn, or a KB
    # entry with no other specialty, now routes to a real bookable
    # general-medicine doctor instead of a plain referral phrase.
    GENERAL_PRACTICE: GENERAL_PRACTICE,
}

assert set(SPECIALTY_MAP.values()) <= LARAVEL_SPECIALTIES, (
    "SPECIALTY_MAP has a value outside LARAVEL_SPECIALTIES — every "
    "mapped value must be one of Laravel's real specializations.name_ar "
    "rows, or a future doctor-matching lookup against that table would "
    "silently never match."
)

# Defensive fallback only — every currently-known KB specialty string,
# including GENERAL_PRACTICE itself, now has a real SPECIALTY_MAP target
# (see LARAVEL_SPECIALTIES's own comment), so this is not reached by any
# combination rag/knowledge_base/*.json's current 19 specialty strings can
# produce. Kept for a future KB entry whose specialties list uses a
# string not yet added to SPECIALTY_MAP — better than KeyError-ing or
# silently mapping to an arbitrary Laravel specialty.
GENERAL_REFERRAL_PHRASE = "استشر طبيب أو راجع أقرب مركز رعاية أولية"


def _resolve_laravel_specialty(kb_specialties: list[str]) -> str:
    """KB specialty string(s) -> the value ChatResponse.specialty
    actually carries. Dedupes (two KB specialties can map to the same
    Laravel value)."""
    mapped = [SPECIALTY_MAP[s] for s in kb_specialties if s in SPECIALTY_MAP]
    if not mapped:
        return GENERAL_REFERRAL_PHRASE

    seen: set[str] = set()
    deduped = [value for value in mapped if not (value in seen or seen.add(value))]
    return "، ".join(deduped)


def route_specialty(state: HealixState) -> dict[str, Any]:
    diagnosis = state.get("diagnosis") or {}
    differential = diagnosis.get("differential") or []

    if diagnosis.get("status") != "differential" or not differential:
        return {
            "specialty": GENERAL_PRACTICE,
            "specialty_laravel": _resolve_laravel_specialty([GENERAL_PRACTICE]),
        }

    top_candidate = differential[0]  # already sorted by match_score, descending
    specialties = top_candidate.get("specialties") or [GENERAL_PRACTICE]

    return {
        "specialty": "، ".join(specialties),
        "specialty_laravel": _resolve_laravel_specialty(specialties),
    }
