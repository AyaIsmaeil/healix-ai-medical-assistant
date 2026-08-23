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
# real name_ar column (11 rows) — this service has no live DB connection,
# so a real Laravel-side change needs a matching manual edit here.
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
        "طب عام",  # General Practice
        "انف وأذن وحنجرة",  # ENT
        "امراض الصدر والجهاز التنفسي",  # Pulmonology
        "طب الغدد الصماء",  # Endocrinology
        "امراض النساء والتوليد",  # Obstetrics
        "طب العيون",  # Ophthalmology (duplicate, but included in the original list)""
        "الطب النفسي",  # Psychiatry (duplicate, but included in the original list)

    }
)

# Translates every KB specialty EXCEPT GENERAL_PRACTICE (handled
# separately below) into one of LARAVEL_SPECIALTIES.
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
    # No Laravel equivalent — mapped to the closest real specialty.
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
    "طب عام": "طب عام",
    "انف وأذن وحنجرة": "انف وأذن وحنجرة",
    "امراض الصدر والجهاز التنفسي": "امراض الصدر والجهاز التنفسي",
    "طب الغدد الصماء": "طب الغدد الصماء",
    "امراض النساء والتوليد": "امراض النساء والتوليد",
    "طب العيون": "طب العيون",
    "الطب النفسي": "الطب النفسي",
}

assert set(SPECIALTY_MAP.values()) <= LARAVEL_SPECIALTIES, (
    "SPECIALTY_MAP has a value outside LARAVEL_SPECIALTIES — every "
    "mapped value must be one of Laravel's real specializations.name_ar "
    "rows, or a future doctor-matching lookup against that table would "
    "silently never match."
)

# Laravel has no general-practice specialty to map GENERAL_PRACTICE onto
# (forcing it onto e.g. Pediatrics would misroute adult patients), so a
# turn whose KB specialties reduce to nothing gets this instead.
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
