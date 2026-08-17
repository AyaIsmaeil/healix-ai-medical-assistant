"""Canonical Arabic symptom vocabulary — single source of truth for symptom names.

Every symptom name used anywhere in this service (rule definitions in
rules/red_flags.py, extract_symptoms output, state["symptoms"]) must appear
here. Matching elsewhere is membership/equality against this set, so a name
that is absent — or present in a different spelling — silently fails to
match rather than raising. That failure mode is the reason this file exists.

Entries are stored NORMALIZED (see rules/crisis.normalize): alef variants
folded to ا, ة folded to ه, diacritics and tatweel stripped, elongation and
whitespace collapsed. Callers should normalize before comparing rather than
assuming the LLM produced a particular spelling. Source spellings below are
written naturally and normalized at import, so this file stays readable.

--- Provenance ---

Built in two stages, both citation-backed rather than invented:

1. The emergency symptoms rules/red_flags.py requires — each red-flag rule
   cites the guideline/scoring tool it is drawn from.
2. The symptoms rag/knowledge_base/'s disease entries require — each entry
   cites a source (WHO/NICE/CDC/GINA/ICHD-3, etc.) and is marked
   translation_reviewed. See rag/coverage.py, which reports the gap between
   what a knowledge-base entry needs and what's already here.

This supersedes an earlier plan to reconcile against a single external
"approved 52-symptom list": that number had no verifiable source of its
own, so it was dropped in favor of building the vocabulary bottom-up from
entries that do cite one. EXPECTED_SYMPTOM_COUNT below is not a ceiling
copied from anywhere external — it is this file's own running count. Adding
more entries the same way (add a rag/knowledge_base/ entry, run
rag/coverage.py, add what it reports missing here, bump the count) is
expected and fine; editing the count alone, without entries to back it, is
exactly what the two tests below exist to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

from rules.crisis import normalize

# EXPECTED_SYMPTOM_COUNT is a manually-maintained checkpoint, not a fixed
# external target: it must equal len(CANONICAL_SYMPTOMS) whenever this file
# is in a reconciled state, bumped deliberately alongside any addition made
# through the reviewed process described above — never edited by itself.
#
# Held in place from both sides by tests/unit/test_vocabulary_completeness.py:
#   - test_vocabulary_never_exceeds_the_approved_count (always runs) catches
#     an addition to _SYMPTOM_NAMES below without a matching bump here.
#   - test_vocabulary_matches_the_approved_symptom_count (marked
#     vocabulary_completeness, deselected by default — see pytest.ini)
#     catches this number being raised without the entries to back it.
#
# Run `python -m vocabulary.symptoms` to see the current count.
EXPECTED_SYMPTOM_COUNT = 121

# Written in natural Arabic spelling for readability; normalized below.
# Grouped by where each name came from: the red-flag rule that introduced
# it (rules/red_flags.py), or the rag/knowledge_base/ disease entry that
# first needed it (several later entries reuse a name added by an earlier
# one — see each disease's own JSON file for its full symptom list).
_SYMPTOM_NAMES: tuple[str, ...] = (
    # acs_chest_pain
    "ألم في الصدر",
    "ألم منتشر للذراع أو الفك",
    "تعرق غزير",
    "تقيؤ",
    "ضيق تنفس",
    "غثيان",

    # loss_of_consciousness
    "فقدان الوعي",

    # stroke_fast
    "تدلي في الوجه",
    "تلعثم مفاجئ في الكلام",
    "ضعف مفاجئ في نصف الجسم",
    "فقدان التوازن المفاجئ",
    "فقدان مفاجئ للرؤية",

    # bacterial_meningitis
    "حمى",
    "تغير مفاجئ في مستوى الوعي",
    "تيبس الرقبة",
    "حساسية للضوء",
    "صداع شديد ومفاجئ",

    # anaphylaxis
    "تورم في الوجه أو الحلق",
    "إغماء أو دوخة شديدة",
    "طفح جلدي منتشر",

    # sepsis
    "تخليط ذهني مفاجئ",
    "تسارع في التنفس",

    # gi_bleed
    "براز أسود",
    "تقيؤ دم",
    "دم في البراز",

    # pulmonary_embolism
    "تورم في ساق واحدة",
    "خفقان القلب",

    # dka
    "عطش شديد",
    "تبول متكرر",
    "تنفس سريع وعميق",
    "نعاس أو تشوش ذهني",

    # ectopic_pregnancy
    "نزيف مهبلي",
    "تأخر الدورة الشهرية",

    # --- rag/knowledge_base/ (rag/coverage.py) ---

    # Influenza
    "ألم عضلي",
    "التهاب حلق",
    "تعب",
    "سعال",
    "سيلان أنف",
    "صداع",

    # Hypertension
    "دوخة",

    # Urinary Tract Infection
    "حرقة عند التبول",
    "ألم أسفل البطن",

    # Streptococcal Pharyngitis
    "ألم حلق شديد ومفاجئ",
    "صعوبة بلع",
    "تورم غدد الرقبة",

    # Acute Gastroenteritis
    "إسهال",
    "ألم بطن",

    # Type 2 Diabetes
    "نقص وزن غير مبرر",
    "تشوش رؤية",

    # Migraine
    "صداع نابض من جهة واحدة",
    "حساسية للصوت",

    # Asthma
    "أزيز صدر",
    "ضيق بالصدر",

    # Iron Deficiency Anaemia
    "شحوب",

    # Rheumatoid Arthritis
    "ألم مفاصل",
    "تورم مفاصل",
    "تيبس صباحي",

    # Acute Sinusitis
    "احتقان أنف",
    "إفرازات أنفية",
    "ألم بالوجه",
    "فقدان حاسة الشم",

    # Irritable Bowel Syndrome
    "انتفاخ",
    "إمساك",
    # Location-granularity pass on "ألم بطن" (CLAUDE.md > Known
    # limitations / rag_retrieve's sex-gating docstring): NICE CG61
    # explicitly documents IBS pain's site as "can be anywhere in the
    # abdomen... and whether this varies" — a genuine diagnostic feature,
    # not an absence of information, so this is its own distinct term
    # rather than the bare generic "ألم بطن".
    "ألم بطن معمم",

    # Urticaria
    "طفح جلدي",
    "حكة",
    "تورم موضعي",

    # Conjunctivitis
    "احمرار العين",
    "حكة بالعين",
    "إفرازات عينية",
    "دموع زائدة",

    # Benign Paroxysmal Positional Vertigo
    "دوار",

    # Acute Otitis Media
    "ألم أذن",
    "فقدان سمع",

    # Kidney Stones
    "ألم شديد بالخاصرة",
    "دم في البول",

    # Atopic Dermatitis
    "جفاف الجلد",
    "احمرار",

    # Tonsillitis
    "ألم حلق شديد",
    "تورم اللوزتين",

    # Tendinitis
    "ألم موضعي عند الحركة",
    "تيبس",

    # Measles
    "بقع بيضاء بالفم",

    # Mumps
    "تورم الغدد اللعابية",

    # Hand, Foot, and Mouth Disease
    "تقرحات الفم",

    # Infectious Mononucleosis
    "تعب شديد",

    # Pinworm Infection
    "حكة شرجية",
    "اضطراب نوم",
    "تهيج",

    # Roseola
    "حمى مرتفعة مفاجئة",

    # Viral Pharyngitis
    "حمى خفيفة",

    # Scabies
    "حكة شديدة ليلية",

    # Vaginal Candidiasis
    "حكة مهبلية",
    "إفرازات مهبلية",

    # Dysmenorrhea
    "ألم أسفل الظهر",

    # Polycystic Ovary Syndrome
    "اضطراب الدورة الشهرية",
    "حب الشباب",
    "زيادة وزن",
    "نمو شعر زائد",

    # Hepatitis A
    "يرقان",
    "بول داكن",
    # Location-granularity pass on "ألم بطن": CDC's Hepatitis A Clinical
    # Overview specifies "right upper quadrant abdominal pain" during the
    # prodromal phase — explicit source support for "upper", not the bare
    # generic term (see rag_retrieve's module docstring, "Sex-specific
    # gating" section, for the related investigation this was found
    # alongside).
    "ألم أعلى البطن",

    # Cutaneous Leishmaniasis
    "تقرح جلدي",

    # Gout
    "ألم مفاصل شديد ومفاجئ",

    # Acute Musculoskeletal Strain
    "ألم موضعي عضلي",
    "صعوبة حركة",

    # Gastroesophageal Reflux Disease
    "حرقة في الصدر",
    "ارتجاع حمضي",
    "طعم حامض أو مر بالفم",

    # Acute Bronchitis
    "بلغم",

    # Otitis Externa
    "حكة بالأذن",
    "إفرازات من الأذن",

    # Allergic Rhinitis
    "عطس متكرر",

    # Impetigo
    "قشور عسلية اللون على الجلد",

    # Pediculosis Capitis
    "حكة فروة الرأس",
    "احساس بحركة بفروة الرأس",

    # Rubella
    "تورم غدد خلف الأذن",

    # Herpes Zoster
    "حرقان أو وخز بالجلد",

    # Peptic Ulcer Disease
    "فقدان شهية",

    # Bacterial Vaginosis
    "رائحة مهبلية كريهة",

    # Tension-Type Headache
    "صداع ضاغط من الجهتين",
)

CANONICAL_SYMPTOMS: frozenset[str] = frozenset(
    normalize(name) for name in _SYMPTOM_NAMES
)


def is_canonical(name: str) -> bool:
    """True if `name` is in the canonical vocabulary, ignoring spelling variation."""
    return normalize(name) in CANONICAL_SYMPTOMS


@dataclass(frozen=True)
class VocabularyCompleteness:
    present: int
    expected: int

    @property
    def missing(self) -> int:
        return max(0, self.expected - self.present)

    @property
    def is_complete(self) -> bool:
        return self.present >= self.expected

    def describe(self) -> str:
        if self.is_complete:
            return (
                f"Symptom vocabulary: {self.present}/{self.expected} entries — complete.\n"
                "Count alone does not prove wording — that still rests on each\n"
                "entry's citation (a red-flag rule's guideline, or a\n"
                "rag/knowledge_base/ entry's source) being correct, not on this\n"
                "number matching."
            )
        return (
            f"Symptom vocabulary: {self.present}/{self.expected} entries — "
            f"{self.missing} MISSING.\n"
            "\n"
            "Consequences while this stands:\n"
            f"  - schemas/symptoms.py generates a {self.present}-value enum, so the\n"
            "    model cannot report a symptom that is not in it.\n"
            "  - golden test cases referencing a missing symptom will fail at\n"
            "    validation, not produce a wrong triage result.\n"
            "  - evaluation numbers computed now measure a truncated vocabulary.\n"
            "\n"
            "Fix by adding entries through the process in this file's module\n"
            "docstring (rag/knowledge_base/ + rag/coverage.py, or a red-flag\n"
            "rule's own citation) and bumping EXPECTED_SYMPTOM_COUNT to match —\n"
            "not by inventing plausible names. See CLAUDE.md > Symptom vocabulary."
        )


def check_completeness() -> VocabularyCompleteness:
    """Report entries present vs. expected, so the gap stays visible."""
    return VocabularyCompleteness(
        present=len(CANONICAL_SYMPTOMS), expected=EXPECTED_SYMPTOM_COUNT
    )


if __name__ == "__main__":
    print(check_completeness().describe())
