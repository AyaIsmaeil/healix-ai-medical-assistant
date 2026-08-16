"""Deterministic red-flag detection over extracted symptoms — no LLM calls.

Runs after extract_symptoms, on state["symptoms"] (CLAUDE.md > Graph flow:
check_red_flags follows extract_symptoms; it does not see raw patient
text — that's rules/crisis.py's job, earlier in the graph). Per CLAUDE.md
> Non-negotiable safety rules, this layer is never removed, weakened, or
made conditional on LLM output — the check_red_flags node combines this
with an LLM check via OR, it does not replace this with one.

Every symptom name referenced below must exist in the canonical
vocabulary (vocabulary/symptoms.py). That is enforced at import time by
_validate_rule_symptoms_are_canonical: a typo or a renamed symptom makes
this module fail to load, rather than leaving a rule that silently never
fires. Matching is done on normalized names (rules/crisis.normalize) on
both sides, so a spelling variation out of extract_symptoms — أ vs ا,
ة vs ه — does not cause a miss either.

Citations on each rule name the guideline/scoring tool it is drawn from,
as a starting point for the project's formal documentation. They are not
a substitute for that documentation step: verify the exact edition/number
against the current published version before citing formally, since these
guidelines are periodically revised and I can't confirm live versions here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rules.crisis import normalize
from state import Symptom
from vocabulary.symptoms import CANONICAL_SYMPTOMS

# Arabic proclitics that attach directly to the front of a word: the definite
# article ال, and single-letter particles و/ف/ب/ك/ل (and pairs of them, e.g.
# لل, وب). Allowing these is what makes "بالسكري" and "السكري" match the
# keyword "سكري" while "عسكرية" (military) still does not — the boundary is
# anchored before the proclitic, so an arbitrary preceding letter blocks it.
_PROCLITICS = r"(?:[وفبكل]{0,2}(?:ال)?)"


def _compile_keyword_matcher(keywords: frozenset[str]) -> re.Pattern[str] | None:
    """Build a word-boundary-aware matcher over normalized text.

    Naive substring containment is wrong here in both directions: "سكر"
    appears inside "سكرتير"/"سكران", and "سكري" appears inside "عسكرية",
    so a military-service note would silently lower a cardiac threshold.
    Strict \\b is wrong too — it rejects "السكري" and "بالسكري", which is
    the more dangerous direction, since a missed chronic condition means a
    red flag that should have fired doesn't.

    Keywords are normalized here so callers may write them naturally.

    Known limitation: suffixed forms are not matched (e.g. the nisba
    "السكرية" won't match "سكري"). Widening the trailing boundary to allow
    arbitrary suffixes would reopen the "سكرتير" class of false positive,
    so the safer under-match is kept until there's real record data to
    tune against.
    """
    if not keywords:
        return None
    alternatives = "|".join(re.escape(normalize(k)) for k in sorted(keywords))
    return re.compile(rf"(?<!\w){_PROCLITICS}(?:{alternatives})(?!\w)")


def _compile_english_keyword_matcher(keywords: frozenset[str]) -> re.Pattern[str] | None:
    """Build a case-insensitive SUBSTRING matcher over English keyword roots.

    Deliberately NOT _compile_keyword_matcher's word-boundary-aware
    approach. That machinery exists because Arabic's short roots collide
    with unrelated real words (سكر inside سكرتير, سكري inside عسكرية) —
    none of the English roots this is used for have an equivalent
    collision (no unrelated common English word contains "diabet",
    "neoplas", "immunosuppress", "transplant", or "graft"), so a
    boundary requirement would only cost recall for no safety benefit.
    Concretely: a word-START boundary before "graft" would MISS
    "allograft rejection" — DrugCentral's own real indication phrasing
    for transplant immunosuppressants (verified directly against
    drugcentral.org, not assumed) — since "graft" sits mid-word there,
    not at a word start. Plain substring containment catches it
    correctly; a stricter regex would silently under-match exactly the
    real-world phrasing this exists to catch.

    Not run through rules.crisis.normalize() first — verified directly
    that doing so would be a no-op for English text (that function is
    Arabic diacritic/hamza/digit-fold logic with no case-folding at
    all), so this does its own .lower() rather than leaning on
    normalize() for a job it was never built for.
    """
    if not keywords:
        return None
    alternatives = "|".join(re.escape(keyword.lower()) for keyword in sorted(keywords))
    return re.compile(alternatives)


# --- generic/specific symptom subsumption ---------------------------------
#
# A handful of RedFlagRule terms are written in generic form (e.g. "حمى"),
# but vocabulary/symptoms.py separately holds more specific canonical
# siblings for the same underlying symptom — added later, for
# rag/knowledge_base entries that needed the specific form (e.g. "حمى
# مرتفعة مفاجئة" for Roseola, "حمى خفيفة" for Viral Pharyngitis).
# extract_symptoms has no reason to prefer the generic spelling when a
# specific one describes what the patient actually said — the schema
# enum accepts either equally — so a rule written against only the
# generic term can silently never fire against a real extraction that
# correctly picked the specific one. Surfaced by tests/golden/cases.json's
# emergency_03: "حمى عالية" (high fever) extracted to "حمى مرتفعة
# مفاجئة", not "حمى", and bacterial_meningitis's requirement.all_of=
# {"حمى"} never matched despite the patient clearly having a fever.
#
# _SUBSUMES is the explicit, hand-authored fix: generic term -> the
# specific siblings that should ALSO satisfy a requirement written
# against the generic term. Deliberately not a fuzzy/substring matcher —
# every pair was checked individually and only added where the specific
# term is unambiguously "the same symptom, described more precisely" —
# never where the "specific" term actually encodes a DIFFERENT severity
# or extent that the rule's own wording depends on.
#
# Audited and DELIBERATELY EXCLUDED below, so this list isn't silently
# reopened later by someone re-deriving it without this reasoning:
#   - "تقيؤ" -> "تقيؤ دم" (acs_chest_pain, dka): vomiting blood is
#     technically vomiting, but "تقيؤ دم" is authored for gi_bleed, a
#     different clinical scenario — left as a separate judgment call,
#     not folded in here.
#   - "فقدان الوعي" <- "تغير مفاجئ في مستوى الوعي": a mere CHANGE in
#     consciousness level is not the same as LOSS of it — merging would
#     wrongly broaden loss_of_consciousness to a lesser presentation.
#   - "إغماء أو دوخة شديدة" <- bare "دوخة": the compound term's "شديدة"
#     (severe) qualifier IS the signal anaphylaxis/sepsis actually need;
#     plain dizziness would weaken both rules.
#   - "طفح جلدي منتشر" <- bare "طفح جلدي": "منتشر" (widespread) is the
#     actual anaphylaxis-relevant severity signal; a localized rash
#     isn't.
#   - "تيبس الرقبة": no more-specific sibling exists; the vocabulary's
#     other تيبس* entries are unrelated body locations (joints,
#     general), not neck-specific variants.
#   - "تورم في الوجه أو الحلق" vs other swelling terms (joints/neck-
#     glands/localized/salivary): all different body locations, not
#     face-or-throat variants.
#   - "تأخر الدورة الشهرية" <- "اضطراب الدورة الشهرية": wrong direction
#     — ectopic_pregnancy already uses the specific term; the
#     vocabulary's sibling is BROADER and would over-include if
#     accepted.
#   - "ألم في الصدر", "ضيق تنفس": checked for siblings; none exist
#     ("ضيق بالصدر"/"تسارع في التنفس"/"تنفس سريع وعميق" are different
#     symptom qualities — tightness, rate, depth — not more-specific
#     phrasings of the same concept).
_SUBSUMES: dict[str, frozenset[str]] = {
    "حمى": frozenset({"حمى مرتفعة مفاجئة", "حمى خفيفة"}),
    "ألم بطن": frozenset({"ألم أسفل البطن", "ألم أعلى البطن", "ألم بطن معمم"}),
}

_SUBSUMES_N: dict[str, frozenset[str]] = {
    normalize(generic): frozenset(normalize(specific) for specific in specifics)
    for generic, specifics in _SUBSUMES.items()
}


def _term_matches(term: str, present: frozenset[str]) -> frozenset[str]:
    """The normalized symptom name(s) in `present` satisfying `term` —
    `term` itself if it's directly present, else any of its _SUBSUMES_N
    siblings actually present, else empty (unsatisfied).

    Returns the symptom(s) actually found, not the generic term the
    patient never literally said — so RedFlagMatch.matched_symptoms
    (built from this) reports what was really extracted, e.g. "حمى
    مرتفعة مفاجئة" rather than a "حمى" the patient's own wording didn't use.
    """
    if term in present:
        return frozenset({term})
    return _SUBSUMES_N.get(term, frozenset()) & present


@dataclass(frozen=True)
class SymptomRequirement:
    """A combination a rule needs to see in the confirmed-symptom set.

    all_of: every one of these must be present (the "symptom X" part).
    any_of: at least one of these must also be present (the "together
        with any of Y, Z" part). Leave empty for a plain single/combined
        AND-only requirement with no "any of" clause. Leaving all_of
        empty instead turns this into "any single one of any_of is
        independently sufficient" (e.g. the FAST stroke signs).

    Names are written in natural Arabic spelling and normalized once at
    construction; matching uses the normalized copies. The originals are
    kept so the vocabulary guard can report a bad name as the author
    actually typed it.
    """

    all_of: frozenset[str] = frozenset()
    any_of: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_all_of_n", frozenset(normalize(n) for n in self.all_of))
        object.__setattr__(self, "_any_of_n", frozenset(normalize(n) for n in self.any_of))

    def satisfied_by(self, present: frozenset[str]) -> frozenset[str] | None:
        """Return the normalized symptoms satisfying this requirement, or None.

        `present` must already be normalized — check_red_flags does that.
        Each required term is checked via _term_matches, so a term
        written generically (e.g. "حمى") is also satisfied by any of its
        _SUBSUMES_N specific siblings actually present — see that
        mapping's own module-level comment for the full reasoning and the
        pairs deliberately left out of it.
        """
        all_of: frozenset[str] = self._all_of_n  # type: ignore[attr-defined]
        any_of: frozenset[str] = self._any_of_n  # type: ignore[attr-defined]

        matched_all: set[str] = set()
        for term in all_of:
            hit = _term_matches(term, present)
            if not hit:
                return None
            matched_all |= hit

        any_hit: set[str] = set()
        for term in any_of:
            any_hit |= _term_matches(term, present)
        if any_of and not any_hit:
            return None

        return frozenset(matched_all) | frozenset(any_hit)

    def all_names(self) -> frozenset[str]:
        """Every name this requirement references, in its authored spelling."""
        return self.all_of | self.any_of


@dataclass(frozen=True)
class RedFlagRule:
    id: str
    category: str
    reason_ar: str  # doctor-report-facing rationale, standard medical Arabic
    source: str  # guideline/tool this rule is drawn from
    requirement: SymptomRequirement
    # A chronic condition mentioned in the record summary can lower the bar:
    # if any chronic_condition_keyword is found there, chronic_requirement
    # is checked instead of (as a fallback to) the base requirement.
    chronic_condition_keywords: frozenset[str] = frozenset()
    # English-language counterpart, checked via a SEPARATE mechanism
    # (_compile_english_keyword_matcher — substring, not word-boundary
    # regex) — never a translation of the Arabic set, an independently-
    # sourced one. Exists because Laravel's medical_record_summary (built
    # from a patient's real chronic_diseases/current_medications) stores
    # English DrugCentral-standard condition/drug names, which the
    # Arabic-only matcher above cannot recognize at all — see CLAUDE.md >
    # Known limitations for the full investigation behind this field.
    chronic_condition_keywords_en: frozenset[str] = frozenset()
    chronic_requirement: SymptomRequirement | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_chronic_re", _compile_keyword_matcher(self.chronic_condition_keywords)
        )
        object.__setattr__(
            self,
            "_chronic_re_en",
            _compile_english_keyword_matcher(self.chronic_condition_keywords_en),
        )

    def mentions_chronic_condition(self, medical_record_summary: str) -> bool:
        """True if the record summary mentions one of this rule's chronic
        keywords, in EITHER language — checked independently against each
        set, never translated between them (see _compile_english_keyword_matcher's
        own docstring for why the two use different matching mechanisms)."""
        arabic_matcher: re.Pattern[str] | None = self._chronic_re  # type: ignore[attr-defined]
        english_matcher: re.Pattern[str] | None = self._chronic_re_en  # type: ignore[attr-defined]

        if arabic_matcher is not None and arabic_matcher.search(normalize(medical_record_summary)):
            return True
        if english_matcher is not None and english_matcher.search(medical_record_summary.lower()):
            return True
        return False


@dataclass(frozen=True)
class RedFlagMatch:
    rule_id: str
    category: str
    reason_ar: str
    source: str
    matched_symptoms: frozenset[str]
    lowered_by_chronic_condition: bool


# --- rule set ----------------------------------------------------------
#
# Well-established emergency presentations only, each drawn from a named
# triage protocol or clinical guideline. See module docstring re: citation
# verification.

RED_FLAG_RULES: tuple[RedFlagRule, ...] = (
    # Source: ESC Guidelines for the management of acute coronary syndromes
    # (European Society of Cardiology); ACC/AHA Guideline for the Evaluation
    # and Diagnosis of Chest Pain. Threshold lowered for diabetic patients:
    # atypical/silent myocardial ischemia in diabetes is well documented in
    # ACC/AHA guidance and ADA clinical literature.
    RedFlagRule(
        id="acs_chest_pain",
        category="cardiac",
        reason_ar=(
            "ألم الصدر المترافق مع أعراض إضافية (كضيق التنفس أو التعرق الغزير "
            "أو الألم المنتشر للذراع أو الفك) قد يشير إلى متلازمة الشريان "
            "التاجي الحادة، وهي حالة إسعافية تستوجب تقييمًا فوريًا."
        ),
        source=(
            "ESC Guidelines for the management of acute coronary syndromes; "
            "ACC/AHA Chest Pain Guideline. Chronic-lowering rationale: "
            "atypical/silent MI presentation in diabetic patients "
            "(ACC/AHA guidance; ADA literature) — verify current edition."
        ),
        requirement=SymptomRequirement(
            all_of=frozenset({"ألم في الصدر"}),
            any_of=frozenset({
                "ضيق تنفس",
                "تعرق غزير",
                "ألم منتشر للذراع أو الفك",
                "غثيان",
                "تقيؤ",
            }),
        ),
        # Keyword review: both terms are matched word-boundary-aware on
        # normalized text (_compile_keyword_matcher), so the bare سكر no longer
        # hits سكرتير / سكران, and سكري no longer hits عسكرية (military).
        # سكر is kept because it carries real recall in record summaries
        # (سكر الدم, ارتفاع السكر) that سكري alone would miss.
        chronic_condition_keywords=frozenset({"سكري", "سكر"}),
        # HIGH confidence, unlike sepsis's English set below: "diabet"
        # verified directly against the app's real, currently-stored
        # values (Laravel lang/en/enums.php's ddi_condition picker —
        # "Diabetes mellitus", "Diabetes mellitus type 1", "Diabetes
        # mellitus type 2" — all three contain this root). One root
        # keyword, not three separate phrases, mirroring the Arabic
        # side's own "prefer the shorter root for broader recall"
        # choice above.
        chronic_condition_keywords_en=frozenset({"diabet"}),
        chronic_requirement=SymptomRequirement(all_of=frozenset({"ألم في الصدر"})),
    ),
    # Source: ESC Guidelines for the diagnosis and management of syncope
    # (European Society of Cardiology) — verify current edition.
    RedFlagRule(
        id="loss_of_consciousness",
        category="neuro",
        reason_ar=(
            "فقدان الوعي، ولو لفترة وجيزة، يستوجب تقييمًا إسعافيًا فوريًا "
            "لاستبعاد أسباب قلبية أو عصبية خطيرة."
        ),
        source="ESC Guidelines for the diagnosis and management of syncope — verify current edition.",
        requirement=SymptomRequirement(all_of=frozenset({"فقدان الوعي"})),
    ),
    # Source: FAST / BE-FAST stroke recognition tool (American Stroke
    # Association public education campaign; Cincinnati Prehospital Stroke
    # Scale, Kothari et al. 1997) — any one sign alone is independently
    # sufficient, hence all_of is empty and any_of carries the whole rule.
    RedFlagRule(
        id="stroke_fast",
        category="neuro",
        reason_ar=(
            "ظهور أي علامة من علامات السكتة الدماغية المفاجئة (تدلي الوجه، "
            "ضعف مفاجئ في نصف الجسم، تلعثم مفاجئ في الكلام، فقدان مفاجئ "
            "للرؤية أو للتوازن) يستوجب تقييمًا إسعافيًا فوريًا — الوقت عامل "
            "حاسم."
        ),
        source="FAST / BE-FAST stroke recognition (American Stroke Association; Cincinnati Prehospital Stroke Scale) — verify current edition.",
        requirement=SymptomRequirement(
            any_of=frozenset({
                "تدلي في الوجه",
                "ضعف مفاجئ في نصف الجسم",
                "تلعثم مفاجئ في الكلام",
                "فقدان مفاجئ للرؤية",
                "فقدان التوازن المفاجئ",
            }),
        ),
    ),
    # Source: NICE guideline — Meningitis (bacterial) and meningococcal
    # disease: recognition, diagnosis and management (originally CG102,
    # updated as NG240) — verify current edition.
    RedFlagRule(
        id="bacterial_meningitis",
        category="infectious_cns",
        reason_ar=(
            "الحمى المترافقة مع علامات تنذر بالتهاب السحايا (كتيبس الرقبة، "
            "أو الصداع الشديد المفاجئ، أو حساسية الضوء، أو تغير مستوى "
            "الوعي) حالة إسعافية تستوجب تقييمًا وعلاجًا فوريين."
        ),
        source="NICE guideline — Meningitis (bacterial) and meningococcal disease: recognition, diagnosis and management — verify current edition.",
        requirement=SymptomRequirement(
            all_of=frozenset({"حمى"}),
            any_of=frozenset({
                "تيبس الرقبة",
                "صداع شديد ومفاجئ",
                "حساسية للضوء",
                "تغير مفاجئ في مستوى الوعي",
            }),
        ),
    ),
    # Source: World Allergy Organization (WAO) Anaphylaxis Guidance, 2020
    # update; NICE CG134 — Anaphylaxis: assessment and referral after
    # emergency treatment — verify current edition.
    RedFlagRule(
        id="anaphylaxis",
        category="allergic",
        reason_ar=(
            "تورم الوجه أو الحلق مع أعراض إضافية (كضيق التنفس أو الدوخة "
            "الشديدة أو الطفح الجلدي المنتشر) قد يشير إلى تفاعل تحسسي شديد "
            "(صدمة تأقية)، وهي حالة تهدد الحياة وتستوجب تدخلاً إسعافيًا "
            "فوريًا."
        ),
        source="World Allergy Organization Anaphylaxis Guidance; NICE CG134 — verify current edition.",
        requirement=SymptomRequirement(
            all_of=frozenset({"تورم في الوجه أو الحلق"}),
            any_of=frozenset({"ضيق تنفس", "إغماء أو دوخة شديدة", "طفح جلدي منتشر"}),
        ),
    ),
    # Source: Surviving Sepsis Campaign guidelines (SCCM/ESICM); qSOFA /
    # Sepsis-3 definitions (Singer et al., JAMA 2016); NICE NG51 — Sepsis:
    # recognition, diagnosis and early management — verify current edition.
    # Threshold lowered for immunocompromised patients, for whom fever may
    # be the only presenting sign.
    RedFlagRule(
        id="sepsis",
        category="sepsis",
        reason_ar=(
            "الحمى المترافقة مع تخليط ذهني مفاجئ أو تسارع في التنفس أو "
            "دوخة شديدة قد تشير إلى إنتان (تعفن الدم)، وهي حالة إسعافية "
            "تستوجب تقييمًا فوريًا."
        ),
        source=(
            "Surviving Sepsis Campaign guidelines; qSOFA / Sepsis-3 "
            "(Singer et al., JAMA 2016); NICE NG51. Chronic-lowering "
            "rationale: fever may be the only sign in immunocompromised "
            "patients — verify current edition."
        ),
        requirement=SymptomRequirement(
            all_of=frozenset({"حمى"}),
            any_of=frozenset({"تخليط ذهني مفاجئ", "تسارع في التنفس", "إغماء أو دوخة شديدة"}),
        ),
        chronic_condition_keywords=frozenset({"مثبطات المناعة", "علاج كيماوي", "سرطان", "زراعة أعضاء"}),
        # MEDIUM confidence English roots, added anyway per rules/crisis.py's
        # own "bias toward false positives: a missed crisis is unacceptable,
        # a false positive costs one gentle redirect message" principle,
        # applied here to the same asymmetry (a missed chronic-lowering
        # signal risks an unrecognized sepsis presentation; a keyword that
        # never matches real data costs exactly zero, not negative).
        #
        # KNOWN COVERAGE GAP, not a bug here: Laravel's own chronic_diseases
        # condition picker (lang/en/enums.php's ddi_condition list, 46
        # entries — checked directly) offers NO cancer, transplant,
        # chemotherapy, or immunosuppression option at all today. That is a
        # Laravel-side data-entry gap, out of scope for this Python project
        # to fix. Until/unless that picker is extended, these roots are
        # expected to match primarily through FREE-TEXT diagnosis/
        # treatment_plan/current_medications content reaching
        # medical_record_summary (chronic_diseases.* is validated as plain
        # 'string|max:255' on the Laravel side, not restricted to the
        # picker — see UpdateMedicalRecordRequest — so free text is
        # possible, just not the primary structured path the way
        # diabetes's roots are). See CLAUDE.md > Known limitations for the
        # full investigation.
        #
        # Per-root sourcing, each checked independently, not translated:
        #   "cancer" / "neoplas"    — DrugCentral has no single generic
        #     "Cancer" indication (it splits into specific types: "Non-
        #     small cell lung cancer", "Chronic myeloid leukemia", etc.),
        #     but does use a broader "Neoplasms"/"neoplastic" grouping
        #     (verified directly against drugcentral.org, ~100 drugs
        #     indication-tagged under it) — "neoplas" catches that;
        #     "cancer" catches the lay term more likely in free text,
        #     since the picker offers neither.
        #   "transplant" / "graft" — DrugCentral's own real indication
        #     phrasing for transplant immunosuppressants (belatacept,
        #     everolimus, basiliximab) is "GRAFT REJECTION"/"allograft
        #     rejection", not "organ transplant" — verified directly, not
        #     assumed. "graft" catches that; "transplant" catches the more
        #     colloquial phrasing ("kidney transplant") a clinician would
        #     more plausibly type.
        #   "chemotherap"           — standard, unambiguous term.
        #     DrugCentral does not treat it as a condition (it is a
        #     treatment, not an indication), so it is more likely to
        #     appear in free-text notes than a structured field.
        #   "immunosuppress" / "immunocompromis" — checked and confirmed
        #     "immunosuppression" is NOT itself a DrugCentral condition/
        #     indication entry anywhere — nothing is indicated FOR
        #     immunosuppression, drugs CAUSE it. مثبطات المناعة (this
        #     rule's Arabic keyword) more precisely describes a
        #     MEDICATION CLASS than a "condition" — realistically this
        #     will fire from a specific drug name (tacrolimus, cyclosporine,
        #     prednisone, mycophenolate) in current_medications reaching
        #     the summary, not a chronic_diseases entry. Two separate
        #     roots since "immunosuppressed"/"immunosuppressive" and
        #     "immunocompromised" do not share a common stem.
        chronic_condition_keywords_en=frozenset({
            "cancer", "neoplas",
            "transplant", "graft",
            "chemotherap",
            "immunosuppress", "immunocompromis",
        }),
        chronic_requirement=SymptomRequirement(all_of=frozenset({"حمى"})),
    ),
    # Source: ACG Clinical Guideline — Management of Patients With Acute
    # Lower Gastrointestinal Bleeding (American Journal of Gastroenterology);
    # Glasgow-Blatchford score for upper GI bleeding — verify current
    # edition. Any one sign alone is independently sufficient.
    RedFlagRule(
        id="gi_bleed",
        category="gi_bleed",
        reason_ar=(
            "وجود دم في القيء أو البراز، أو براز أسود قطراني، يشير إلى "
            "نزيف هضمي، وهي حالة إسعافية تستوجب تقييمًا فوريًا."
        ),
        source="ACG Clinical Guideline on acute GI bleeding; Glasgow-Blatchford score — verify current edition.",
        requirement=SymptomRequirement(
            any_of=frozenset({"تقيؤ دم", "براز أسود", "دم في البراز"}),
        ),
    ),
    # Source: Wells Criteria for Pulmonary Embolism; ESC Guidelines for the
    # diagnosis and management of acute pulmonary embolism (European
    # Society of Cardiology) — verify current edition.
    RedFlagRule(
        id="pulmonary_embolism",
        category="thromboembolic",
        reason_ar=(
            "ضيق التنفس المترافق مع ألم صدر أو تورم في ساق واحدة أو خفقان "
            "القلب قد يشير إلى انصمام رئوي، وهي حالة إسعافية تستوجب "
            "تقييمًا فوريًا."
        ),
        source="Wells Criteria for Pulmonary Embolism; ESC Guidelines on acute pulmonary embolism — verify current edition.",
        requirement=SymptomRequirement(
            all_of=frozenset({"ضيق تنفس"}),
            any_of=frozenset({"ألم في الصدر", "تورم في ساق واحدة", "خفقان القلب"}),
        ),
    ),
    # Source: American Diabetes Association — Hyperglycemic Crises in
    # Adult Patients With Diabetes; Joint British Diabetes Societies (JBDS)
    # guideline for the management of DKA in adults — verify current
    # edition.
    RedFlagRule(
        id="dka",
        category="endocrine_metabolic",
        reason_ar=(
            "العطش الشديد المترافق مع تنفس سريع وعميق أو تشوش ذهني أو "
            "تقيؤ متكرر أو تبول متكرر قد يشير إلى حماض كيتوني سكري، وهي "
            "حالة إسعافية تستوجب تقييمًا وعلاجًا فوريين."
        ),
        source="ADA Hyperglycemic Crises guideline; JBDS DKA management guideline — verify current edition.",
        requirement=SymptomRequirement(
            all_of=frozenset({"عطش شديد"}),
            any_of=frozenset({"تقيؤ", "نعاس أو تشوش ذهني", "تنفس سريع وعميق", "تبول متكرر"}),
        ),
    ),
    # Source: ACOG Practice Bulletin — Ectopic Pregnancy — verify current
    # edition.
    #
    # OVER-INCLUSION, ACCEPTED DELIBERATELY: this rule has no way to gate
    # on reproductive-age/female patient status. state has no structured
    # demographic fields at all — only medical_record_summary, a free-text
    # filtered summary Laravel sends (CLAUDE.md > State,
    # ChatRequest.medical_record_summary) — and inventing a gating
    # mechanism on top of free text (age/sex inference from a summary
    # string) is not something to improvise into a safety-critical rule.
    # So today, this rule fires for ANY patient matching the symptom
    # combination below, regardless of age or sex. That is the same
    # "bias toward false positives" this project already commits to for
    # crisis detection (rules/crisis.py: "a missed crisis is unacceptable,
    # a false positive costs one gentle redirect message. When in doubt,
    # match.") — applied here to a different cost: a false positive here
    # costs one unnecessary ER referral, a missed true case is a
    # life-threatening emergency going unrecognized. Accepted on that
    # basis, not an oversight. Revisit if/when state ever gains structured
    # demographic fields (CLAUDE.md > Working style: flag explicitly
    # rather than proceed, if a future change touches this).
    RedFlagRule(
        id="ectopic_pregnancy",
        category="obstetric_gynecological",
        reason_ar=(
            "ألم البطن المترافق مع نزيف مهبلي أو تأخر في الدورة الشهرية عند "
            "امرأة في سن الإنجاب قد يشير إلى حمل خارج الرحم، وهي حالة "
            "إسعافية قد تهدد الحياة وتستوجب تقييمًا فوريًا."
        ),
        source="ACOG Practice Bulletin — Ectopic Pregnancy — verify current edition.",
        requirement=SymptomRequirement(
            all_of=frozenset({"ألم بطن"}),
            any_of=frozenset({"نزيف مهبلي", "تأخر الدورة الشهرية"}),
        ),
    ),
)


def _validate_rule_symptoms_are_canonical(rules: tuple[RedFlagRule, ...]) -> None:
    """Fail fast if a rule references a symptom not in the canonical vocabulary.

    Symptom matching is set membership against names produced by
    extract_symptoms. A rule naming a symptom that no longer exists (or
    never did — a typo, a renamed entry) can therefore never fire, and
    nothing about the rule's own text says so: the tests for every *other*
    rule still pass, and the broken one just quietly stops protecting
    anyone. In the emergency path that is the worst possible failure mode,
    so it is caught at import time instead.
    """
    problems = []
    for rule in rules:
        names = rule.requirement.all_names()
        if rule.chronic_requirement is not None:
            names |= rule.chronic_requirement.all_names()

        for name in sorted(names):
            if normalize(name) not in CANONICAL_SYMPTOMS:
                problems.append(f"  [{rule.id}] {name!r}")

    if problems:
        raise ValueError(
            "RED_FLAG_RULES reference symptom name(s) missing from the canonical "
            "vocabulary (vocabulary/symptoms.py). Add them there, or correct the "
            "spelling here:\n" + "\n".join(problems)
        )


_validate_rule_symptoms_are_canonical(RED_FLAG_RULES)


def check_red_flags(
    symptoms: list[Symptom],
    medical_record_summary: str = "",
) -> list[RedFlagMatch]:
    """Evaluate RED_FLAG_RULES against a confirmed-symptom list.

    Pure function: no I/O, no LLM calls, no logging — the caller decides
    what to do with the result (route to emergency_node, log to audit,
    include in the doctor report). medical_record_summary is optional and
    only consulted for rules that define a chronic_requirement.

    Incoming names are normalized before comparison, so a spelling
    variation out of extract_symptoms (أ vs ا, ة vs ه, stray diacritics)
    does not cause a red flag to be missed. RedFlagMatch.matched_symptoms
    therefore reports normalized names. medical_record_summary is checked
    against TWO independent keyword sets per rule, in two different
    languages and by two different mechanisms — Arabic via
    rules.crisis.normalize() + word-boundary regex (_compile_keyword_matcher,
    for chat-typed Syrian Arabic), English via lowercasing + plain
    substring match (_compile_english_keyword_matcher, for Laravel-sourced
    DrugCentral-standard condition/drug names) — see
    RedFlagRule.mentions_chronic_condition and CLAUDE.md > Known
    limitations for why one mechanism does not serve both languages.
    """
    present = frozenset(
        normalize(name) for symptom in symptoms if (name := symptom.get("name"))
    )

    matches: list[RedFlagMatch] = []
    for rule in RED_FLAG_RULES:
        hit = rule.requirement.satisfied_by(present)
        lowered = False

        if hit is None and rule.chronic_requirement is not None:
            if rule.mentions_chronic_condition(medical_record_summary):
                hit = rule.chronic_requirement.satisfied_by(present)
                lowered = hit is not None

        if hit is not None:
            matches.append(
                RedFlagMatch(
                    rule_id=rule.id,
                    category=rule.category,
                    reason_ar=rule.reason_ar,
                    source=rule.source,
                    matched_symptoms=hit,
                    lowered_by_chronic_condition=lowered,
                )
            )

    return matches
