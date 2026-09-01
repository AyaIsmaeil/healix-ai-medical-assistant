"""Fetches real Tier-1 clinical-guideline pages (WHO / NICE / CDC) for a
disease, drafts a knowledge-base entry from the real page text via LLM, and
writes it DIRECTLY into rag/knowledge_base/{disease}.json in the exact live
schema (rag/schema.py's KnowledgeBaseEntry) — ready to use as-is, not a
separate draft shape that still needs manual reshaping.

`source` is the fetched page's own <title> (falling back to the org label
given in TARGET_SOURCES) plus the real URL, e.g.:
    "NICE CKS: Chest infections - adult — https://cks.nice.org.uk/topics/..."
so the citation is both readable and — wherever this text is rendered
somewhere that linkifies URLs — a real, clickable link back to the source.

translation_reviewed is always written as false: an LLM drafted this
entry's Arabic, a human has not checked it against the source yet. Do not
flip it to true by hand until someone fluent in the target dialect has
actually compared it against the raw excerpt saved alongside it (see
below) — same standard every existing translation_reviewed=true entry in
rag/knowledge_base/ was already held to.

A raw evidence record (the fetched page text + the exact draft the LLM
produced) is also kept at scripts/kb_evidence_output/{disease}.json —
independent of the live KB file, so what was fetched and when stays
auditable even after a human edits or approves the KB entry itself.

Skips (never overwrites) any disease that already has a
rag/knowledge_base/{disease}.json file, unless --force is passed — this
script must never silently clobber an existing, possibly human-reviewed
entry.

Same lightweight, offline-only dependency footprint as
scripts/scrape_webteb.py (requests + bs4, deliberately not in
requirements.txt — this is an authoring tool, not a runtime dependency of
api/main.py).

Run it:

    python scripts/collect_kb_evidence.py
    python scripts/collect_kb_evidence.py --force           # overwrite existing KB files too
    python scripts/collect_kb_evidence.py --evidence-only   # (re)fetch+draft every disease into
                                                             # scripts/kb_evidence_output/ only —
                                                             # never touches rag/knowledge_base/,
                                                             # even for a disease that already has
                                                             # a KB file. This is what produces,
                                                             # for every disease currently in the
                                                             # live KB, a real evidence record
                                                             # proving it traces back to an actual
                                                             # fetched source page — without
                                                             # discarding the hand-curated/
                                                             # reviewed KB content itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from llm_client import LLMError, call_llm  # noqa: E402
from prompts.base import build_prompt  # noqa: E402
from rag.schema import KNOWLEDGE_BASE_DIR, KnowledgeBaseEntry  # noqa: E402

load_dotenv()

_EVIDENCE_DIR = Path(__file__).resolve().parent / "kb_evidence_output"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en;q=0.9",
}

# Real, directly-verified URLs — each one fetched and confirmed to match
# its disease during this project's own source-verification pass (not
# guessed; see git history around the rag/knowledge_base/*.json "source"
# field for the same URLs, added there by hand). One disease can list more
# than one source; the first one that fetches AND drafts successfully wins
# (see collect_one / main).
#
# Every disease below already has a rag/knowledge_base/*.json file today —
# a plain run is a no-op for all of them (each is SKIPPED) until that file
# is removed or --force is passed; use --evidence-only to fetch+draft
# every one of them into scripts/kb_evidence_output/ regardless, without
# ever touching rag/knowledge_base/ (see main()) — this is what to use to
# (re)produce proof that every current KB entry traces back to a real,
# fetched source, without discarding the hand-curated/reviewed KB content
# itself.
TARGET_SOURCES: dict[str, list[tuple[str, str]]] = {
    "acute_bronchitis": [("NICE CKS", "https://cks.nice.org.uk/topics/chest-infections-adult/")],
    "acute_gastroenteritis": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease")],
    "acute_musculoskeletal_strain": [("NICE CKS", "https://cks.nice.org.uk/topics/neck-pain-non-specific/")],
    "acute_otitis_media": [("AAP/AAFP", "https://publications.aap.org/pediatrics/article/131/3/e964/30912/The-Diagnosis-and-Management-of-Acute-Otitis-Media")],
    "acute_sinusitis": [("AAO-HNS", "https://www.entnet.org/quality-practice/quality-products/clinical-practice-guidelines/cpg-adult-sinusitis/")],
    "allergic_rhinitis": [("ARIA/EUFOREA", "https://www.euforea.org/aria/")],
    "asthma": [("GINA", "https://ginasthma.org/2026-gina-strategy-report/")],
    "atopic_dermatitis": [("AAD", "https://www.aad.org/member/clinical-quality/guidelines/atopic-dermatitis")],
    "bacterial_vaginosis": [("CDC", "https://www.cdc.gov/std/treatment-guidelines/bv.htm")],
    "benign_paroxysmal_positional_vertigo": [("AAO-HNS", "https://www.entnet.org/quality-practice/quality-products/clinical-practice-guidelines/bppv/")],
    "chickenpox": [("CDC", "https://www.cdc.gov/chickenpox/hcp/clinical-overview/index.html")],
    "community_acquired_pneumonia": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/pneumonia")],
    "conjunctivitis": [("AAO", "https://www.aao.org/education/preferred-practice-pattern/conjunctivitis-ppp-2023")],
    "cutaneous_leishmaniasis": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/leishmaniasis")],
    "dysmenorrhea": [("ACOG", "https://www.acog.org/womens-health/faqs/dysmenorrhea-painful-periods")],
    "gastroesophageal_reflux_disease": [("ACG", "https://doi.org/10.14309/ajg.0000000000001538")],
    "gout": [("ACR", "https://doi.org/10.1002/art.41247")],
    "hand_foot_and_mouth_disease": [("CDC", "https://www.cdc.gov/hand-foot-mouth/about/index.html")],
    "hepatitis_a": [("CDC", "https://www.cdc.gov/hepatitis-a/hcp/clinical-overview/index.html")],
    "herpes_zoster": [("CDC", "https://www.cdc.gov/shingles/about/index.html")],
    "hypertension": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/hypertension")],
    "impetigo": [("CDC", "https://www.cdc.gov/group-a-strep/about/impetigo.html")],
    "infectious_mononucleosis": [("CDC", "https://www.cdc.gov/epstein-barr/about/mononucleosis.html")],
    "influenza": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)")],
    "iron_deficiency_anaemia": [("WHO/UNICEF/UNU", "https://cdn.who.int/media/docs/default-source/2021-dha-docs/ida_assessment_prevention_control.pdf")],
    "irritable_bowel_syndrome": [("NICE", "https://www.nice.org.uk/guidance/cg61")],
    "kidney_stones": [("EAU", "https://uroweb.org/guidelines/urolithiasis")],
    "measles": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/measles")],
    "migraine": [("IHS ICHD-3", "https://ihs-headache.org/en/resources/ichd/")],
    "mumps": [("CDC", "https://www.cdc.gov/mumps/hcp/clinical-overview/")],
    "otitis_externa": [("AAO-HNS", "https://www.entnet.org/quality-practice/quality-products/clinical-practice-guidelines/aoe/")],
    "pediculosis_capitis": [("CDC", "https://www.cdc.gov/lice/about/index.html")],
    "peptic_ulcer_disease": [("NICE CKS", "https://cks.nice.org.uk/topics/dyspepsia-proven-peptic-ulcer/")],
    "pinworm_infection": [("CDC", "https://www.cdc.gov/pinworm/about/index.html")],
    "polycystic_ovary_syndrome": [("ACOG", "https://www.acog.org/clinical/clinical-guidance/practice-bulletin/articles/2018/06/polycystic-ovary-syndrome")],
    "rheumatoid_arthritis": [("NICE", "https://www.nice.org.uk/guidance/ng100")],
    "roseola": [("AAP", "https://www.healthychildren.org/English/health-issues/conditions/skin/Pages/Roseola-Infantum.aspx")],
    "rubella": [("CDC", "https://www.cdc.gov/rubella/about/index.html")],
    "scabies": [("CDC", "https://www.cdc.gov/scabies/hcp/clinical-care/index.html")],
    "streptococcal_pharyngitis": [("CDC", "https://www.cdc.gov/group-a-strep/hcp/clinical-guidance/strep-throat.html")],
    "tendinitis": [("Cleveland Clinic", "https://my.clevelandclinic.org/health/diseases/10919-tendonitis")],
    "tension_type_headache": [("IHS ICHD-3", "https://ihs-headache.org/en/resources/ichd/")],
    "tonsillitis": [("Windfuhr et al. 2016", "https://pmc.ncbi.nlm.nih.gov/articles/PMC7087627")],
    "type_2_diabetes": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/diabetes")],
    "typhoid_fever": [("WHO", "https://www.who.int/news-room/fact-sheets/detail/typhoid")],
    "urinary_tract_infection": [("NICE", "https://www.nice.org.uk/guidance/ng109")],
    "urticaria": [("EAACI/GA2LEN/EDF/WAO", "https://onlinelibrary.wiley.com/doi/10.1111/all.13397")],
    "vaginal_candidiasis": [("ACOG", "https://www.acog.org/clinical/clinical-guidance/practice-bulletin/articles/2020/01/vaginitis-in-nonpregnant-patients")],
    "viral_pharyngitis": [("NICE NG84", "https://www.nice.org.uk/guidance/ng84")],
}

_MAX_EXCERPT_CHARS = 6000

# Exactly the specialty strings nodes/route_specialty.py's SPECIALTY_MAP
# and LARAVEL_SPECIALTIES already know how to route to a real Laravel
# specialization row (confirmed against every rag/knowledge_base/*.json
# entry's own "specialties" values). Never widen this set here without
# adding the matching Laravel-side mapping first — an unmapped specialty
# string would silently never match a doctor-lookup query (see that
# module's own docstring).
_KB_SPECIALTIES: tuple[str, ...] = (
    "أطفال",
    "أنف وأذن وحنجرة",
    "باطنية",
    "جلدية",
    "حساسية",
    "دموية",
    "روماتيزم",
    "صدرية",
    "طب عام",
    "عصبية",
    "عظمية",
    "علاج طبيعي",
    "عينية",
    "غدد صماء",
    "قلبية",
    "مسالك بولية",
    "معدية",
    "نسائية",
    "هضمية",
)
_Specialty = Literal[_KB_SPECIALTIES]  # type: ignore[valid-type]


class KbEntryDraft(BaseModel):
    """LLM output for one disease's draft entry — every field grounded
    ONLY in the fetched excerpt, never inferred or added from general
    medical knowledge (enforced by prompts/templates/kb_evidence_draft.txt,
    not just this schema)."""

    name: str = Field(description="Disease name in English, as given by the source.")
    name_ar: str = Field(
        min_length=1,
        description="Plain, patient-facing Arabic name for this disease — the "
        "way a Syrian patient would actually say it, not a clinical/diagnostic "
        "label.",
    )
    symptoms_ar: list[str] = Field(
        min_length=1,
        description="Symptoms/signs explicitly stated in the excerpt, in "
        "natural patient-facing Syrian-colloquial Arabic — never inferred or "
        "added, never a literal clinical/Latin term.",
    )
    specialties: list[_Specialty] = Field(
        min_length=1,
        description="Which of the given specialty values a matching diagnosis "
        "for this disease would route to. Every entry must be exactly one of "
        "the allowed values, spelled exactly as given.",
    )
    applicable_sex: Literal["male", "female"] | None = Field(
        default=None,
        description="Set ONLY when this condition is anatomically restricted "
        "to one sex (e.g. dysmenorrhea, vaginal candidiasis) — null for "
        "almost every disease. Never set this just because a condition is "
        "more common in one sex (e.g. urinary tract infection) — that is a "
        "prevalence fact, not an anatomical exclusion.",
    )
    note: str | None = Field(
        default=None,
        description="One short clinical nuance from the excerpt worth "
        "flagging for review, or null if none.",
    )


def fetch_page(url: str) -> str:
    response = requests.get(url, headers=_HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_title(html: str) -> str | None:
    """The page's own <title> text, collapsed to one line — this is what
    ends up as the human-readable half of `source` (see _citation), since
    it is usually the exact citation label a reader would expect (e.g.
    "NICE CKS: Chest infections - adult")."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    return None


def _citation(source_org: str, title: str | None, url: str) -> str:
    label = title or source_org
    return f"{label} — {url}"


def draft_entry(raw_excerpt: str) -> KbEntryDraft | None:
    """Best-effort LLM draft — returns None (not raised) on any LLM
    failure, since the raw evidence (the real source text) is still worth
    saving even when drafting fails."""
    try:
        prompt = build_prompt("kb_evidence_draft", excerpt=raw_excerpt[:_MAX_EXCERPT_CHARS])
    except FileNotFoundError:
        return None
    try:
        result = call_llm(
            prompt, schema=KbEntryDraft, tier="quality", prompt_name="kb_evidence_draft"
        )
    except LLMError as exc:
        print(f"  (LLM draft step skipped: {exc})")
        return None
    assert isinstance(result, KbEntryDraft)
    return result


@dataclass
class EvidenceRecord:
    disease: str
    source_org: str
    source_url: str
    page_title: str | None
    fetched_at: str
    raw_excerpt: str
    status: str
    draft: dict | None = None


def collect_one(
    disease: str, source_org: str, url: str
) -> tuple[EvidenceRecord, KnowledgeBaseEntry | None]:
    print(f"Fetching {disease} <- {source_org} ({url})")
    html = fetch_page(url)
    text = extract_text(html)
    title = extract_title(html)
    excerpt = text[:_MAX_EXCERPT_CHARS]

    draft = draft_entry(excerpt)

    record = EvidenceRecord(
        disease=disease,
        source_org=source_org,
        source_url=url,
        page_title=title,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        raw_excerpt=excerpt,
        status="drafted" if draft else "fetch_only_no_draft",
        draft=draft.model_dump() if draft else None,
    )

    if draft is None:
        return record, None

    entry = KnowledgeBaseEntry(
        name=draft.name,
        name_ar=draft.name_ar,
        symptoms=draft.symptoms_ar,
        specialties=list(draft.specialties),
        source=_citation(source_org, title, url),
        note=draft.note,
        translation_reviewed=False,
        applicable_sex=draft.applicable_sex,
    )
    return record, entry


def _entry_to_json_dict(entry: KnowledgeBaseEntry) -> dict:
    """Same key set and order as every existing rag/knowledge_base/*.json
    file — applicable_sex is OMITTED entirely when null (never written as
    "applicable_sex": null), matching how every sex-unrestricted entry in
    the live KB already looks; `note` stays present even when null, same
    as those files too."""
    data: dict = {
        "name": entry.name,
        "name_ar": entry.name_ar,
        "symptoms": entry.symptoms,
        "specialties": entry.specialties,
    }
    if entry.applicable_sex is not None:
        data["applicable_sex"] = entry.applicable_sex
    data["source"] = entry.source
    data["note"] = entry.note
    data["translation_reviewed"] = entry.translation_reviewed
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing rag/knowledge_base/{disease}.json instead of skipping it.",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Fetch + draft every disease in TARGET_SOURCES into "
        "scripts/kb_evidence_output/ regardless of whether it already has a "
        "rag/knowledge_base/ file — but NEVER write to rag/knowledge_base/ "
        "itself. Use this to (re)produce proof that every current KB entry "
        "traces back to a real, fetched source, without touching the "
        "hand-curated/reviewed KB content. Overrides --force.",
    )
    args = parser.parse_args()

    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    for disease, sources in TARGET_SOURCES.items():
        kb_path = KNOWLEDGE_BASE_DIR / f"{disease}.json"
        if kb_path.exists() and not args.force and not args.evidence_only:
            print(f"{disease}: SKIPPED — {kb_path} already exists (pass --force to overwrite)")
            continue

        record: EvidenceRecord | None = None
        entry: KnowledgeBaseEntry | None = None
        for source_org, url in sources:
            try:
                record, entry = collect_one(disease, source_org, url)
            except requests.RequestException as exc:
                print(f"  FAILED: {exc}")
                continue
            if entry is not None:
                break  # first successfully-fetched-and-drafted source wins

        if record is not None:
            evidence_path = _EVIDENCE_DIR / f"{disease}.json"
            evidence_path.write_text(
                json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  evidence -> {evidence_path}")

        if entry is None:
            print(f"  -> no draft entry for {disease}; nothing written to the knowledge base")
            continue

        if args.evidence_only:
            print(f"  -> (--evidence-only: rag/knowledge_base/{disease}.json left untouched)")
            continue

        try:
            KnowledgeBaseEntry.model_validate(entry.model_dump())
        except ValidationError as exc:
            print(f"  -> REFUSED to write {disease}: entry failed schema validation:\n{exc}")
            continue

        kb_path.write_text(
            json.dumps(_entry_to_json_dict(entry), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  -> {kb_path}  (translation_reviewed=false — needs human review before trusting)")


if __name__ == "__main__":
    main()
