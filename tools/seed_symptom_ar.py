"""
Healix - Arabic synonym seed for the symptom-coding lexicon (offline tool).

Fills the ``ar`` synonym lists in ``app/dictionaries/symptom_ontology.json`` for
a first, high-value subset of symptoms (MSA + common Levantine/Gulf dialect).
Reproducible and auditable: run it, review the diff, extend ``SEED`` under
clinical review. It PRESERVES every field except ``ar`` for the ids it touches,
and never removes ids or Arabic terms it does not mention.

Run:
    PYTHONPATH=. venv/Scripts/python.exe tools/seed_symptom_ar.py
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PATH = _ROOT / "app" / "dictionaries" / "symptom_ontology.json"

# HEALIX id (last 4 digits) -> Arabic synonyms. Distinct terms per id on purpose:
# a generic word ("تعب") is anchored to ONE id so one phrase never inflates
# several evidences. Ambiguous near-duplicates get their most specific wording.
SEED: dict[str, list[str]] = {
    "0001": ["حمى", "حرارة", "سخونة", "سخونية", "ارتفاع حرارة", "حراره", "سخونه"],
    "0003": ["فقدان وعي", "غيبوبة", "إغماء", "فقدت الوعي", "أغمي علي", "غاب عن الوعي"],
    "0004": ["طفح جلدي", "طفح", "بقع جلدية", "احمرار الجلد", "حبوب بالجلد", "التهاب جلد"],
    "0005": ["شحوب", "شحوب الجلد", "بشرة شاحبة", "لون شاحب"],
    "0006": ["خفقان", "خفقان القلب", "تسارع القلب", "دقات قلب سريعة", "قلبي بيدق بسرعة", "نبض سريع"],
    "0009": ["إسهال", "اسهال", "إسهال متكرر", "بطني ماشي"],
    "0011": ["تعب", "إرهاق", "إعياء", "تعبان", "مرهق"],
    "0013": ["دوخة شديدة", "إغماء وشيك", "رح يغمى علي", "حاسس رح أوقع"],
    "0014": ["غثيان", "غثيان ورغبة بالتقيؤ", "نفسي تلوّع", "حاسس بدي أستفرغ"],
    "0015": ["قشعريرة", "رعشة", "رجفة", "برد ورعشة", "قشعريره"],
    "0017": ["فقدان الشهية", "ما في شهية", "بشبع بسرعة", "فقدت شهيتي"],
    "0019": ["نقص وزن", "خسارة وزن", "نحول", "نقص وزن غير مقصود"],
    "0020": ["حرقة معدة", "حموضة", "حرقان بالمعدة", "ارتجاع حمضي"],
    "0024": ["ضيق تنفس", "صعوبة تنفس", "ضيق نفس", "ما بقدر أتنفس", "نهجان", "ضيق بالنفس"],
    "0028": ["احتقان أنف", "رشح", "سيلان أنف", "زكام", "أنف مسدود", "رشح بالأنف"],
    "0032": ["ألم عضلي", "أوجاع عضلات", "وجع بالعضلات", "آلام جسم", "وجع بالجسم"],
    "0034": ["سعال", "كحة", "سعله", "كحه", "بسعل"],
    "0040": ["التهاب حلق", "ألم حلق", "وجع بلعوم", "حلقي بيوجعني", "التهاب بالحلق"],
    "0041": ["تورم غدد", "غدد منتفخة", "تضخم العقد اللمفاوية", "غدد متورمة"],
    "0043": ["صعوبة بلع", "ألم عند البلع", "ما بقدر أبلع", "صعوبة بالبلع"],
    "0044": ["احمرار العين", "عيون حمرا", "احمرار بالعين", "عيوني حمرا"],
    "0049": ["تشوش ذهني", "تشوش", "ارتباك", "عدم تركيز", "تشوش بالتفكير"],
    "0051": ["تقرحات فم", "قرحة بالفم", "حبوب بالفم", "تقرحات بالفم"],
    "0053": ["نوبات سعال", "نوبات كحة شديدة", "سعال متواصل"],
    "0058": ["تقيؤ", "استفراغ", "ترجيع", "تقيؤ متكرر", "عم بستفرغ"],
    "0063": ["نفث دم", "سعال مدمى", "دم مع الكحة", "بصق دم"],
    "0071": ["قلة الشهية", "ضعف الشهية", "شهية قليلة"],
    "0075": ["حكة بالعين", "عيون بتحك", "حكة شديدة بالعين"],
    "0076": ["بلغم", "سعال مع بلغم", "كحة مع بلغم", "بلغم ملون"],
    "0078": ["ألم صدر", "وجع بالصدر", "ألم بالصدر", "ألم صدر وقت الراحة"],
    "0079": ["حكة بالأنف", "حكة بالحلق", "أنفي بيحك"],
    "0095": ["فقدان الشم", "ما عم بشم", "فقدان حاسة الشم", "بطّلت أشم"],
}


def main() -> None:
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    symptoms = data["symptoms"]

    filled = 0
    for suffix, terms in SEED.items():
        healix_id = f"HEALIX_SYMPTOM_{suffix}"
        if healix_id not in symptoms:
            raise SystemExit(f"[error] unknown id {healix_id} — regenerate template first")
        # de-dup while preserving order; keep any pre-existing terms too
        existing = symptoms[healix_id].get("ar", [])
        merged = list(dict.fromkeys([*existing, *terms]))
        symptoms[healix_id]["ar"] = merged
        filled += 1

    total_with_ar = sum(1 for s in symptoms.values() if s.get("ar"))
    _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] seeded {filled} ids | {total_with_ar}/{len(symptoms)} symptoms now have Arabic terms")


if __name__ == "__main__":
    main()
