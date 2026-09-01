
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ar,en;q=0.8",
}

_BASE = "https://www.webteb.com"
_ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "webteb_output"
_DIRECTORY_CSV = _OUTPUT_DIR / "webteb_directory.csv"
_CANDIDATES_JSON = _OUTPUT_DIR / "webteb_candidates.json"

TARGET_DISEASES: dict[str, list[str]] = {
    "Acute Bronchial Infection": ["bronchitis"],
    "Acute Bronchitis": ["bronchitis"],
    "Acute Otitis Media": ["otitis media", "ear infection"],
    "Acute Sinusitis": ["sinusitis"],
    "Asthma": ["asthma"],
    "Chickenpox": ["chickenpox", "chicken pox", "varicella"],
    "Conjunctivitis": ["conjunctivitis", "pink eye"],
    "Gout": ["gout"],
    "Impetigo": ["impetigo"],
    "Irritable Bowel Syndrome": ["irritable bowel", "ibs"],
    "Migraine": ["migraine"],
    "Mumps": ["mumps"],
    "Pinworm Infection": ["pinworm"],
    "Rheumatoid Arthritis": ["rheumatoid arthritis"],
    "Scabies": ["scabies"],
    "Tendinitis": ["tendinitis", "tendonitis"],
    "Tonsillitis": ["tonsillitis"],
    "Typhoid Fever": ["typhoid"],
    "Urinary Tract Infection": ["urinary tract infection", " uti"],
    "Community-Acquired Pneumonia": ["pneumonia"],
    "Gastroesophageal Reflux Disease": ["gastroesophageal reflux", "gerd", "acid reflux"],
    "Herpes Zoster": ["herpes zoster", "shingles"],
    "Infectious Mononucleosis": ["mononucleosis", "glandular fever"],
    "Iron Deficiency Anaemia": ["iron deficiency anemia", "iron deficiency anaemia"],
    "Otitis Externa": ["otitis externa", "swimmer's ear", "swimmers ear"],
    "Polycystic Ovary Syndrome": ["polycystic ovary", "pcos"],
    "Tension-Type Headache": ["tension headache", "tension-type headache"],
    "Vaginal Candidiasis": ["vaginal candidiasis", "yeast infection", "vaginal thrush"],
    "Benign Paroxysmal Positional Vertigo": ["bppv", "positional vertigo"],
    "Acute Gastroenteritis": ["gastroenteritis"],
    "Peptic Ulcer Disease": ["peptic ulcer"],
    "Acute Musculoskeletal Strain": ["muscle strain", "sprain"],
    "Allergic Rhinitis": ["allergic rhinitis", "hay fever"],
    "Atopic Dermatitis": ["atopic dermatitis", "eczema"],
    "Bacterial Vaginosis": ["bacterial vaginosis"],
    "Cutaneous Leishmaniasis": ["leishmaniasis"],
    "Dysmenorrhea": ["dysmenorrhea", "menstrual pain", "period pain"],
    "Hand, Foot, and Mouth Disease": ["hand foot and mouth", "hand foot mouth"],
    "Hepatitis A": ["hepatitis a"],
    "Hypertension": ["blood pressure", "hypertension"],
    "Influenza": ["influenza", "flu"],
    "Kidney Stones": ["kidney stone", "renal stone"],
    "Measles": ["measles"],
    "Pediculosis Capitis": ["lice", "pediculosis"],
    "Roseola": ["roseola"],
    "Rubella": ["rubella", "german measles"],
    "Streptococcal Pharyngitis": ["strep throat", "streptococcal pharyngitis"],
    "Type 2 Diabetes": ["type 2 diabetes", "type ii diabetes"],
    "Urticaria": ["urticaria", "hives"],
    "Viral Pharyngitis": ["viral pharyngitis", "pharyngitis"],
}


def _get(url: str, retries: int = 3, delay: float = 2.0) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            print(f"  [محاولة {attempt + 1}/{retries}] فشل {url}: {exc}")
            time.sleep(delay)
    return None


def build_directory() -> None:
    """يستعرض ويب طب لكل حرف عربي، يجمع كل أمراضه، ويخزّنها في CSV.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    rows: list[dict[str, str]] = []

    for letter in _ARABIC_LETTERS:
        url = f"{_BASE}/diseases/list/{quote(letter)}"
        print(f"فهرسة الحرف '{letter}' -> {url}")
        html = _get(url)
        if html is None:
            print(f"  تخطّي '{letter}' (فشل التحميل)")
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", class_="bold", href=True):
            href = a["href"]
            if "/diseases/" not in href or href.endswith("/diseases"):
                continue
            full_url = href if href.startswith("http") else f"{_BASE}{href}"
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            text = a.get_text(strip=True)
            # الصيغة المتوقعة: "الاسم العربي-English Name"
            if "-" in text:
                ar_name, en_name = text.rsplit("-", 1)
            else:
                ar_name, en_name = text, ""
            rows.append(
                {
                    "arabic_name": ar_name.strip(),
                    "english_name": en_name.strip(),
                    "url": full_url,
                }
            )

        time.sleep(1.5)  

    with open(_DIRECTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arabic_name", "english_name", "url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nتمّ حفظ {len(rows)} مرضًا فريدًا بـ {_DIRECTORY_CSV}")


def _find_matches(directory: list[dict[str, str]]) -> dict[str, dict[str, str]]:

    matches: dict[str, dict[str, str]] = {}
    for row in directory:
        en_lower = row["english_name"].lower()
        if not en_lower:
            continue
        for healix_name, keywords in TARGET_DISEASES.items():
            if healix_name in matches:
                continue
            if any(kw in en_lower for kw in keywords):
                matches[healix_name] = row
    return matches


def _extract_symptoms(html: str) -> tuple[str, str, list[str]] | None:
    """Return (arabic_name, english_name, symptoms) or None if no symptoms section found.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    h1 = soup.find("h1")
    ar_name = h1.get_text(strip=True) if h1 else ""

    en_h2 = soup.find("h2", class_="en-name")
    en_name = en_h2.get_text(strip=True) if en_h2 else ""

    symptoms_h2 = None
    for h2 in soup.find_all("h2"):
        if h2.get_text(strip=True).startswith("أعراض"):
            symptoms_h2 = h2
            break
    if symptoms_h2 is None:
        return None

    content_div = symptoms_h2.find_next_sibling("div")
    if content_div is None:
        return None

    symptoms = []
    for li in content_div.find_all("li"):
        text = li.get_text(strip=True).rstrip(".،")
        if text:
            symptoms.append(text)

    return ar_name, en_name, symptoms


def match_and_scrape() -> None:
    
    if not _DIRECTORY_CSV.exists():
        raise SystemExit(
            f"لا يوجد {_DIRECTORY_CSV} — شغّلي build_directory() أولًا "
            "(--build-directory أو --all)."
        )

    with open(_DIRECTORY_CSV, encoding="utf-8") as f:
        directory = list(csv.DictReader(f))

    matches = _find_matches(directory)
    print(f"تطابقات لقيتها: {len(matches)} من أصل {len(TARGET_DISEASES)} مرض مستهدف")

    missing = set(TARGET_DISEASES) - set(matches)
    if missing:
        print(" ما لقيت تطابق لهاي الأمراض على ويب طب:")
        for name in sorted(missing):
            print(f"  - {name}")

    candidates = []
    for healix_name, row in matches.items():
        print(f"سحب: {healix_name} <- {row['url']}")
        html = _get(row["url"])
        if html is None:
            print("  فشل التحميل، تخطّي")
            continue

        result = _extract_symptoms(html)
        if result is None:
            print("  ما لقيت قسم أعراض بالصفحة، تخطّي")
            continue

        ar_name, en_name, symptoms = result
        candidates.append(
            {
                "healix_target_name": healix_name,
                "name": en_name or healix_name,
                "name_ar": ar_name,
                "symptoms": symptoms,
                "specialties": [],  # فاضي عمدًا — يحتاج تحديد يدوي
                "source": row["url"],
                "note": "مسحوب آليًا من ويب طب — يحتاج مراجعة طبية ولغوية كاملة",
                "translation_reviewed": False,
            }
        )
        time.sleep(1.5)

    with open(_CANDIDATES_JSON, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    print(f"\nتمّ حفظ {len(candidates)} مرشّح بـ {_CANDIDATES_JSON}")
    print(" راجعي كل حقل يدويًا (خصوصًا specialties الفاضية، وصحة الأعراض")
    print("طبيًا) قبل أي نقل لـ rag/knowledge_base/ — نفس معيار translation_reviewed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-directory", action="store_true", help="بناء فهرس كل أمراض الموقع (مرة أولى)"
    )
    parser.add_argument(
        "--scrape", action="store_true", help="مطابقة وسحب الأمراض المستهدفة من الفهرس"
    )
    parser.add_argument("--all", action="store_true", help="الاثنين بالتتابع")
    args = parser.parse_args()

    if args.all or args.build_directory:
        build_directory()
    if args.all or args.scrape:
        match_and_scrape()
    if not (args.all or args.build_directory or args.scrape):
        parser.print_help()
