# AHD Dataset — Inspection Report

**Scope:** read-only, structural/statistical inspection of three files in `data/ahd_raw/`. No file in that directory was modified, renamed, moved, or committed. No cleaning, filtering, indexing, or dependency installation was performed. This report is the required stop-point before any of that work begins.

**Files inspected**

| File | Size | SHA-256 |
|---|---|---|
| `AHD.xlsx` | 180,897,828 bytes (~172 MB) | `37098efd6a85c63f26d7778e0b98c5d274ed9ef3702fc6579f81e72fcdec00ed` |
| `AHD_english.xlsx` | 123,764,685 bytes (~118 MB) | `ecb842e81268b70c7455ced45ae8a5ec94f06df4ab0b9afec5a20a148d310ac6` |
| `Distribution of Question and Answer per category.xlsx` | 12,614 bytes | `5d910a009b8657a152706cb35539e436afa6894b4f69d687b67fa81a579d48bb` |

All inspection was done with `openpyxl.load_workbook(path, read_only=True, data_only=True)` streaming iteration — no full in-memory/pandas load of the large files, and no bulk content was printed at any point (samples below are truncated to 40–70 characters each).

---

## 1. Structure

| File | Sheet | Rows (incl. header) | Columns |
|---|---|---|---|
| `AHD.xlsx` | `Sheet1` | 808,473 | `Question`, `Answer`, `Category` |
| `AHD_english.xlsx` | `Sheet1` | 808,473 | `Question`, `Answer`, `Category` |
| `Distribution of Question and Answer per category.xlsx` | `ورقة1` | 91 | `Category`, `English translation`, `Size` |

`AHD.xlsx` and `AHD_english.xlsx` have identical row counts and column layout. A spot check of rows 2–6 across both files confirms they are **row-aligned translations of each other**, not independent samples — e.g. row 2's Arabic question about a breast MRI corresponds exactly to row 2's English "A breast MRI was performed…" (same clinical content, same category, `جراحة عامة` / `General Surgery`).

The distribution file is a **category-size summary**, not raw Q&A data (91 rows = 90 categories + header; columns are `Category`, `English translation`, `Size`, e.g. `['أمراض نسائية', 'Gynaecological ', 162142]`). It is useful for understanding class balance, not a data source itself.

## 2. Role of each file (for a future retrieval component)

- **`AHD.xlsx`** — the actual Arabic Q&A corpus. This is the only file that would be used for Arabic health-education retrieval.
- **`AHD_english.xlsx`** — English translation of the same corpus, row-aligned. Not needed for an Arabic-only retrieval endpoint; potentially useful later only if an English-language mode is ever built.
- **`Distribution…xlsx`** — reference/summary only (category names + sizes). Useful for reporting/QA, not for retrieval.

## 3. Full-scan statistics (`AHD.xlsx`, all 808,472 data rows)

| Check | Count | % of rows |
|---|---|---|
| Empty/null `Question` | 2 | ~0.0002% |
| Empty/null `Answer` | 2 | ~0.0002% |
| Empty/null `Category` | 0 | 0% |
| Non-empty `Answer` under 5 characters | 306 | ~0.04% |
| Unique questions | 807,333 | — |
| Questions appearing more than once | 1,093 distinct texts, 2,232 rows total | ~0.28% of rows |
| Question text containing no Arabic characters at all | 314 | ~0.04% |
| Rows containing a URL | 676 | ~0.08% |
| Answers over 3,000 characters | 819 | ~0.10% |
| Distinct categories | 90 | — |

**Duplicate questions**, on inspection, are short generic phrasings different patients plausibly typed independently (e.g. `"هل هذا الغشاء سليم ؟"` — "is this hymen intact?"), not obvious copy-paste artifacts. Not a data-quality red flag on its own.

**Non-Arabic-question rows (314)** are not garbage — on inspection they are real patient submissions in three recognizable patterns:
- Lab-result value dumps (e.g. `WBC 7.5 / LYM 2.4 / HGB 12…`) — numeric/English lab panels pasted as the "question," with an Arabic answer.
- French/English free text from the patient (e.g. `"J'ai 25 ans, je suis mariée depuis 8 mois…"`).
- At least one clear junk/test row: `"test calll from lina 33"` with the answer `"أسئلة وأجوبة طبية - أمراض باطنية | الطبي"` (a scraped category-page label, not a real answer) — evidence the scrape picked up at least some non-Q&A site content.

No HTML markup was found in any row (0 matches for an HTML-tag pattern).

## 4. Medication / dosage content

**108,228 rows (~13.4%)** contain at least one medication/dosage-related keyword (`جرعة`, `مجم`, `ملغ`, `قرص`/`أقراص`, `حقنة`, `دواء`, `mg`, `ml`, etc.). This is a large fraction of the corpus. Per your own binding constraint, this dataset must never be used to prescribe medication or provide dosage guidance — any retrieval built on this corpus needs an explicit filter or answer-time guard so a matched Q&A pair that contains dosage language is not surfaced as medical advice.

## 5. PII scan

- **Email addresses:** 0 matches.
- **Phone-number-like patterns:** 986 raw regex matches, but on manual review these are essentially all **false positives** — dates (`٢٠-١٢-٢٠٢٢`), lab reference ranges (`700-----1000`), calorie ranges (`1800 - 2000`), HPV genotype lists (`18-51-53-31-16-45`). No genuine phone numbers were found in the sampled matches. Treat the 986 figure as noise, not a real PII count.
- **Self-disclosed first names ("اسمي X" / "my name is X"):** 671 rows. Unlike the phone check, this one is **real** — patients genuinely write their first name when asking a question (`"اسمي محمد عمري ٢١…"`, `"اسمي عبدالله عمري 21 عام…"`, `"السلام عليكم اسمي (م) عمري 23 سنة…"`). This directly contradicts the source paper's own claim (see §7) that "all the data is non-sensitive." It is limited to first names in free text (no phone numbers, emails, or national IDs found), but it is genuine, unredacted PII and should be scrubbed/masked before any of this text is stored, indexed, or shown back to a user.

## 6. Data quality summary

The dataset is essentially clean at the structural level (near-zero nulls, no malformed rows in the schema sense — every row has exactly 3 columns). The issues that exist are all content-level and expected of raw, unprocessed, scraped forum-style data:
- Nulls/empty fields: negligible (4 rows total across Question+Answer).
- Truly malformed/junk rows: rare but present (at least one confirmed test/junk row; likely a handful more given 808k rows).
- Non-Arabic-only questions: 0.04%, real patient content (mixed-language or lab-value questions), not corruption.
- Category imbalance: severe — top category (`أمراض نسائية`, gynecology) has 162,142 rows vs. the smallest categories with a few hundred. Confirmed by both my own count and the paper's own disclosed limitation (§7).
- Real (if limited) PII present in free text: first names.
- Medication/dosage content is common (~13%) and needs filtering given your constraint against dosage/prescription output.

## 7. Independent provenance/license verification

Verified live against the dataset's own Mendeley Data page and its peer-reviewed companion paper (Data in Brief, PMC), not just the version you supplied:

- **Dataset name:** AHD: Arabic Healthcare Dataset. **Author:** Hezam Gawbah (with co-authors Akram Alsubari, Nashwan Ahmed Al-Majmar — Ibb University / Aljazeera University, Yemen).
- **DOI:** `10.17632/mgj29ndgrk.5` — confirmed exactly matches what you stated.
- **License:** CC BY 4.0 (`http://creativecommons.org/licenses/by/4.0/`) — confirmed.
- **Record count:** "over 808,000 question-and-answer pairs across 90 categories" — confirmed consistent with my own count (808,472 data rows, 90 categories).
- **Source / provenance — confirmed and important:** the paper states the data was **scraped from the Altibbi website** using Python Requests + BeautifulSoup. The authors state they "followed the source website Altibbi's ToS, privacy laws, and user consents," claim the data was "anonymized," and claim "all the data is non-sensitive." **My own PII scan (§5) contradicts the "non-sensitive" claim** — 671 rows contain a self-disclosed first name. This is a real discrepancy between the paper's claim and the actual file contents, and is the strongest reason to keep this dataset out of anything that stores or re-displays raw rows verbatim without a scrub step.
- **Ethics approval:** the paper explicitly states "ethical approval has not been sought," reasoning that Altibbi's content is public and user-consented. No IRB or formal ethics review exists for this dataset.
- **Stated limitations (from the paper itself):** collected from a single website only; category-unbalanced; released in "raw format" with "no cleaning, stemming or any type of pre-processing."
- **No clinical/diagnostic-use disclaimer** is stated in the paper — the authors don't warn against clinical use, which makes it your own project's job (already reflected in your binding constraints) to enforce that boundary.

Sources: [Mendeley Data — AHD v5](https://data.mendeley.com/datasets/mgj29ndgrk/5), [PMC11403399 — "AHD: Arabic healthcare dataset" (Data in Brief)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403399/).

## 8. File comparison / recommendation

| | `AHD.xlsx` | `AHD_english.xlsx` | `Distribution…xlsx` |
|---|---|---|---|
| Use for Arabic Q&A retrieval | **Yes — primary source** | No (English, not needed for an Arabic endpoint) | No (metadata only) |
| Row-aligned with AHD.xlsx | — | Yes, confirmed | N/A |
| Needed at all for Phase 1 | Yes | Not unless an English mode is added later | Optional — useful for reporting/QA sampling strategy only |

**Recommendation: use `AHD.xlsx` only** as the source corpus for the future health-education Q&A endpoint. `AHD_english.xlsx` can stay untouched in `data/ahd_raw/` (no need to delete it — just not part of the pipeline). The distribution file is a nice-to-have for QA/reporting on category balance, not a pipeline input.

## 9. Proposed processing plan (not executed — for your approval)

None of this has been done. If you approve, the next phase would be:

1. **PII scrub**: strip/mask self-disclosed first names (pattern-based, e.g. the `اسمي X` construction and similar) before any row is indexed or ever shown to a user.
2. **Row filtering**: drop the ~4 null rows; flag (don't necessarily drop) the 819 very-long answers and the 314 non-Arabic-question rows for manual/spot review rather than blind inclusion.
3. **Dosage/medication guard**: either exclude rows matched by the medication-keyword scan from the retrieval index, or keep them but attach a hard "do not restate dosage" instruction at answer-generation time — needs your call on which, since ~13% of the corpus qualifies.
4. **Separate, isolated storage**: keep processed output outside `rag/knowledge_base/` (per your constraint) — e.g. a new `rag/health_education/` or similar, so the diagnostic RAG path is never touched.
5. **New, separate endpoint only** (e.g. `POST /health-questions`), sitting behind Healix's existing crisis/red-flag checks unchanged, per the design already proposed in `research/ARABMEDRAG_INTEGRATION_PLAN.md`.
6. **No training** — used purely for lexical/semantic retrieval (e.g. BM25 or similar), never to fine-tune or train a diagnostic model, per your constraint.

Waiting for your go-ahead before doing any of steps 1–6, or before installing any new dependency.
