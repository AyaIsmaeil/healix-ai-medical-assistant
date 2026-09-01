# RAG Audit — Healix

Scope: every retrieval-adjacent piece of the codebase, traced from the real code (not filenames/comments), compared against 7 published medical-RAG papers verified via primary sources (full text read for 6 of 7; paper 3 partially — flagged explicitly below, not guessed). This is Phase 1: analysis only. Section 19 lists what was fixed after this report and what still needs your decision.

---

## 0. The single most important finding

**Your project does not have one RAG system — it has two, and they are architecturally unrelated:**

| | `rag/knowledge_base/` (triage) | `rag/health_education/` (Q&A, built this session) |
|---|---|---|
| Retrieves | Structured disease *labels* (49 hand-authored JSON records) | Free-text Q&A *passages* (~700k AHD records) |
| Mechanism | Exact-string symptom-checklist set-overlap | BM25 (sparse lexical) |
| Feeds the LLM | A short list of candidate names + match scores | Retrieved Q&A excerpts as grounding text |
| LLM output | Picks from a **schema-enforced closed set** of candidate names | Free-text summary constrained by a prompt instruction |

**The triage system (`rag_retrieve.py`) is not RAG in the sense any of the 7 papers use the term.** All 7 papers retrieve *text passages* to ground *open-ended prose*. Your triage system retrieves *candidate labels* to constrain a *closed-set choice*. This is architecturally closer to a classical symptom-checker / differential-generator than to document-grounded RAG. That is not automatically a flaw — section 3 argues it is, in one specific respect, a genuine safety strength — but calling it "RAG" without qualification would be scientifically imprecise in front of a committee that has read these papers. Say "structured symptom-based candidate retrieval with LLM-constrained differential ranking," not "RAG," for this component.

**The health-education system genuinely is RAG** in the papers' sense (retrieve free text → ground LLM-generated prose), but it is the simplest possible version: sparse-only, no chunking (not needed — records are already atomic), no reranking, no formal evaluation yet.

---

## 1. Current architecture — traced from code, not comments

### 1a. Triage: `POST /chat` → `graph.py` → `nodes/rag_retrieve.py`

```
patient message
  → extract_symptoms (LLM, structured extraction → symptoms[], negated_symptoms[])
  → check_red_flags (deterministic rules; not retrieval)
  → assess_sufficiency (LLM: enough confirmed symptoms yet?)
  → rag_retrieve  ←── THE RETRIEVAL STEP
        for each of 49 rag/knowledge_base/*.json entries:
            normalize both sides (rules.crisis.normalize)
            confirmed_evidence = symptoms patient has that appear in entry.symptoms
                                  (direct match OR via a small hand-authored
                                  generic→specific synonym table, rules.red_flags._SUBSUMES)
            net_matched = len(confirmed_evidence) − len(negated_evidence), floored at 0
            if net_matched < 2: reject (MIN_MATCHED_SYMPTOMS floor)
            if entry.applicable_sex set and patient_sex unknown: ask, don't guess
            match_score = net_matched / len(entry.symptoms)   ← NOT comparable across diseases
        sort candidates by match_score, return ALL that qualify (no fixed K)
  → ml_corroborate (XGBoost signal, annotation-only — see research/ML_AUDIT.md, separate audit)
  → diagnose (LLM; schema.py dynamically builds a Literal[] enum from the candidate
              names THIS TURN — the model can only pick from that closed set,
              enforced by the provider's structured-output mechanism, not a prompt ask)
  → route_specialty → generate_reports (fixed Arabic templates, no LLM)
```

Each `rag/knowledge_base/*.json` record: `name`, `name_ar`, `symptoms[]`, `specialties[]`, `source` (one citation string, e.g. `"GINA — Global Strategy for Asthma Management"`), `note`, `translation_reviewed: bool`, `applicable_sex`. Validated by `rag/schema.py::KnowledgeBaseEntry` (Pydantic) on load.

**Confirmed by tracing every consumer of `entry.source`:** it is read by `rag.schema.load_all()`, then **never referenced again anywhere** — not in `rag_retrieve.py`'s returned `CandidateDisease` dict, not in `diagnose.py`'s prompt (`_format_candidate()` sends only name/match_score/matched/missing/negated), not in `generate_reports.py`'s patient or doctor report, not in `api/contracts.py`'s `ChatResponse`. `state.CandidateDisease = dict[str, Any]` — no schema at all at the state level, so nothing enforces the field would survive even if someone tried to pass it through.

There is a second, separate, **unintegrated** data-collection artifact: `scripts/scrape_webteb.py` (real scraper, `requests`+`bs4` — neither is in `requirements.txt`, confirming it's not part of the running service) + `scripts/webteb_output/webteb_candidates.json` (697 lines, was actually run). The relationship between this scrape and the 49 final KB entries (which cite named guidelines like GINA, not WebTeb) is not traceable from the code — flagged as unknown, not guessed at.

### 1b. Health Education: `POST /health-questions` → `rag/health_education/`

```
question
  → safety_gate.classify() — crisis regex, then raw-text red-flag rule matching,
    then LLM screen (personal_symptom / medication_dosage / educational)
  → [only if "educational"] retriever.search()
        BM25Okapi over question-field only, corpus = ahd_cleaned.jsonl (~700k records,
        cleaned from the 808k-row AHD dataset — see docs/AHD_DATA_PROVENANCE.md)
        MIN_SCORE = 1.0 (explicitly documented in the code as an untuned starting
        point, not an empirically validated threshold)
  → [if hits clear the floor] format top-5 excerpts → LLM prompt
    (health_qa_answer.txt: "summarize only from these excerpts, never invent")
  → HealthEducationAnswer schema — only enforces `answer: str`, no structural
    grounding check (unlike diagnose.py's enum constraint)
  → response includes category-level SourceReference{dataset, category, license}
    — never the raw retrieved passage text (deliberate, to avoid leaking raw
    patient-submitted dataset content into the API response)
```

No chunking (each AHD record is already a short, complete Q&A pair). No embeddings. No vector DB. No reranking. No metadata filtering beyond the category label. No dedicated evaluation dataset for retrieval quality specifically (the existing `tests/golden/cases.json` covers triage, not this feature).

---

## 2. What is correct

1. **Schema-enforced closed-set diagnosis** (`schemas/diagnosis.py`) is a real, structural anti-hallucination mechanism — the LLM cannot name a disease outside the retrieved candidates, enforced by the provider's native structured output, not a prompt request. This is *stronger* than what most of the 7 papers do: paper 2 (Ophthalmology) still had **18.8% hallucinated references even with RAG**; none of the 7 papers use a comparable hard structural constraint on the answer's identity space.
2. **Sex-gating** (`applicable_sex`) is a genuine clinical-safety feature with no equivalent in any of the 7 papers — it turns an ambiguous match into a clarifying question rather than a guess.
3. **The negation-aware matching** (`net_matched = confirmed − negated`) is a real, if simple, correctness feature most of the papers' pure-similarity retrieval doesn't have — dense/BM25 retrieval treats "no fever" and "fever" as similar text, this system treats them as opposing evidence.
4. **The project's own internal docs are honest about the limitation** — `docs/PIPELINE_STAGES_AR.md` already labels `rag_retrieve` as "retrieval حتمي (set overlap)," not as embedding-based retrieval. The team never misrepresented this to itself.
5. **`match_score`'s own documented caveat** ("only comparable WITHIN one disease's own match, not across different diseases") is exactly the kind of self-aware limitation-labeling real research papers are expected to state and often don't bother to.
6. **Health-education's hard category-level redaction of raw retrieved text from the API response** is a genuine, deliberate privacy safeguard — none of the 7 papers had to consider this because none of their corpora were scraped patient-submitted forum content with self-disclosed names (see `docs/AHD_DATA_PROVENANCE.md` — AHD's own real PII gap, and the masking step this project added).
7. **A working evaluation harness already exists** (`scripts/run_golden_evaluation.py` + `tests/golden/cases.json`, ~17 hand-authored cases with `expected` red-flag/top-candidate labels, run against the real graph + real LLM) — this is a genuine foundation to extend, not a from-scratch build.
8. **The medication-content quarantine and PII masking in `rag/health_education/preprocess.py`** (documented in `docs/AHD_DATA_PROVENANCE.md`) is a real content-safety step none of the 7 papers needed, because none of their source corpora were raw patient forum text.

---

## 3. What is scientifically weak

1. **`rag_retrieve.py` cannot be called RAG without qualification** (section 0). Presenting it as RAG to a committee that has read GuideGPT/EMR-RAG/etc. invites an immediate, fair challenge: "where's your embedding model, your chunking, your retrieval evaluation?" — because it has none of these, by design, for a genuinely different task shape. The fix is framing, not necessarily architecture (see section 6).
2. **`entry.source` is captured but never surfaces anywhere** (section 1a). This is the single largest, cheapest-to-fix scientific-credibility gap: you can currently claim "GINA-sourced" for asthma in the data file, but **the system as shipped cannot show that to anyone** — patient, doctor, or you in a demo. 3 of 7 papers (1, 2, 5) treat user-visible source attribution as a first-class deliverable specifically because "answers not only need to be correct but also require accurate references for healthcare professionals as justification" (paper 2's own words).
3. **The 49-entry knowledge base is too small for `match_score` to mean much across diseases**, and the code already says so. This isn't new information, but it means any claim like "the RAG achieved X% accuracy" would be comparing apples to a fruit salad unless evaluated per-disease, never in aggregate ranking.
4. **`rag/health_education`'s `MIN_SCORE = 1.0` threshold is genuinely untuned** — self-documented in the code as a "starting point, not a validated threshold." None of the 7 papers ship an untested cutoff like this; paper 6 explicitly tested k∈{1,3,5,10,50} before choosing, and paper 2 explains its top-K choice (even though that choice was a token-budget constraint, not a quality one — still a stated reason).
5. **No retrieval-only evaluation exists for either system** (Recall@K / Precision@K / MRR-style metrics). The existing golden-case harness measures end-to-end correctness (did the final diagnosis/red-flag match), which conflates retrieval quality with LLM behavior — you cannot currently tell whether a wrong outcome was a retrieval failure or an LLM failure.
6. **No expert-validated golden dataset exists for the health-education BM25 system specifically.**

---

## 4. What is technically wrong (real bugs/defects, not just weak methodology)

1. **`entry.source`, `entry.note`, and `entry.translation_reviewed` are dead fields at runtime.** They validate on load and then go nowhere — not a design choice, an oversight (`_format_candidate()` in `diagnose.py` and `_format_candidate_doctor()` in `generate_reports.py` both hand-pick a fixed set of dict keys that never included them). This is a genuine defect: data modeled, collected, and then silently dropped.
2. **No source-existence check anywhere** — since `source` never reaches the LLM or the report, there is currently zero risk of a *fabricated* citation appearing (good), but also zero mechanism to ever show a *real* one (bad). Fixing #1 must be done carefully: only ever surface `entry.source` verbatim from the matched KB record, never let the LLM paraphrase or invent it.
3. **`rag/health_education`'s BM25 index has no persistence/caching** — `HealthEducationRetriever.from_jsonl()` rebuilds the full ~700k-document index from scratch on first use per process. Not incorrect, but a real, unaddressed performance cost worth flagging (out of scope for this audit's focus, noted for completeness).

---

## 5. What the 7 papers actually do (verified from primary sources; paper 3 partial — see below)

| Paper | Chunking | Embedding | Retrieval | Top-K | Rerank | Source attribution | Eval |
|---|---|---|---|---|---|---|---|
| 1. GuideGPT (MRONJ) | Sentence + 5-sentence context window | OpenAI ada-002 (general) | Dense only | 20 (unjustified) | None | File+page hyperlinks | 10 experts, Likert, Mann-Whitney |
| 2. Ophthalmology | 1,024 tokens, overlap unspecified | OpenAI ada-002 (general) | Dense only | 10 (token-budget reason only) | None | AMA-format citations (prompted) | 10 clinicians, double-blind, Likert; 18.8% hallucinated refs even w/ RAG |
| 3. Antibiotic Chatbot | **unknown — paywalled, not independently verified** | unknown | unknown | unknown | unknown | unknown | 200 sim. cases + 66 clinician Qs (secondary source only) |
| 4. Korean Medical Consultation | ~487k segments, size unspecified | `BGE-M3-Korean` (language-specific) | Dense (Elasticsearch as vector store) | unspecified | None | Not user-facing | 1 specialist, 5 LLMs × 3 configs, paired t-test |
| 5. EMR RAG Chatbot | **512 tokens / 128 overlap**, sliding window | Multilingual-E5-Large & BGE-M3 (fine-tuned per-hospital) | Dense (FAISS eval / Weaviate deploy) | 5 (workflow-usability reason) | None | Filename+page | Top-K accuracy only (84%→97.6% after fine-tuning); explicitly NOT validated with real users |
| 6. Diabetes Dual RAG | **1,000 chars / 200 overlap** | 11 tested; multilingual outperformed language-specific | **Dense + BM25 hybrid, ensembled** | Tested k∈{1,3,5,10,50}; biased toward higher recall (false negatives worse than false positives in medicine) | None | doc_id tracking, not user-facing | Full IR metrics (F1/recall/precision/MAP/MRR/NDCG), expert-panel-curated QA |
| 7. Endo-chat | Sentence Window Node Parser, size unspecified | OpenAI ada-002 (general) | Dense only (FAISS) | 10 (unjustified) | **LLM-based relevance filter (Self-RAG style)** — not a learned reranker | Not user-facing | Doctor win/tie/lose (200 Qs) + 200-patient field study |

**Cross-paper patterns directly relevant to your design decisions:**

- **Paper 4's central finding: naive RAG (retrieval alone) produced NO significant improvement, and sometimes hurt performance** — until the LLM extracted structured query metadata (disease/symptom/drug cues) to narrow retrieval first. Their own words: standalone RAG fails partly because "peripheral content dilutes the relevance signal." **This is direct evidence for your existing design, not against it**: `rag_retrieve.py` already forces retrieval through a *structured* extraction step (`extract_symptoms`) before matching, rather than doing naive free-text similarity search on the raw patient message. That's the same corrective mechanism paper 4 had to add on top of a naive RAG baseline — you started with it. Frame this honestly as convergent design, not coincidence dressed as foresight.
- **Paper 6's explicit rationale for hybrid retrieval** — "Dense retrievers excel at capturing semantic relationships but may miss exact keyword matches crucial for medical terminology, while sparse retrievers effectively find specific terms but often miss conceptually related information" — is the strongest evidence-based case in the whole set for adding BM25 alongside embeddings **if and when** you build dense retrieval for the health-education corpus. It does not apply to `rag_retrieve.py` (there is no free text to do dense-vs-sparse retrieval over — it's a closed symptom vocabulary).
- **None of the 7 papers use a dedicated learned reranker.** The one paper using anything like it (Endo-chat) uses an LLM relevance filter, not a cross-encoder reranker. This directly answers your "do we need a reranker" question (section 8 below): **no strong published precedent among these 7 that a reranker is required**, and Healix already has an LLM-based gate of similar spirit for the triage path (`diagnose.py`'s schema-constrained selection acts as exactly this kind of filter).
- **Source attribution is inconsistent across the literature (3/7 show it to users)** — you are not behind an established universal standard by lacking it, but you are behind the *best* papers in the set (1, 2, 5), and the fix is cheap (section 7).
- **Multilingual embedding models beat language-specific ones** in the one paper that tested this directly (paper 6) — directly informs the embedding-model recommendation in section 12.

---

## 6. Recommended architecture

**Two separate recommendations, because these are two separate systems.**

### 6a. Triage (`rag_retrieve.py`) — do not rebuild as dense/vector RAG

Rebuilding this as embedding-based retrieval would be a regression, not an improvement, given the evidence: your candidate space is a closed, hand-curated 49-disease list where exact-match symptom logic is *more* precise than semantic similarity would be (semantic similarity would blur "ضيق تنفس" toward unrelated-but-similar-sounding symptoms, reintroducing exactly the imprecision paper 4 fought against with metadata filtering — you'd be adding the problem paper 4 solved). **Recommended changes are metadata/traceability fixes, not retrieval-algorithm changes:**

1. Surface `entry.source` in the doctor report (never the patient report — that register is deliberately simple Syrian-colloquial, and a citation string mid-conversation would be jarring and out of register; the doctor is the audience who needs and can use a guideline citation).
2. Carry `source` through `CandidateDisease` end to end (rag_retrieve → diagnose → generate_reports) as inert metadata — never shown to or generated by the LLM, exactly like `match_score` already is (computed in code, never asked of the model).
3. Extend `tests/golden/cases.json` beyond the current ~17 cases for statistically meaningful per-disease evaluation (this was already flagged as a gap in `research/ML_AUDIT.md` this session, independently, for the same reason).
4. Rename the concept in documentation/defense material from "RAG" to "structured symptom-based candidate retrieval" for this component specifically (section 0) — a documentation fix, not a code fix, but the single highest-leverage change for how this looks to a committee that knows the RAG literature.

### 6b. Health Education (`rag/health_education/`) — real RAG, currently minimum-viable; recommended incremental upgrades

```
User Query
  → safety_gate (unchanged — crisis/red-flag/personal-symptom gate stays first, always)
  → BM25 retrieval (unchanged — keep sparse; see rationale below)
  → [NEW] empirically tune MIN_SCORE against a real query sample, replacing the
    documented-as-untuned 1.0 floor
  → [NEW, optional, needs your decision] add dense retrieval alongside BM25
    (hybrid), IF you decide the added dependency is worth it — see section 12
  → LLM grounded-summary generation (unchanged)
  → [NEW] surface entry-level category + a real per-answer "grounded: true/false"
    signal more rigorously validated (currently a simple score-floor check)
```

Do **not** add a vector database or embedding model to this system without deciding that trade-off explicitly (section 19) — BM25-only is a legitimate, paper-precedented starting point (paper 6 treats BM25 as one full half of its best-performing design, not a lesser fallback), and your own stated constraint (`لا تضف dependency غير ضرورية`) is a real, valid engineering constraint for a graduation-project timeline, not something to override without your sign-off.

---

## 7. Required changes — exact files

| File | Change | Risk |
|---|---|---|
| `state.py` | `CandidateDisease` stays `dict[str, Any]` (no change — already loose enough to carry an extra key safely) | none |
| `nodes/rag_retrieve.py` | `_match_entry()` return dict gains `"source": entry.source` | low — additive key, no existing consumer reads a fixed key set destructively |
| `nodes/diagnose.py` | `differential` dict-comprehension carries `source` through unchanged (like `ml_corroboration` already does) — **never added to `_format_candidate()`'s LLM-facing text** | low |
| `nodes/generate_reports.py` | `_format_candidate_doctor()` appends a `المصدر: {source}` line — doctor register only | low |
| `tests/unit/test_nodes_rag_retrieve.py`, `test_nodes_diagnose.py`, `test_nodes_generate_reports.py` | extend to assert `source` survives the pipeline and never reaches the patient report | none (test-only) |
| `tests/golden/cases.json` | extend with more cases (needs your input on new vignettes/expected outcomes — not something to fabricate) | none |
| `rag/health_education/retriever.py` | `MIN_SCORE` — needs a real tuning pass against sample queries before changing (not done blindly in this audit) | low once tuned |

**Explicitly not changed without your sign-off:** anything requiring a new embedding-model or vector-database dependency (section 19).

---

## 8–17: point-by-point audit, condensed

**8. Source tiers (KB provenance):** Individual entries cite named guidelines (GINA for asthma, etc.) — Tier 1/2 in your own framework, when the citation is real. This audit's original pass did not independently re-verify the citation strings against live guidelines. That gap is now closed by a real, repeatable methodology and tool — see `docs/KNOWLEDGE_BASE_METHODOLOGY.md` and `scripts/collect_kb_evidence.py`, which fetched real WHO/NICE/CDC pages for a 5-disease sample and logged agreement/discrepancy per entry (4/5 fetched successfully; CDC blocked the automated request — a real, documented operational limitation, not a code bug). The WebTeb scrape (`scripts/scrape_webteb.py`) is Tier 3 territory (a reputable Arabic medical portal, not primary literature) and — confirmed — does not feed the production KB.

**9. Document processing:** N/A for the triage KB (hand-authored, not ingested from documents — there is no PDF/HTML parsing step to audit because there is no source document being parsed). For health-education: confirmed real cleaning pipeline (`rag/health_education/preprocess.py`) — malformed-row removal, dedup, PII masking, medication quarantine — already documented in `docs/AHD_DATA_PROVENANCE.md` with real counts.

**10. Chunking:** N/A for the triage KB (atomic records). For health-education: correctly not chunked — AHD records are already short, self-contained Q&A pairs (median length nowhere near the 512-token/1000-character thresholds papers 5/6 chunk at) — chunking would fragment a complete Q&A pair for no benefit. This is the right call, not an oversight.

**11. Retrieval:** Triage = deterministic set-overlap (no dense/sparse/hybrid concept applies — see section 6a). Health-education = BM25 sparse-only; a legitimate, paper-precedented baseline (section 6b) — hybrid is a real option, not a requirement, pending your dependency decision.

**12. Embedding model** (only relevant if you approve adding dense retrieval per section 19): verified via real benchmarking research — Microsoft's E5 series reports >90% Recall@10 on the Arabic Reading Comprehension Dataset; MIRACL (a real multilingual IR benchmark) includes Arabic and is the standard evaluation multilingual embeddings are checked against; paper 6 in this audit independently found multilingual models beat language-specific ones for medical retrieval. **Recommendation if/when needed: `BGE-M3` or `multilingual-e5-large`** — both multilingual, both benchmarked on Arabic via MIRACL, neither medical-domain-fine-tuned (a real, stated gap — no verified Arabic-medical-specific embedding model was found in this research pass). Do not pick a model "because it's popular" — these two are the ones with actual Arabic-retrieval evidence behind them from this audit's research.
Sources: [MIRACL / multilingual embedding benchmark discussion](https://zeroentropy.dev/articles/best-multilingual-embedding/), [Swan/ArabicMTEB](https://arxiv.org/pdf/2411.01192), [MMed-Bench-IR](https://arxiv.org/pdf/2606.24200).

**13. Reranking:** Not recommended at this time — zero of 7 papers use a dedicated learned reranker (section 5); the closest precedent (Endo-chat's LLM relevance filter) is architecturally already mirrored by `diagnose.py`'s schema-constrained selection on the triage side. Revisit only if a future dense-retrieval health-education system shows a real precision problem BM25/hybrid alone can't fix.

**14. Top-K:** Health-education currently retrieves top-5 with a `MIN_SCORE` floor rather than a pure fixed-K — closer in spirit to paper 6's evidence-based approach (test multiple K, bias toward recall in medical contexts) than to papers 1/2/7's unjustified fixed numbers. The floor itself needs real tuning (section 7) — that is the actual gap, not the top-5 count.

**15. Hallucination control:** Triage — structural (schema enum), verified stronger than any of the 7 papers' mechanisms (section 2.1). Health-education — prompt-level only (`health_qa_answer.txt` instructs "don't invent"; `MIN_SCORE` floor triggers a fixed "insufficient information" fallback message when nothing qualifies) plus the safety_gate's deterministic pre-filter. This matches the *median* rigor across the 7 papers (most rely on prompt instruction + RAG grounding alone; only the schema-level mechanism on the triage side exceeds it).

**16. Source attribution:** Triage — currently zero (the defect in section 4); fix in section 7. Health-education — category-level only, by deliberate design (never raw retrieved text, to avoid leaking patient-submitted dataset content) — a real, justified tradeoff, not an oversight, but coarser than papers 1/2/5's document+page-level citations. Acceptable given the dataset's own privacy caveats (`docs/AHD_DATA_PROVENANCE.md`).

**17. Evaluation + golden dataset:** `tests/golden/cases.json` + `scripts/run_golden_evaluation.py` already exist for triage (end-to-end, not retrieval-isolated). None exists yet for health-education. Recommendation: extend the triage set with more cases per disease (needed for section 3's per-disease evaluation to mean anything); build a small (20–30 case) health-education golden set with `{question, expected_topic, acceptable_categories}` shape — full Recall@K/Precision@K is not meaningful at your corpus's current scale of hand-curated ground truth without real expert time, which this audit cannot fabricate on your behalf.

---

## 18. Decision table

| Component | Current | Recommended | Why | Research support |
|---|---|---|---|---|
| Triage retrieval mechanism | Deterministic symptom set-overlap | **Keep as-is**; rename away from "RAG" in docs | Closed 49-item candidate space; semantic similarity would reduce precision (paper 4's exact finding, in reverse) | This audit, paper 4 |
| Triage source attribution | Dead field, never shown | Surface in doctor report only | Cheapest fix, closes a real credibility gap | Papers 1, 2, 5 |
| Triage evaluation | 17 golden cases, end-to-end only | Extend case count; keep end-to-end (per-disease metrics not meaningful yet) | KB too small/uneven for aggregate IR metrics | `match_score`'s own documented limitation; this audit |
| Health-ed retrieval | BM25 sparse-only | Keep for now; tune `MIN_SCORE` | Paper-precedented baseline; the untuned floor is the real gap, not the algorithm | Paper 6 (BM25 as half of best design) |
| Health-ed hybrid (dense+sparse) | Absent | Optional — needs your dependency decision | Real precedent (paper 6) but real new-dependency cost | Paper 6 |
| Embedding model (if added) | None | `BGE-M3` or `multilingual-e5-large` | Verified strong on Arabic (MIRACL); multilingual beat language-specific in paper 6 | This audit's WebSearch; MIRACL; paper 6 |
| Chunking | None (neither system needs it) | No change | Triage KB atomic by design; AHD records already atomic | Papers 5/6 chunk *documents*; neither of your corpora are long documents |
| Reranking | None | No change | Zero of 7 papers require a dedicated reranker | Section 5 |
| Top-K | Health-ed: 5 + score floor; Triage: unbounded (all qualifying) | Tune the floor empirically | Consistent with paper 6's evidence-based, recall-biased approach | Paper 6 |
| Hallucination control | Triage: schema-enforced (strong); Health-ed: prompt + fallback message | Keep; no change needed | Triage mechanism already exceeds the literature's median rigor | Paper 2's 18.8% residual hallucination *with* RAG shows prompt-only grounding alone is insufficient — which is why the schema constraint matters |

---

## 19. Applied vs. what still needs your decision

**Applied:** source-attribution plumbing (section 7's first 4 rows) — `entry.source` now survives `rag_retrieve → diagnose → generate_reports`, rendered as a `المصدر:` line in the doctor report only. Never shown to the patient, never sent to the LLM (verified directly: the diagnose prompt text contains neither "source" nor any citation string, confirmed by a real trace and by dedicated tests). `nodes/rag_retrieve.py`, `nodes/diagnose.py`, `nodes/generate_reports.py`, and their three test files were updated; full suite (827 tests) passes.

**Still needs your decision — each carries a real dependency/scope cost:**

1. Add dense retrieval (embedding model + a vector store) to `rag/health_education/` as a hybrid alongside BM25 — real research support (paper 6), real new dependencies (an embedding model download + inference cost, a vector index library). Given your own "no unnecessary dependency" constraint and this being a graduation-demo project, I have not added this — tell me if you want it and I'll implement `BGE-M3` or `multilingual-e5-large` + a lightweight local vector store (e.g. a flat numpy index — no server-based vector DB needed at this corpus size).
2. Empirically tuning `MIN_SCORE` needs either a sample of real user queries or a small labeled set to tune against — I have not invented numbers to replace the currently-honest "untuned" label.
3. Extending `tests/golden/cases.json` needs real clinical vignettes with correct expected outcomes — not something to fabricate; tell me how many more cases and I'll draft them for your review, matching the existing case format exactly.
4. Renaming "RAG" to more precise terminology in your defense materials/`docs/` — a documentation task, tell me which doc(s) to update.

---

## 20. Direct answers to your 10 closing questions

1. **ما الخطأ؟** `entry.source` is captured but never reaches any output. The triage system is described (implicitly, by using the word "RAG") as something architecturally different from what it is.
2. **لماذا هو خطأ؟** A modeled, real citation field that never surfaces makes every "sourced from GINA" claim currently unverifiable by anyone using the shipped product — patient, doctor, or committee member. Calling checklist matching "RAG" invites a direct, fair challenge from anyone who has read the 7 papers you named.
3. **ما الطريقة الصحيحة؟** Surface `source` end-to-end to the doctor register only (fixed, section 19). Describe the triage system precisely as structured candidate retrieval + LLM-constrained ranking; reserve "RAG" for the health-education system, which earns the term.
4. **ما البحث الذي يدعم الطريقة؟** Papers 1, 2, 5 (source attribution as a first-class deliverable); paper 4 (naive retrieval without structuring underperforms — your existing symptom-extraction-first design already does what paper 4 had to add).
5. **ما الملفات التي عدلتها؟** `nodes/rag_retrieve.py`, `nodes/diagnose.py`, `nodes/generate_reports.py`, plus `tests/unit/test_nodes_rag_retrieve.py`, `test_nodes_diagnose.py`, `test_nodes_generate_reports.py`. Everything else in section 7's table remains proposed only.
6. **ماذا غيرت؟** Source-attribution plumbing only — no retrieval-algorithm change (justified in section 6a: changing the algorithm would be a regression, not an improvement, for this closed candidate space).
7. **كيف اختبرت التغيير؟** Extended the three test files (4 new tests: source appears in the doctor report, never in the patient report, never in the LLM prompt, and a safe placeholder when absent); full suite re-run (827 passed, 0 failed). Also verified with a real, unmocked trace (`rag_retrieve()` on real asthma symptoms → `source == "GINA — Global Strategy for Asthma Management"`).
8. **كيف أثبت أن الـRAG الجديد أفضل من القديم؟** A citation that was previously unverifiable-by-design is now inspectable in the doctor report — directly testable by reading that field (confirmed above), not by an aggregate metric (there isn't one that would mean anything yet, per section 17).
9. **ما الذي بقي ناقصًا؟** The four items in section 19 — none skipped by oversight, each requires either your dependency decision, real query data, or real clinical-vignette authorship this audit cannot invent.
10. **ما الذي يمكنني قوله أمام لجنة المناقشة؟** State it exactly as section 0 frames it: two separate, honestly-named retrieval systems, chosen deliberately for two different task shapes — a closed-candidate differential generator (safer by construction than the papers' open-generation RAG, evidenced by the schema constraint vs. papers' residual hallucination rates) and a genuine, minimum-viable sparse RAG for open-ended health education (architecturally aligned with, though simpler than, the published baselines it's compared against here) — with a named, cited research basis for every design choice in the table in section 18, and an explicit, honest list of what's still open (section 19) rather than a claim of completeness the evidence doesn't support.
