"""Health Education Q&A — a separate feature from the medical-triage graph.

Built on the AHD (Arabic Healthcare Dataset, Mendeley Data v5,
DOI 10.17632/mgj29ndgrk.5, CC BY 4.0 — see docs/AHD_DATA_PROVENANCE.md).
Answers general health-education questions only. Never diagnoses, never
touches triage severity, never overrides an emergency decision, and
shares no state with graph.py/state.py.

Modules:
    preprocess.py  — cleans data/ahd_raw/AHD.xlsx into data/ahd_cleaned.jsonl
    retriever.py   — BM25 retrieval over the cleaned corpus
    safety_gate.py — crisis/red-flag/personal-symptom/medication routing,
                     run BEFORE any retrieval
    service.py     — orchestrates safety_gate + retriever + LLM summary
                     into api.health_qa_contracts.HealthQuestionResponse
"""
