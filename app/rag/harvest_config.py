"""
إعدادات سكربت جمع المعرفة الطبية (من متغيّرات البيئة).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

# أمراض شائعة إضافية ليست ضمن DDXPlus — تُدمَج مع قائمة الـ49 عند التحميل
_SUPPLEMENTARY_DISEASES: tuple[str, ...] = (
    "Hypertension",
    "Myocardial Infarction",
    "Appendicitis",
    "Asthma",
    "Diabetes Mellitus",
    "Stroke",
    "Heart Failure",
    "Chronic Kidney Disease",
    "Migraine",
    "Deep Vein Thrombosis",
)

# احتياطي عند غياب release_conditions.json (بيئة بلا DDXPlus)
_FALLBACK_DISEASES: tuple[str, ...] = (
    "Hypertension",
    "Myocardial Infarction",
    "Appendicitis",
    "Asthma",
    "Diabetes Mellitus",
    "Pneumonia",
    "Pulmonary Embolism",
    "Atrial Fibrillation",
    "Stroke",
    "Influenza",
    "Gastroesophageal Reflux Disease",
    "Anaphylaxis",
    "Acute Pancreatitis",
    "Urinary Tract Infection",
    "Migraine",
)


def _load_ddxplus_disease_names() -> tuple[str, ...]:
    """يحمّل أسماء الأمراض الـ49 من DDXPlus (cond-name-eng)."""
    conditions_path = REPO_ROOT / "app" / "data" / "raw" / "ddxplus" / "release_conditions.json"
    if not conditions_path.exists():
        return ()

    import json

    data = json.loads(conditions_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for entry in data.values():
        eng = (entry.get("cond-name-eng") or entry.get("condition_name") or "").strip()
        if eng:
            names.append(eng)
    return tuple(sorted(set(names)))


def _build_default_diseases() -> tuple[str, ...]:
    """
    قائمة الأمراض الافتراضية للجمع:
    - 49 مرض DDXPlus (إن وُجد release_conditions.json)
    - + أمراض شائعة إضافية غير مكرّرة
    """
    use_ddxplus = os.getenv("HARVEST_USE_DDXPLUS_DISEASES", "true").lower() == "true"
    ddxplus = _load_ddxplus_disease_names() if use_ddxplus else ()

    if ddxplus:
        merged: dict[str, None] = {name: None for name in ddxplus}
        for name in _SUPPLEMENTARY_DISEASES:
            merged.setdefault(name, None)
        return tuple(sorted(merged.keys()))

    return _FALLBACK_DISEASES


_DEFAULT_DISEASES = _build_default_diseases()


@dataclass(frozen=True)
class HarvestConfig:
    """إعدادات PubMed + LLM + التخزين المحلي."""

    # --- PubMed (NCBI Entrez) ---
    # البريد إلزامي وفق سياسة NCBI — https://www.ncbi.nlm.nih.gov/home/about/policies/
    ncbi_email: str = os.getenv("NCBI_EMAIL", "healix-dev@example.com").strip()
    ncbi_api_key: str = os.getenv("NCBI_API_KEY", "").strip()
    pubmed_max_results_per_disease: int = int(os.getenv("PUBMED_MAX_RESULTS", "5"))
    pubmed_request_delay_seconds: float = float(os.getenv("PUBMED_REQUEST_DELAY", "0.4"))

    # --- مسارات knowledge_base ---
    knowledge_base_dir: Path = field(
        default_factory=lambda: REPO_ROOT / os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    )
    chroma_persist_dir: Path = field(default_factory=lambda: Path())  # يُحسب في __post_init__
    extracted_json_dir: Path = field(default_factory=lambda: Path())
    markdown_dir: Path = field(default_factory=lambda: Path())
    raw_pubmed_dir: Path = field(default_factory=lambda: Path())

    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "healix_medical_knowledge")
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- أمراض افتراضية للبحث ---
    # 49 مرض DDXPlus + 10 أمراض شائعة إضافية (≈54) — أو _FALLBACK_DISEASES
    default_diseases: tuple[str, ...] = _DEFAULT_DISEASES


def _build_harvest_config() -> HarvestConfig:
    base = REPO_ROOT / os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    return HarvestConfig(
        knowledge_base_dir=base,
        chroma_persist_dir=base / "chroma",
        extracted_json_dir=base / "extracted",
        markdown_dir=base / "markdown",
        raw_pubmed_dir=base / "raw_pubmed",
    )


harvest_config = _build_harvest_config()
