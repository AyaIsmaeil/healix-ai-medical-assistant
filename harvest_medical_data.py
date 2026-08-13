#!/usr/bin/env python3
"""
Healix — جمع المعرفة الطبية من PubMed وبناء قاعدة RAG

يجلب ملخصات أبحاث PubMed لأمراض شائعة، يستخرج حقولاً منظَّمة عبر LLM
(OpenRouter/Qwen)، يحفظ JSON/Markdown في knowledge_base/، ويفهرس ChromaDB.

Usage:
  python harvest_medical_data.py
  python harvest_medical_data.py --diseases "Hypertension,Asthma" --max-per-disease 3
  python harvest_medical_data.py --heuristic-only   # بلا LLM (تطوير/اختبار)
  python harvest_medical_data.py --skip-vector      # حفظ ملفات فقط
  python harvest_medical_data.py --reset-vector     # إعادة بناء فهرس ChromaDB

  # الافتراضي: 49 مرض DDXPlus + 10 أمراض شائعة إضافية (≈54)
  # لتعطيل DDXPlus: HARVEST_USE_DDXPLUS_DISEASES=false

متغيّرات البيئة المهمة:
  NCBI_EMAIL          — مطلوب وفق سياسة NCBI
  NCBI_API_KEY        — اختياري (يرفع حد المعدّل)
  LLM_PROVIDER        — qwen_openrouter للاستخراج الحقيقي، mock للوضع التجريبي
  OPENROUTER_API_KEY  — مطلوب عند LLM_PROVIDER=qwen_openrouter
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# ضمان استيراد app.* عند التشغيل من جذر المستودع
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import config
from app.rag.harvest_config import harvest_config
from app.rag.knowledge_extractor import MedicalKnowledgeExtractor
from app.rag.pubmed_client import PubMedClient
from app.rag.storage import KnowledgeStorage
from app.rag.vector_store import MedicalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("harvest_medical_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="جمع المعرفة الطبية من PubMed وبناء knowledge_base + ChromaDB",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        default="",
        help="قائمة أمراض مفصولة بفاصلة (افتراضي: الأمراض الخمسة الشائعة)",
    )
    parser.add_argument(
        "--max-per-disease",
        type=int,
        default=harvest_config.pubmed_max_results_per_disease,
        help="أقصى عدد مقالات PubMed لكل مرض",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="استخراج تجريبي بلا LLM (للتطوير)",
    )
    parser.add_argument(
        "--skip-vector",
        action="store_true",
        help="تخطّي فهرسة ChromaDB (حفظ JSON/Markdown فقط)",
    )
    parser.add_argument(
        "--reset-vector",
        action="store_true",
        help="حذف مجموعة ChromaDB وإعادة بنائها",
    )
    parser.add_argument(
        "--reindex-only",
        action="store_true",
        help="فهرسة ChromaDB من الملفات المحفوظة في knowledge_base/extracted/ (بدون PubMed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="جلب PubMed فقط بدون استخراج LLM أو فهرسة",
    )
    return parser.parse_args()


def resolve_diseases(raw: str) -> List[str]:
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(harvest_config.default_diseases)


def main() -> int:
    args = parse_args()

    if args.reindex_only:
        logger.info("=== إعادة فهرسة ChromaDB من knowledge_base/extracted/ ===")
        storage = KnowledgeStorage()
        documents = storage.load_all_from_disk()
        if not documents:
            logger.error("لا توجد ملفات في knowledge_base/extracted/ — شغّل harvest أولاً.")
            return 1
        vector_store = MedicalVectorStore(reset=args.reset_vector)
        indexed = vector_store.index_documents(documents)
        total = vector_store.count()
        logger.info("ChromaDB: فُهرست %d وثيقة (إجمالي: %d).", indexed, total)
        logger.info("=== اكتمل — أعد تشغيل API وتحقّق من /api/health ===")
        return 0

    diseases = resolve_diseases(args.diseases)

    logger.info("=== Healix Medical Knowledge Harvest ===")
    logger.info("Knowledge base: %s", harvest_config.knowledge_base_dir)
    logger.info("Diseases: %s", ", ".join(diseases))
    logger.info("Max per disease: %d", args.max_per_disease)
    logger.info("LLM provider: %s", config.LLM_PROVIDER)

    if not harvest_config.ncbi_email or harvest_config.ncbi_email.endswith("@example.com"):
        logger.warning(
            "NCBI_EMAIL غير مضبوط أو افتراضي — عيّنه في .env وفق سياسة NCBI."
        )

    pubmed = PubMedClient()
    articles = pubmed.fetch_batch(diseases, max_results=args.max_per_disease)
    if not articles:
        logger.error("لم تُجلب أي مقالات — تحقّق من الاتصال أو استعلامات الأمراض.")
        return 1

    logger.info("إجمالي المقالات المجلوبة: %d", len(articles))

    if args.dry_run:
        for article in articles:
            logger.info("PMID=%s | %s", article.pmid, article.title[:80])
        return 0

    use_llm = not args.heuristic_only
    if use_llm and (config.LLM_PROVIDER or "mock").strip().lower() == "mock":
        logger.warning(
            "LLM_PROVIDER=mock — سيُستخدم الاستخراج التجريبي. "
            "للاستخراج الحقيقي: LLM_PROVIDER=qwen_openrouter و OPENROUTER_API_KEY."
        )

    extractor = MedicalKnowledgeExtractor(use_llm=use_llm)
    knowledge_items = extractor.extract_batch(articles)

    storage = KnowledgeStorage()
    documents = storage.save_batch(articles, knowledge_items)
    logger.info("حُفظت %d وثيقة في knowledge_base/", len(documents))

    if args.skip_vector:
        logger.info("تخطّي فهرسة ChromaDB (--skip-vector).")
        return 0

    vector_store = MedicalVectorStore(reset=args.reset_vector)
    indexed = vector_store.index_documents(documents)
    total = vector_store.count()
    logger.info("ChromaDB: %d وثيقة مفهرسة (إجمالي المجموعة: %d).", indexed, total)

    # استعلام تحقّق سريع
    sample_query = diseases[0]
    try:
        preview = vector_store.query(sample_query, n_results=2)
        ids = preview.get("ids", [[]])[0]
        logger.info("استعلام تجريبي «%s» → %s", sample_query, ids)
    except Exception as exc:
        logger.warning("تعذّر الاستعلام التجريبي: %s", exc)

    logger.info("=== اكتمل بناء قاعدة المعرفة ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
