"""
حفظ المعرفة المستخرجة كـ JSON و Markdown داخل knowledge_base/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.extraction_schema import ExtractedMedicalKnowledge
from app.rag.harvest_config import HarvestConfig, harvest_config
from app.rag.pubmed_client import PubMedArticle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeDocument:
    """وثيقة جاهزة للتخزين والفهرسة."""

    doc_id: str
    article: PubMedArticle
    knowledge: ExtractedMedicalKnowledge
    json_path: Path
    markdown_path: Path


class KnowledgeStorage:
    """يكتب ملفات JSON/Markdown ويُحدّث manifest."""

    def __init__(self, config: HarvestConfig = harvest_config) -> None:
        self._config = config
        self._ensure_dirs()

    def save_batch(
        self,
        articles: List[PubMedArticle],
        knowledge_items: List[ExtractedMedicalKnowledge],
    ) -> List[KnowledgeDocument]:
        if len(articles) != len(knowledge_items):
            raise ValueError("articles و knowledge_items يجب أن يكونا بنفس الطول.")

        documents: List[KnowledgeDocument] = []
        for article, knowledge in zip(articles, knowledge_items):
            documents.append(self.save_one(article, knowledge))

        self._write_manifest(documents)
        return documents

    def save_one(
        self,
        article: PubMedArticle,
        knowledge: ExtractedMedicalKnowledge,
    ) -> KnowledgeDocument:
        doc_id = f"pmid_{article.pmid}"
        json_path = self._config.extracted_json_dir / f"{doc_id}.json"
        markdown_path = self._config.markdown_dir / f"{doc_id}.md"
        raw_path = self._config.raw_pubmed_dir / f"{doc_id}.json"

        payload = self._build_json_payload(article, knowledge)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._build_markdown(article, knowledge),
            encoding="utf-8",
        )
        raw_path.write_text(
            json.dumps(self._article_to_dict(article), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("حُفظت الوثيقة %s", doc_id)
        document = KnowledgeDocument(
            doc_id=doc_id,
            article=article,
            knowledge=knowledge,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        self._write_manifest([document])
        return document

    def _ensure_dirs(self) -> None:
        for directory in (
            self._config.knowledge_base_dir,
            self._config.extracted_json_dir,
            self._config.markdown_dir,
            self._config.raw_pubmed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_json_payload(
        article: PubMedArticle,
        knowledge: ExtractedMedicalKnowledge,
    ) -> Dict[str, Any]:
        return {
            "doc_id": f"pmid_{article.pmid}",
            "harvested_at": datetime.now(timezone.utc).isoformat(),
            "source": KnowledgeStorage._article_to_dict(article),
            "extracted": knowledge.model_dump(),
        }

    @staticmethod
    def _article_to_dict(article: PubMedArticle) -> Dict[str, Any]:
        return {
            "pmid": article.pmid,
            "title": article.title,
            "abstract": article.abstract,
            "journal": article.journal,
            "year": article.year,
            "authors": list(article.authors),
            "mesh_terms": list(article.mesh_terms),
            "query_disease": article.query_disease,
        }

    @staticmethod
    def _build_markdown(
        article: PubMedArticle,
        knowledge: ExtractedMedicalKnowledge,
    ) -> str:
        symptoms = "\n".join(f"- {symptom}" for symptom in knowledge.associated_symptoms) or "- (none extracted)"
        red_flags = "\n".join(f"- {flag}" for flag in knowledge.red_flags) or "- (none extracted)"
        authors = ", ".join(article.authors[:8]) if article.authors else "N/A"

        return (
            f"# {knowledge.disease_name}\n\n"
            f"## Source (PubMed)\n\n"
            f"- **PMID**: [{article.pmid}](https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/)\n"
            f"- **Title**: {article.title}\n"
            f"- **Journal**: {article.journal or 'N/A'} ({article.year or 'N/A'})\n"
            f"- **Authors**: {authors}\n"
            f"- **Search disease**: {article.query_disease or 'N/A'}\n\n"
            f"## Extracted Knowledge\n\n"
            f"- **Medical specialty**: {knowledge.medical_specialty}\n"
            f"- **Triage level**: {knowledge.triage_level}\n"
            f"- **Extraction mode**: {knowledge.extraction_mode or 'unknown'}\n\n"
            f"### Associated symptoms\n\n"
            f"{symptoms}\n\n"
            f"### Red flags\n\n"
            f"{red_flags}\n\n"
            f"### Clinical summary\n\n"
            f"{knowledge.clinical_summary or '(empty)'}\n\n"
            f"## Abstract\n\n"
            f"{article.abstract or '(no abstract)'}\n"
        )

    def _write_manifest(self, documents: List[KnowledgeDocument]) -> None:
        manifest_path = self._config.knowledge_base_dir / "manifest.json"
        existing: Dict[str, Any] = {"documents": []}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("manifest.json تالف — سيتم إعادة بنائه.")

        by_id = {item["doc_id"]: item for item in existing.get("documents", []) if "doc_id" in item}
        for doc in documents:
            by_id[doc.doc_id] = {
                "doc_id": doc.doc_id,
                "pmid": doc.article.pmid,
                "disease_name": doc.knowledge.disease_name,
                "triage_level": doc.knowledge.triage_level,
                "json_path": str(doc.json_path.relative_to(self._config.knowledge_base_dir)),
                "markdown_path": str(doc.markdown_path.relative_to(self._config.knowledge_base_dir)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        manifest = {
            "version": "1.0.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "document_count": len(by_id),
            "documents": sorted(by_id.values(), key=lambda item: item["doc_id"]),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_all_from_disk(self) -> List[KnowledgeDocument]:
        """يحمّل كل الوثائق من knowledge_base/extracted/*.json للفهرسة."""
        extracted_dir = self._config.extracted_json_dir
        if not extracted_dir.exists():
            return []

        documents: List[KnowledgeDocument] = []
        for json_path in sorted(extracted_dir.glob("pmid_*.json")):
            doc = self._load_document(json_path)
            if doc is not None:
                documents.append(doc)

        logger.info("حُمّلت %d وثيقة من %s", len(documents), extracted_dir)
        return documents

    def _load_document(self, json_path: Path) -> Optional[KnowledgeDocument]:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("تخطّي %s: %s", json_path.name, exc)
            return None

        source = payload.get("source") or {}
        extracted = payload.get("extracted") or {}
        if not source.get("pmid"):
            return None

        article = PubMedArticle(
            pmid=str(source["pmid"]),
            title=str(source.get("title") or ""),
            abstract=str(source.get("abstract") or ""),
            journal=source.get("journal"),
            year=source.get("year"),
            authors=tuple(source.get("authors") or []),
            mesh_terms=tuple(source.get("mesh_terms") or []),
            query_disease=source.get("query_disease"),
        )
        knowledge = ExtractedMedicalKnowledge.model_validate(extracted)
        doc_id = payload.get("doc_id") or f"pmid_{article.pmid}"
        markdown_path = self._config.markdown_dir / f"{doc_id}.md"

        return KnowledgeDocument(
            doc_id=doc_id,
            article=article,
            knowledge=knowledge,
            json_path=json_path,
            markdown_path=markdown_path,
        )
