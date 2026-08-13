"""
فهرسة المعرفة في ChromaDB (Vector DB) للاسترجاع لاحقاً في RAG.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.rag.harvest_config import HarvestConfig, harvest_config
from app.rag.storage import KnowledgeDocument

logger = logging.getLogger(__name__)


class MedicalVectorStore:
    """يُدير مجموعة ChromaDB دائمة للمعرفة الطبية."""

    def __init__(
        self,
        config: HarvestConfig = harvest_config,
        reset: bool = False,
    ) -> None:
        self._config = config
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._reset = reset

    def index_documents(self, documents: List[KnowledgeDocument]) -> int:
        """يُفهرس الوثائق ويُعيد عدد السجلات المُضافة/المُحدَّثة."""
        if not documents:
            return 0

        collection = self._get_collection()
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            ids.append(doc.doc_id)
            texts.append(self._document_text(doc))
            metadatas.append(self._document_metadata(doc))

        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("ChromaDB: فُهرست %d وثيقة في «%s».", len(ids), self._config.chroma_collection_name)
        return len(ids)

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """استعلام Vector DB — يرفع استثناءً عند غياب chromadb أو الفهرس."""
        collection = self._get_collection()
        return collection.query(query_texts=[query_text], n_results=n_results)

    def try_query(self, query_text: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """استعلام آمن لمسار API — يُعيد None بدل إنهاء العملية."""
        try:
            return self.query(query_text, n_results=n_results)
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return None

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as exc:
            raise RuntimeError(
                "chromadb غير مثبّت — pip install chromadb sentence-transformers"
            ) from exc

        self._config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._config.chroma_persist_dir))
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self._config.embedding_model
        )

        if self._reset:
            try:
                self._client.delete_collection(self._config.chroma_collection_name)
                logger.info("ChromaDB: حُذفت المجموعة القديمة «%s».", self._config.chroma_collection_name)
            except Exception:
                pass

        self._collection = self._client.get_or_create_collection(
            name=self._config.chroma_collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    @staticmethod
    def _document_text(doc: KnowledgeDocument) -> str:
        knowledge = doc.knowledge
        symptoms = ", ".join(knowledge.associated_symptoms)
        red_flags = ", ".join(knowledge.red_flags)
        return (
            f"Disease: {knowledge.disease_name}\n"
            f"Specialty: {knowledge.medical_specialty}\n"
            f"Triage: {knowledge.triage_level}\n"
            f"Symptoms: {symptoms}\n"
            f"Red flags: {red_flags}\n"
            f"Summary: {knowledge.clinical_summary}\n"
            f"Title: {doc.article.title}\n"
            f"Abstract: {doc.article.abstract}"
        )

    @staticmethod
    def _document_metadata(doc: KnowledgeDocument) -> Dict[str, Any]:
        return {
            "doc_id": doc.doc_id,
            "pmid": doc.article.pmid,
            "disease_name": doc.knowledge.disease_name,
            "medical_specialty": doc.knowledge.medical_specialty,
            "triage_level": doc.knowledge.triage_level,
            "query_disease": doc.article.query_disease or "",
            "extraction_mode": doc.knowledge.extraction_mode or "",
            "json_path": str(doc.json_path),
            "markdown_path": str(doc.markdown_path),
        }
