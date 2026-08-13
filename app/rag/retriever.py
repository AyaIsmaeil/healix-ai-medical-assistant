"""
استرجاع المعرفة الطبية من ChromaDB لمسار التقييم الهجين.

القواعد + ML + YAML تبقى مصدر القرار؛ RAG يُغذّي **الشرح** فقط بمراجع PubMed.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.domain.assessment import ClinicalFeatureSet
from app.domain.prediction import DiseasePredictionResult
from app.domain.rag_context import RagSource
from app.rag.harvest_config import HarvestConfig, harvest_config
from app.rag.vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)


class MedicalKnowledgeRetriever:
    """يسترجع مقاطع PubMed ذات صلة — بلا تأثير على التنبؤ/الخطورة/التخصّص."""

    def __init__(
        self,
        enabled: bool = True,
        top_k: int = 3,
        config: HarvestConfig = harvest_config,
    ) -> None:
        self._enabled = enabled
        self._top_k = max(1, top_k)
        self._config = config
        self._store: Optional[MedicalVectorStore] = None
        self._ready: Optional[bool] = None

    def is_ready(self) -> bool:
        """هل RAG جاهز (مفعّل + فهرس ChromaDB موجود وغير فارغ)؟"""
        if not self._enabled:
            return False
        if self._ready is not None:
            return self._ready
        self._ready = self._probe_ready()
        return self._ready

    def retrieve_for_assessment(
        self,
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
    ) -> List[RagSource]:
        """يسترجع أعلى ``top_k`` مقاطع — قائمة فارغة عند التعطيل أو الفشل."""
        if not self.is_ready():
            return []

        query = self._build_query(clinical_features, prediction_result)
        if not query.strip():
            return []

        try:
            store = self._get_store()
            raw = store.try_query(query, n_results=self._top_k)
            if raw is None:
                return []
        except Exception as exc:
            logger.warning("فشل استرجاع RAG (يُتابع بدون سياق): %s", exc)
            return []

        return self._parse_results(raw)

    @staticmethod
    def _build_query(
        clinical_features: ClinicalFeatureSet,
        prediction_result: DiseasePredictionResult,
    ) -> str:
        parts: List[str] = []

        if prediction_result.predictions:
            top = max(prediction_result.predictions, key=lambda item: item.score)
            parts.append(top.disease)

        for symptom in clinical_features.symptoms:
            if not symptom.negated and symptom.name.strip():
                parts.append(symptom.name.strip())

        if clinical_features.derived.has_red_flag:
            parts.append("emergency red flags urgent care")

        primary = next((s for s in clinical_features.symptoms if s.is_primary), None)
        if primary and primary.name.strip():
            parts.insert(0, primary.name.strip())

        # إزالة التكرار مع الحفاظ على الترتيب
        seen: set[str] = set()
        unique: List[str] = []
        for part in parts:
            key = part.lower()
            if key not in seen:
                seen.add(key)
                unique.append(part)

        return " ".join(unique[:8])

    def _probe_ready(self) -> bool:
        chroma_dir = self._config.chroma_persist_dir
        if not chroma_dir.exists():
            logger.info("RAG: مجلد ChromaDB غير موجود (%s) — معطّل.", chroma_dir)
            return False
        try:
            count = self._get_store().count()
        except Exception as exc:
            logger.warning("RAG: تعذّر فتح ChromaDB — %s", exc)
            return False
        if count <= 0:
            logger.info("RAG: فهرس ChromaDB فارغ — شغّل harvest_medical_data.py.")
            return False
        logger.info("RAG: جاهز (%d وثيقة في ChromaDB).", count)
        return True

    def _get_store(self) -> MedicalVectorStore:
        if self._store is None:
            self._store = MedicalVectorStore(config=self._config)
        return self._store

    @staticmethod
    def _parse_results(raw: dict) -> List[RagSource]:
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        sources: List[RagSource] = []
        for index, doc_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) else ""
            pmid = str(metadata.get("pmid") or "").strip() or doc_id.replace("pmid_", "")

            distance = distances[index] if index < len(distances) else None
            relevance = None if distance is None else round(max(0.0, 1.0 - float(distance)), 4)

            snippet = (document or "")[:500].strip()
            sources.append(
                RagSource(
                    doc_id=str(metadata.get("doc_id") or doc_id),
                    pmid=pmid,
                    disease_name=str(metadata.get("disease_name") or ""),
                    medical_specialty=str(metadata.get("medical_specialty") or ""),
                    triage_level=str(metadata.get("triage_level") or ""),
                    snippet=snippet,
                    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    relevance_score=relevance,
                )
            )
        return sources
