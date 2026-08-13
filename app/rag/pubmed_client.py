"""
عميل PubMed عبر NCBI Entrez E-utilities (esearch + efetch).

المصدر: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlencode

import httpx

from app.rag.harvest_config import HarvestConfig, harvest_config

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass(frozen=True)
class PubMedArticle:
    """مقالة PubMed مع ملخّص وبيانات وصفية."""

    pmid: str
    title: str
    abstract: str
    journal: Optional[str] = None
    year: Optional[str] = None
    authors: tuple[str, ...] = field(default_factory=tuple)
    mesh_terms: tuple[str, ...] = field(default_factory=tuple)
    query_disease: Optional[str] = None


class PubMedClient:
    """بحث وجلب مقالات PubMed مع احترام حدود معدّل NCBI."""

    def __init__(
        self,
        config: HarvestConfig = harvest_config,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=60.0)
        self._last_request_at: float = 0.0

    def search_and_fetch(
        self,
        disease: str,
        max_results: Optional[int] = None,
    ) -> List[PubMedArticle]:
        """يبحث عن مراجعات/أبحاث للمرض ويجلب الملخصات."""
        limit = max_results or self._config.pubmed_max_results_per_disease
        query = self._build_query(disease)
        pmids = self._esearch(query, retmax=limit)
        if not pmids:
            logger.warning("لا نتائج PubMed للمرض: %s", disease)
            return []
        articles = self._efetch(pmids)
        return [self._attach_query(article, disease) for article in articles]

    def fetch_batch(self, diseases: Sequence[str], max_results: Optional[int] = None) -> List[PubMedArticle]:
        """يجلب مقالات لعدة أمراض."""
        all_articles: List[PubMedArticle] = []
        for disease in diseases:
            logger.info("PubMed: البحث عن «%s»...", disease)
            articles = self.search_and_fetch(disease, max_results=max_results)
            logger.info("PubMed: %d مقالة لـ «%s».", len(articles), disease)
            all_articles.extend(articles)
        return all_articles

    # ------------------------------------------------------------------
    # Entrez
    # ------------------------------------------------------------------
    def _build_query(self, disease: str) -> str:
        # مراجعات + مقالات سريرية لرفع جودة الملخصات
        return f'("{disease}"[Title/Abstract]) AND (Review[pt] OR Clinical Trial[pt] OR "systematic review"[Title/Abstract])'

    def _common_params(self) -> dict[str, str]:
        params = {"email": self._config.ncbi_email, "tool": "healix_harvest"}
        if self._config.ncbi_api_key:
            params["api_key"] = self._config.ncbi_api_key
        return params

    def _throttle(self) -> None:
        delay = self._config.pubmed_request_delay_seconds
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_at = time.monotonic()

    def _esearch(self, query: str, retmax: int) -> List[str]:
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "retmode": "json",
            "sort": "relevance",
        }
        self._throttle()
        url = f"{ESEARCH_URL}?{urlencode(params)}"
        response = self._client.get(url)
        response.raise_for_status()
        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        return [str(pmid) for pmid in id_list]

    def _efetch(self, pmids: Iterable[str]) -> List[PubMedArticle]:
        pmid_list = list(pmids)
        if not pmid_list:
            return []

        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml",
        }
        self._throttle()
        url = f"{EFETCH_URL}?{urlencode(params)}"
        response = self._client.get(url)
        response.raise_for_status()
        return self._parse_pubmed_xml(response.text)

    @staticmethod
    def _attach_query(article: PubMedArticle, disease: str) -> PubMedArticle:
        return PubMedArticle(
            pmid=article.pmid,
            title=article.title,
            abstract=article.abstract,
            journal=article.journal,
            year=article.year,
            authors=article.authors,
            mesh_terms=article.mesh_terms,
            query_disease=disease,
        )

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> List[PubMedArticle]:
        root = ET.fromstring(xml_text)
        articles: List[PubMedArticle] = []

        for article_el in root.findall(".//PubmedArticle"):
            pmid_el = article_el.find(".//PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

            title_el = article_el.find(".//ArticleTitle")
            title = _text(title_el)

            abstract_parts: List[str] = []
            for abstract_text in article_el.findall(".//AbstractText"):
                label = abstract_text.get("Label")
                chunk = _text(abstract_text)
                if label and chunk:
                    abstract_parts.append(f"{label}: {chunk}")
                elif chunk:
                    abstract_parts.append(chunk)
            abstract = "\n".join(abstract_parts).strip()

            journal_el = article_el.find(".//Journal/Title")
            journal = _text(journal_el) or None

            year_el = article_el.find(".//Journal/JournalIssue/PubDate/Year")
            if year_el is None:
                year_el = article_el.find(".//ArticleDate/Year")
            if year_el is None:
                year_el = article_el.find(".//PubDate/Year")
            year = _text(year_el) or None

            authors = tuple(
                _text(author.find("LastName"))
                for author in article_el.findall(".//Author")
                if author.find("LastName") is not None
            )

            mesh_terms = tuple(
                _text(mesh.find("DescriptorName"))
                for mesh in article_el.findall(".//MeshHeading")
                if mesh.find("DescriptorName") is not None
            )

            if not pmid:
                continue

            articles.append(
                PubMedArticle(
                    pmid=pmid,
                    title=title or f"PMID {pmid}",
                    abstract=abstract,
                    journal=journal,
                    year=year,
                    authors=authors,
                    mesh_terms=mesh_terms,
                )
            )

        return articles


def _text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    parts: List[str] = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        if child.text:
            parts.append(child.text.strip())
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(part for part in parts if part).strip()
