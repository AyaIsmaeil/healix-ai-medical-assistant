"""اختبارات وحدة جمع المعرفة الطبية (RAG harvest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.extraction_schema import (
    ExtractedMedicalKnowledge,
    KnowledgeExtractionError,
    validate_knowledge_extraction_json,
    validate_knowledge_extraction_shape,
)
from app.rag.harvest_config import HarvestConfig, _FALLBACK_DISEASES, _load_ddxplus_disease_names, harvest_config
from app.rag.knowledge_extractor import MedicalKnowledgeExtractor
from app.rag.pubmed_client import PubMedArticle, PubMedClient
from app.rag.storage import KnowledgeStorage


SAMPLE_PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345678</PMID>
      <Article>
        <ArticleTitle>Acute appendicitis: diagnosis and management</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Appendicitis presents with abdominal pain.</AbstractText>
          <AbstractText Label="RESULTS">Fever and nausea are common symptoms.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName></Author>
        </AuthorList>
        <Journal>
          <Title>Annals of Surgery</Title>
          <JournalIssue>
            <PubDate><Year>2020</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <History><PubMedPubDate><Year>2020</Year></PubMedPubDate></History>
    </PubmedData>
    <MeshHeadingList>
      <MeshHeading><DescriptorName>Appendicitis</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_pubmed_xml_extracts_fields():
    articles = PubMedClient._parse_pubmed_xml(SAMPLE_PUBMED_XML)
    assert len(articles) == 1
    article = articles[0]
    assert article.pmid == "12345678"
    assert "appendicitis" in article.title.lower()
    assert "abdominal pain" in article.abstract.lower()
    assert article.journal == "Annals of Surgery"
    assert article.year == "2020"
    assert article.authors == ("Smith",)
    assert "Appendicitis" in article.mesh_terms


def test_validate_knowledge_extraction_json_accepts_valid_payload():
    payload = json.dumps(
        {
            "disease_name": "Appendicitis",
            "associated_symptoms": ["abdominal pain", "fever"],
            "medical_specialty": "General Surgery",
            "triage_level": "Urgent",
            "clinical_summary": "Acute appendicitis requires prompt evaluation.",
            "red_flags": ["severe abdominal pain"],
        }
    )
    knowledge = validate_knowledge_extraction_json(payload)
    assert isinstance(knowledge, ExtractedMedicalKnowledge)
    assert knowledge.triage_level == "Urgent"
    validate_knowledge_extraction_shape(payload)


def test_validate_knowledge_extraction_json_rejects_invalid_triage():
    payload = json.dumps(
        {
            "disease_name": "Test",
            "associated_symptoms": [],
            "medical_specialty": "General Medicine",
            "triage_level": "Critical",
            "clinical_summary": "x",
            "red_flags": [],
        }
    )
    with pytest.raises(KnowledgeExtractionError):
        validate_knowledge_extraction_json(payload)


def test_heuristic_extractor_from_article():
    article = PubMedArticle(
        pmid="999",
        title="Acute myocardial infarction review",
        abstract="Patients present with chest pain and dyspnea. Emergency care is required.",
        query_disease="Myocardial Infarction",
    )
    extractor = MedicalKnowledgeExtractor(use_llm=False)
    knowledge = extractor.extract(article)

    assert knowledge.disease_name == "Myocardial Infarction"
    assert "chest pain" in knowledge.associated_symptoms
    assert knowledge.medical_specialty == "Cardiology"
    assert knowledge.triage_level in {"Emergency", "Urgent"}
    assert knowledge.source_pmid == "999"
    assert knowledge.extraction_mode == "heuristic"


def test_default_diseases_includes_ddxplus_catalog():
    ddxplus = _load_ddxplus_disease_names()
    if ddxplus:
        assert len(harvest_config.default_diseases) >= len(ddxplus)
        assert "Pneumonia" in harvest_config.default_diseases
        assert "Hypertension" in harvest_config.default_diseases
    else:
        assert len(harvest_config.default_diseases) >= len(_FALLBACK_DISEASES)


def test_knowledge_storage_writes_files(tmp_path: Path):
    cfg = HarvestConfig(
        knowledge_base_dir=tmp_path,
        chroma_persist_dir=tmp_path / "chroma",
        extracted_json_dir=tmp_path / "extracted",
        markdown_dir=tmp_path / "markdown",
        raw_pubmed_dir=tmp_path / "raw_pubmed",
    )
    article = PubMedArticle(
        pmid="111",
        title="Hypertension overview",
        abstract="Elevated blood pressure is common.",
        journal="Lancet",
        year="2019",
        query_disease="Hypertension",
    )
    knowledge = ExtractedMedicalKnowledge(
        disease_name="Hypertension",
        associated_symptoms=["elevated blood pressure"],
        medical_specialty="Cardiology",
        triage_level="Routine",
        clinical_summary="Overview of hypertension.",
        red_flags=[],
        source_pmid="111",
        extraction_mode="test",
    )

    storage = KnowledgeStorage(config=cfg)
    doc = storage.save_one(article, knowledge)

    assert doc.json_path.exists()
    assert doc.markdown_path.exists()
    saved = json.loads(doc.json_path.read_text(encoding="utf-8"))
    assert saved["extracted"]["disease_name"] == "Hypertension"
    assert (tmp_path / "manifest.json").exists()
