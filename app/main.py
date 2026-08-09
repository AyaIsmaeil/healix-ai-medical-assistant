import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from app.config import config
from app.domain.clinical_priority import ClinicalPriorityEngine
from app.domain.feature_encoder import FeatureEncoder
from app.domain.feature_extraction_rules import RuleBasedFeatureExtractor
from app.domain.feature_validator import FeatureValidator
from app.domain.ml_disease_predictor import MLDiseasePredictor
from app.domain.red_flag_engine import RedFlagEngine
from app.domain.rule_based_confidence_estimator import RuleBasedConfidenceEstimator
from app.domain.rule_based_predictor import RuleBasedDiseasePredictor
from app.domain.rule_based_specialty_recommender import RuleBasedSpecialtyRecommender
from app.domain.rule_based_urgency_classifier import RuleBasedUrgencyClassifier
from app.domain.symptom_evidence_encoder import SymptomEvidenceEncoder
from app.exceptions import HealixError, ModelLoadError, ModelNotLoadedError
from app.infrastructure.dictionary_loader import DictionaryLoader
from app.infrastructure.model_loader import ModelLoader
from app.infrastructure.session_store import InMemorySessionStore
from app.llm.factory import build_llm_provider
from app.parsing.assessment_explainer_parser import (
    EXPLANATION_JSON_NUDGE,
    validate_explanation_shape,
)
from app.parsing.assessment_extraction_parser import (
    EXTRACTION_JSON_NUDGE,
    validate_extraction_shape,
)
from app.parsing.evidence_concept_extraction_parser import (
    EVIDENCE_CONCEPT_JSON_NUDGE,
    validate_evidence_concepts_shape,
)
from app.prompts.assessment_explainer_builder import (
    EXPLANATION_JSON_SCHEMA,
    AssessmentExplainerPromptBuilder,
)
from app.prompts.assessment_extraction_builder import (
    EXTRACTION_JSON_SCHEMA,
    AssessmentExtractionPromptBuilder,
)
from app.prompts.evidence_concept_extraction_builder import (
    EvidenceConceptExtractionPromptBuilder,
)
from app.parsing.interview_parser import (
    INTERVIEW_JSON_NUDGE,
    validate_interview_shape,
)
from app.prompts.interview_builder import (
    INTERVIEW_JSON_SCHEMA,
    InterviewPromptBuilder,
)
from app.routes import assessment, health, interview, speech
from app.services.assessment_explainer import AssessmentExplainer
from app.services.assessment_feature_builder import AssessmentFeatureBuilder
from app.services.conversation_service import ConversationService
from app.services.evidence_concept_extractor import EvidenceConceptExtractor
from app.services.llm_feature_extractor import LLMFeatureExtractor
from app.services.speech_to_text import SpeechToTextService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """تحميل النموذج وتهيئة الخدمات عند بدء التشغيل، والتنظيف عند الإغلاق."""
    logger.info(" بدء تشغيل خدمة Healix...")

    # محرّك الأعلام الحمراء — حتمي بالكامل، يُحمَّل من app/dictionaries/
    # مرّة واحدة هنا. DictionaryLoader يفشل الإقلاع بوضوح لو كان الكتالوج
    # مشوّهاً: خدمة طبية تعمل بكشف طوارئ معطوب أسوأ من خدمة لا تقلع.
    red_flag_engine = RedFlagEngine.from_dict(DictionaryLoader.load_red_flags())
    app.state.red_flag_engine = red_flag_engine
    logger.info("محرّك الأعلام الحمراء جاهز (إصدار %s).", red_flag_engine.version)

    # [P1] محرّك الأولوية السريرية — يقرّر الشكوى الرئيسية الحالية حتمياً
    # بعد كل رسالة. جدول مشوّه يعني ترتيب أسئلة خاطئاً، فيفشل الإقلاع بوضوح.
    priority_engine = ClinicalPriorityEngine.from_dict(
        DictionaryLoader.load_clinical_priority()
    )
    app.state.priority_engine = priority_engine
    logger.info("محرّك الأولوية السريرية جاهز (إصدار %s).", priority_engine.version)

    # وكيل المقابلة السريرية (LLM-first): استدعاء واحد لكل دور يُنتج الاستخراج
    # المنظَّم وقرار السؤال معاً. عقد JSON مخصَّص (schema/validator/نص تصحيح)
    # بنفس آلية بقيّة المستهلكين — لا مزوّد مشترك الحالة بين المحرّكات.
    app.state.conversation_service = ConversationService(
        provider=build_llm_provider(
            response_schema=INTERVIEW_JSON_SCHEMA,
            response_validator=validate_interview_shape,
            response_format_hint=INTERVIEW_JSON_NUDGE,
        ),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(
            ttl_seconds=config.SESSION_TTL_SECONDS,
            max_sessions=config.MAX_ACTIVE_SESSIONS,
        ),
        max_questions=config.MAX_QUESTIONS,
        red_flag_engine=red_flag_engine,
        priority_engine=priority_engine,
    )


    app.state.assessment_feature_builder = AssessmentFeatureBuilder(
        rule_extractor=RuleBasedFeatureExtractor(),
        llm_extractor=LLMFeatureExtractor(
            provider=build_llm_provider(
                response_schema=EXTRACTION_JSON_SCHEMA,
                response_validator=validate_extraction_shape,
                response_format_hint=EXTRACTION_JSON_NUDGE,
            ),
            prompt_builder=AssessmentExtractionPromptBuilder(),
        ),
    )

  
    validation_rules = DictionaryLoader.load_feature_validation_rules()
    app.state.feature_validator = FeatureValidator(rules=validation_rules)

 
    feature_schema = DictionaryLoader.load_feature_schema()
    app.state.feature_encoder = FeatureEncoder(schema=feature_schema)

    # يربط مفاهيم أدلة DDXPlus (يختارها الـLLM من قائمة مغلقة، لا نص حرّ)
    # برموز E_* قبل مُتنبِّئ ML — انظر توثيق الفجوة بـ symptom_evidence_encoder.py.
    # مستقلّ عمداً عن FeatureEncoder (لا يعدّل مخطّط v1.json "المجمَّد").
    symptom_evidence_map = DictionaryLoader.load_symptom_evidence_map()
    symptom_evidence_encoder = SymptomEvidenceEncoder.from_dict(symptom_evidence_map)
    app.state.symptom_evidence_encoder = symptom_evidence_encoder

    # اختيار مفاهيم الأدلة المنطبقة: استدعاء LLM مستقل بعقد JSON خاص (enum
    # مغلق مبني من قائمة المفاهيم المحقونة أعلاه — لا مطابقة نصّية بايثون).
    evidence_concept_prompt_builder = EvidenceConceptExtractionPromptBuilder(
        concepts=symptom_evidence_encoder.concepts
    )
    app.state.evidence_concept_extractor = EvidenceConceptExtractor(
        provider=build_llm_provider(
            response_schema=evidence_concept_prompt_builder.schema(),
            response_validator=validate_evidence_concepts_shape,
            response_format_hint=EVIDENCE_CONCEPT_JSON_NUDGE,
        ),
        prompt_builder=evidence_concept_prompt_builder,
    )

    if config.USE_ML_PREDICTOR:
        ml_model, ml_feature_names, ml_label_encoder = ModelLoader.load_all()
        app.state.disease_predictor = MLDiseasePredictor(
            model=ml_model, feature_names=ml_feature_names, label_encoder=ml_label_encoder,
        )
    else:
        app.state.disease_predictor = RuleBasedDiseasePredictor()

 
    app.state.urgency_classifier = RuleBasedUrgencyClassifier()

   
    specialty_lookup = DictionaryLoader.load_specialty_lookup()
    app.state.specialty_recommender = RuleBasedSpecialtyRecommender(
        specialty_lookup=specialty_lookup
    )

    app.state.confidence_estimator = RuleBasedConfidenceEstimator()


    app.state.assessment_explainer = AssessmentExplainer(
        provider=build_llm_provider(
            response_schema=EXPLANATION_JSON_SCHEMA,
            response_validator=validate_explanation_shape,
            response_format_hint=EXPLANATION_JSON_NUDGE,
        ),
        prompt_builder=AssessmentExplainerPromptBuilder(),
    )

    # تفريغ الصوت (Whisper) — يُحمَّل مرّة واحدة كجزء من الخدمة الموحّدة.
    if config.ENABLE_WHISPER:
        try:
            app.state.speech_service = SpeechToTextService.load()
        except Exception as exc:  # noqa: BLE001 - الخدمة تبقى تعمل للنصّ إن فشل Whisper
            logger.error(" تعذّر تحميل Whisper — سيتعطّل الصوت فقط: %s", exc)
            app.state.speech_service = None
    else:
        logger.info("Whisper معطّل عبر الإعدادات (ENABLE_WHISPER=false).")
        app.state.speech_service = None

    logger.info(" الخدمة جاهزة")
    yield
    logger.info(" إيقاف خدمة Healix...")
    app.state.conversation_service = None
    app.state.assessment_feature_builder = None
    app.state.feature_validator = None
    app.state.feature_encoder = None
    app.state.symptom_evidence_encoder = None
    app.state.evidence_concept_extractor = None
    app.state.disease_predictor = None
    app.state.urgency_classifier = None
    app.state.specialty_recommender = None
    app.state.confidence_estimator = None
    app.state.assessment_explainer = None
    app.state.red_flag_engine = None
    app.state.priority_engine = None
    app.state.speech_service = None


app = FastAPI(
    title="Healix Medical Assistant API",
    description=(
        "خدمة المساعد الطبي الموحّدة: تفريغ الصوت (Whisper) + وكيل المقابلة "
        "السريرية بالـLLM (استخراج منظَّم + أسئلة المتابعة) + محرك التقييم — "
        "في تطبيق FastAPI واحد."
    ),
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(interview.router, prefix="/api")
app.include_router(speech.router, prefix="/api")


@app.exception_handler(ModelNotLoadedError)
async def _model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ModelLoadError)
async def _model_load_handler(request: Request, exc: ModelLoadError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(HealixError)
async def _healix_error_handler(request: Request, exc: HealixError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=str(config.LOG_LEVEL).lower(),
    )
