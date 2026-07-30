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
from app.domain.feature_encoder import FeatureEncoder
from app.domain.feature_extraction_rules import RuleBasedFeatureExtractor
from app.domain.feature_validator import FeatureValidator
from app.domain.ml_disease_predictor import MLDiseasePredictor
from app.domain.rule_based_confidence_estimator import RuleBasedConfidenceEstimator
from app.domain.rule_based_predictor import RuleBasedDiseasePredictor
from app.domain.rule_based_specialty_recommender import RuleBasedSpecialtyRecommender
from app.domain.rule_based_urgency_classifier import RuleBasedUrgencyClassifier
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
from app.prompts.assessment_explainer_builder import (
    EXPLANATION_JSON_SCHEMA,
    AssessmentExplainerPromptBuilder,
)
from app.prompts.assessment_extraction_builder import (
    EXTRACTION_JSON_SCHEMA,
    AssessmentExtractionPromptBuilder,
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
        store=InMemorySessionStore(),
        max_questions=config.MAX_QUESTIONS,
    )

    # محرك التقييم (Phase 3.1: بناء الميزات فقط). مزوّد LLM مستقل عمداً عن
    # مزوّد محرك المحادثة — عزل حالة التدهور/التكيّف بين المحرّكين.
    # عقد JSON مخصَّص لمحرك التقييم (لا عقد المقابلة الافتراضي) — يحل مشكلة
    # فرض json_mode="schema" لشكل قرار المقابلة على استدعاءات استخراج
    # الميزات (انظر توثيق response_schema بـ openrouter_provider.py).
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

    # طبقة التحقّق من الميزات (Phase 3.2) — قواعد النطاق/التعداد محمَّلة من
    # app/dictionaries/feature_validation_rules.json مرّة واحدة هنا، لا قراءة
    # ملفات لكل طلب. DictionaryLoader يفشل الإقلاع بوضوح لو كانت مشوّهة.
    validation_rules = DictionaryLoader.load_feature_validation_rules()
    app.state.feature_validator = FeatureValidator(rules=validation_rules)

    # مشفِّر الميزات (Phase 3.3) — مخطّط الترميز محمَّل من
    # app/dictionaries/feature_schemas/v1.json مرّة واحدة هنا. لا مُتنبِّئ
    # يستهلكه بعد (مراحل قادمة) — يُبنى ويُحقَن فقط، جاهزاً حين يلزم.
    feature_schema = DictionaryLoader.load_feature_schema()
    app.state.feature_encoder = FeatureEncoder(schema=feature_schema)

    # مُتنبِّئ المرض (Phase 3.4) — Adapter قاعدي أول (Placeholder)، يحقّق
    # عقد DiseasePredictorPort الذي ستستخدمه adapters ML لاحقاً (XGBoost/
    # RandomForest/CatBoost) — استبدال هذا السطر فقط، بلا تغيير بقية النظام.
    #
    # مسار ML موازٍ (USE_ML_PREDICTOR): عند التفعيل، ModelLoader.load_all()
    # يحمّل النموذج المحفوظ في models/ (يتحقّق من تطابق الأبعاد فوراً، يفشل
    # الإقلاع بوضوح عند أي انحراف) ويُحقَن MLDiseasePredictor بدلاً من الـ
    # Adapter القاعدي — بلا حذف أو تعليق السطر القاعدي أدناه، فقط تفرّع.
    if config.USE_ML_PREDICTOR:
        ml_model, ml_feature_names, ml_label_encoder = ModelLoader.load_all()
        app.state.disease_predictor = MLDiseasePredictor(
            model=ml_model, feature_names=ml_feature_names, label_encoder=ml_label_encoder,
        )
    else:
        app.state.disease_predictor = RuleBasedDiseasePredictor()

    # مُصنِّف الاستعجال (Phase 3.5) — Adapter قاعدي أول (Placeholder)، يحقّق
    # عقد UrgencyClassifierPort الذي سيستخدمه نموذج ML لاحقاً — استبدال هذا
    # السطر فقط، بلا تغيير بقية النظام. يعمل على ClinicalFeatureSet مباشرة،
    # لا EncodedFeatures (خلافاً لمُتنبِّئ المرض).
    app.state.urgency_classifier = RuleBasedUrgencyClassifier()

    # مُوصي التخصّص الطبي (Phase 3.6) — Adapter قاعدي أول (Placeholder)، يحقّق
    # عقد SpecialtyRecommenderPort الذي سيستخدمه نموذج ML لاحقاً — استبدال هذا
    # السطر فقط، بلا تغيير بقية النظام. قاموس (مرض → تخصّص) محمَّل من
    # app/dictionaries/specialty_lookup.yaml مرّة واحدة هنا، لا قراءة ملفات
    # لكل طلب.
    specialty_lookup = DictionaryLoader.load_specialty_lookup()
    app.state.specialty_recommender = RuleBasedSpecialtyRecommender(
        specialty_lookup=specialty_lookup
    )

    # مُقدِّر الموثوقية (Phase 3.7) — Adapter قاعدي أول (Placeholder)، يحقّق
    # عقد ConfidenceEstimatorPort الذي سيستخدمه مُعايِر ثقة ML لاحقاً —
    # استبدال هذا السطر فقط، بلا تغيير بقية النظام. آخر مرحلة بخطّ التقييم:
    # يقرأ كل ما أُنتج ويُصدر حكماً على الثقة فقط (لا تنبؤ/استعجال/تخصّص).
    app.state.confidence_estimator = RuleBasedConfidenceEstimator()

    # مُفسِّر التقييم (Phase 3.8) — المرحلة الأخيرة بالخطّ: يشرح ما حُسِب سلفاً
    # بالعربية فقط (لا تشخيص، لا تعديل تنبؤ/استعجال/تخصّص/ثقة). مزوّد LLM
    # مستقل بعقد JSON خاص بالتفسير (schema/validator/نص تصحيح) — معزول عن
    # مزوّدَي المقابلة والاستخراج، بنفس آلية response_schema بالمزوّد. الخدمة
    # تضمن الإرجاع دائماً (تدهور حتمي لطيف عند فشل الـLLM) فلا تُسقط التقييم.
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
            logger.error("⚠️ تعذّر تحميل Whisper — سيتعطّل الصوت فقط: %s", exc)
            app.state.speech_service = None
    else:
        logger.info("Whisper معطّل عبر الإعدادات (ENABLE_WHISPER=false).")
        app.state.speech_service = None

    logger.info("✅ الخدمة جاهزة")
    yield
    logger.info(" إيقاف خدمة Healix...")
    app.state.conversation_service = None
    app.state.assessment_feature_builder = None
    app.state.feature_validator = None
    app.state.feature_encoder = None
    app.state.disease_predictor = None
    app.state.urgency_classifier = None
    app.state.specialty_recommender = None
    app.state.confidence_estimator = None
    app.state.assessment_explainer = None
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
