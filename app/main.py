"""
Healix - Symptom Extraction Microservice
خدمة استخراج الأعراض العربية باستخدام نموذج MARBERT المُدرّب (NER).

يُحمّل النموذج مرّة واحدة عند بدء التشغيل ويُخزَّن في ``app.state`` ليُحقَن
في الطبقات الأخرى عبر الاعتمادية (Dependency Injection).
"""

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
from app.exceptions import HealixError, ModelLoadError, ModelNotLoadedError
from app.infrastructure.session_store import InMemorySessionStore
from app.llm.factory import build_llm_provider
from app.prompts.interview_builder import InterviewPromptBuilder
from app.routes import analyze, health, interview, speech
from app.services.conversation_service import ConversationService
from app.services.marbert_service import ModelLoader
from app.services.speech_to_text import SpeechToTextService
from app.services.symptom_extractor import SymptomExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """تحميل النموذج وتهيئة الخدمات عند بدء التشغيل، والتنظيف عند الإغلاق."""
    logger.info(" بدء تشغيل خدمة Healix...")
    loader = ModelLoader()
    loader.load()

    app.state.model_loader = loader
    extractor = SymptomExtractor(
        model_loader=loader,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        max_length=config.MAX_SEQUENCE_LENGTH,
    )
    app.state.symptom_extractor = extractor

    # محرك المحادثة (يعيد استخدام مستخرج الأعراض + مزوّد LLM + مخزن جلسات).
    app.state.conversation_service = ConversationService(
        extractor=extractor,
        provider=build_llm_provider(),
        prompt_builder=InterviewPromptBuilder(),
        store=InMemorySessionStore(),
        max_questions=config.MAX_QUESTIONS,
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
    app.state.symptom_extractor = None
    app.state.speech_service = None
    app.state.model_loader = None


app = FastAPI(
    title="Healix Medical Assistant API",
    description=(
        "خدمة المساعد الطبي الموحّدة: تفريغ الصوت (Whisper) + استخراج الأعراض "
        "(MARBERT) + محرك المقابلة + مزوّد الـ LLM — في تطبيق FastAPI واحد."
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

app.include_router(analyze.router, prefix="/api")
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
