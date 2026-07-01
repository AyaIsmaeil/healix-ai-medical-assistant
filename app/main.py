import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import DEBUG
from app.core.exceptions import SpeechToTextError
from app.routes.chat import router as chat_router
from app.routes.speech import router as speech_router
from app.services.prediction_service import get_prediction_service
from app.services.speech_to_text import SpeechToTextService

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Healix AI Medical Assistant...")
    app.state.speech_to_text_service = SpeechToTextService.load()
    # Load the MARBERT symptom-extraction model once at startup.
    app.state.prediction_service = get_prediction_service()
    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down Healix AI Medical Assistant...")


app = FastAPI(
    title="Healix AI Medical Assistant",
    description="نظام ذكاء اصطناعي طبي متكامل لتشخيص الأعراض وإدارة العيادات بالعامية السورية",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(speech_router, prefix="/api")

@app.exception_handler(SpeechToTextError)
async def speech_to_text_exception_handler(_: Request, exc: SpeechToTextError) -> JSONResponse:
    logger.error("Speech-to-text error: %s", exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.message},
    )


@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": "Healix AI Medical Assistant",
        "message": "Running",
    }


if __name__ == "__main__":
    import uvicorn


    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
