from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from langgraph.graph.state import CompiledStateGraph

from api.contracts import (
    ChatRequest,
    ChatResponse,
    SpeechSynthesizeRequest,
    SpeechTranscribeResponse,
)
from api.health_qa_contracts import HealthQuestionRequest, HealthQuestionResponse
from graph import build_checkpointer, build_graph
from llm_client import LLMError
from rag.health_education.service import answer_health_question
from speech_client import SpeechError, synthesize, transcribe

load_dotenv()

_logger = logging.getLogger("healix.api")

_STATIC_DIR = Path(__file__).parent / "static"
_INTERNAL_TOKEN_ENV_VAR = "HEALIX_INTERNAL_TOKEN"
_INTERNAL_TOKEN_HEADER = "X-Healix-Internal-Token"

# Never returned to a caller: what actually failed goes to _logger only
# (CLAUDE.md > Audit logs and patient data: even an error path must not
# leak conversation-adjacent detail — a provider exception's message can
# quote prompt content). Laravel is "semi-trusted" per this task's own
# framing, not "trusted with our internals."
_UPSTREAM_ERROR_DETAIL = "upstream service unavailable — try again shortly"
_INTERNAL_ERROR_DETAIL = "internal server error"
_SPEECH_UNAVAILABLE_DETAIL = "speech service unavailable — try again shortly"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Build the checkpointer/graph once per process, and refuse to start
    at all if the shared secret isn't configured.

    Failing loudly here — not silently serving /chat unauthenticated —
    is the same "no silent fallback" discipline graph.build_checkpointer()
    and llm_client's provider/model resolution already apply elsewhere in
    this project: a required setting that's missing is a startup error,
    not a runtime default.
    """
    token = os.getenv(_INTERNAL_TOKEN_ENV_VAR, "").strip()
    if not token:
        raise RuntimeError(
            f"{_INTERNAL_TOKEN_ENV_VAR} is not set. This service is not "
            "internet-facing (CLAUDE.md > Architecture) and POST /chat is "
            "gated on this shared secret — refusing to start rather than "
            "silently serving it unauthenticated. Set it in .env; see "
            ".env.example."
        )
    app.state.internal_token = token

    checkpointer = build_checkpointer()
    app.state.checkpointer = checkpointer
    app.state.graph = build_graph(checkpointer)
    try:
        yield
    finally:
        checkpointer.conn.close()


app = FastAPI(
    title="Healix AI",
    lifespan=_lifespan,
)


def get_graph(request: Request) -> CompiledStateGraph:
    """The compiled graph, built once at startup (see _lifespan).

    A real FastAPI dependency, not a bare app.state read inside the
    handler, specifically so tests can override it
    (app.dependency_overrides[get_graph] = ...) with a fake graph
    instead of exercising the real checkpointer/LLM stack for every
    unit test.
    """
    return request.app.state.graph


def require_internal_token(
    x_healix_internal_token: str | None = Header(default=None, alias=_INTERNAL_TOKEN_HEADER),
) -> None:
    """Reject the request before any graph work happens if the shared
    secret is missing or wrong (CLAUDE.md > Architecture).

    hmac.compare_digest, not `==` — a constant-time comparison for a
    security-relevant string check, cheap to get right and exactly the
    kind of thing not worth risking a timing side-channel over.
    """
    expected = os.getenv(_INTERNAL_TOKEN_ENV_VAR, "").strip()
    if not expected or not x_healix_internal_token:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not hmac.compare_digest(x_healix_internal_token, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def _latest_assistant_reply(messages: list[dict[str, str]]) -> str:
    """The reply this turn produced — api.contracts.ChatResponse.reply's
    own documented source: "the last assistant-role entry appended to
    state['messages']". Mirrors nodes._shared.latest_user_message's same
    reversed-scan shape, assistant instead of user.
    """
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    # Not expected in practice (every terminal node appends exactly one),
    # but must not leak that reasoning to the caller — same generic-detail
    # discipline as the graph.invoke() error handling below.
    _logger.error("Graph turn ended with no assistant message in state['messages'].")
    raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_DETAIL)


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_internal_token)])
def chat(request: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)) -> ChatResponse:
    """
    ChatRequest normalizes conversation_id and thread_id to the same
    string (one conversation = one LangGraph thread). This service does
    not mint that id and does not treat reset_stage as a new conversation.
    """
    thread_id = request.thread_id
    assert thread_id is not None  # ChatRequest validator always fills this
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": request.message}],
                "medical_record_summary": request.medical_record_summary or "",
                "patient_sex": request.patient_sex,
            },
            config=config,
        )
    except LLMError:
        _logger.exception("graph.invoke failed (LLMError) for thread_id=%s", thread_id)
        raise HTTPException(status_code=502, detail=_UPSTREAM_ERROR_DETAIL) from None
    except Exception:
        _logger.exception("graph.invoke failed unexpectedly for thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_DETAIL) from None

    stage = result["stage"]
    red_flags: list[dict[str, Any]] = result.get("red_flags") or []
    is_diagnosis_stage = stage == "diagnosis"

    return ChatResponse(
        thread_id=thread_id,
        conversation_id=thread_id,
        reply=_latest_assistant_reply(result.get("messages", [])),
        stage=stage,
        is_crisis=stage == "crisis",
        severity=result.get("severity"),
        red_flags=[flag["id"] for flag in red_flags],
        diagnosis=result.get("diagnosis") if is_diagnosis_stage else None,
        specialty=result.get("specialty_laravel") if is_diagnosis_stage else None,
        reports=result.get("reports") if is_diagnosis_stage else None,
    )


@app.post(
    "/health-questions",
    response_model=HealthQuestionResponse,
    dependencies=[Depends(require_internal_token)],
)
def health_questions(request: HealthQuestionRequest) -> HealthQuestionResponse:
    """General health-education Q&A — a separate feature from POST /chat
    """
    try:
        return answer_health_question(request.question, thread_id=request.thread_id)
    except LLMError:
        _logger.exception(
            "answer_health_question failed (LLMError) for thread_id=%s", request.thread_id
        )
        raise HTTPException(status_code=502, detail=_UPSTREAM_ERROR_DETAIL) from None
    except Exception:
        _logger.exception(
            "answer_health_question failed unexpectedly for thread_id=%s", request.thread_id
        )
        raise HTTPException(status_code=500, detail=_INTERNAL_ERROR_DETAIL) from None


@app.post(
    "/speech/transcribe",
    response_model=SpeechTranscribeResponse,
    dependencies=[Depends(require_internal_token)],
)
def speech_transcribe(file: UploadFile = File(...)) -> SpeechTranscribeResponse:
    """Speech-to-text only — does not itself invoke the graph.
    """
    audio_bytes = file.file.read()
    try:
        text = transcribe(audio_bytes)
    except SpeechError:
        _logger.exception("speech_client.transcribe failed")
        raise HTTPException(status_code=503, detail=_SPEECH_UNAVAILABLE_DETAIL) from None
    return SpeechTranscribeResponse(text=text)


@app.post("/speech/synthesize", dependencies=[Depends(require_internal_token)])
async def speech_synthesize(payload: SpeechSynthesizeRequest) -> Response:
    """Text-to-speech only — typically called with a prior ChatResponse.reply.

    `async def`, unlike the transcribe route above: speech_client.synthesize()
    is itself a native async function (edge-tts streams over a websocket),
    so awaiting it directly here is correct — wrapping it in a threadpool
    would gain nothing and cost a thread.
    """
    try:
        audio_bytes = await synthesize(payload.text)
    except SpeechError:
        _logger.exception("speech_client.synthesize failed")
        raise HTTPException(status_code=503, detail=_SPEECH_UNAVAILABLE_DETAIL) from None
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _inject_dev_token(html: str) -> str:
    token = os.getenv(_INTERNAL_TOKEN_ENV_VAR, "").strip()
    return html.replace("__HEALIX_INTERNAL_TOKEN__", token)


@app.get("/")
def index() -> HTMLResponse:
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(_inject_dev_token(html))
