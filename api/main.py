"""api/main.py — the real Laravel-facing HTTP boundary, plus a local dev
chat UI for manual testing served from the same process.

CLAUDE.md > Laravel <-> service contract: api/contracts.py's ChatRequest/
ChatResponse are reused here unchanged — this file only wires them to a
real route, it does not redefine or narrow either shape.

POST /chat is the real route: it is what Laravel calls, gated on the
shared-secret header CLAUDE.md > Architecture describes ("This service
is not internet-facing. Internal network + shared secret in header.").
GET / (the dev chat page, api/static/index.html) also calls this SAME
route rather than a separate unauthenticated one — deliberately: a
second, parallel "dev-only" implementation of /chat would drift from
what Laravel actually experiences, and testing the real, authenticated
route IS the point of a manual-testing tool, not a reason to bypass it.
The dev page still works: see _inject_dev_token() below for how it gets
the shared secret without hardcoding it into the static file.

GET /health is deliberately the one unauthenticated route — a basic
liveness check Laravel's own client can call to confirm this service is
up at all, before it matters whether the caller has the shared secret
right. (No LabClient.php was present in this repository to check its
exact expected shape against — Laravel is a separate codebase. This
returns the smallest, most conventional shape, {"status": "ok"}; flagged
here in case Laravel's actual client expects something more specific.)

POST /speech/transcribe and POST /speech/synthesize are speech I/O only —
neither invokes the graph. speech_client.py (this project's single entry
point for speech, kept separate the same way llm_client.py stays separate
from nodes/) does the actual work; these routes just apply the same
auth gate POST /chat uses and shape the HTTP boundary around it. A voice
turn is therefore always two calls from a caller's point of view:
POST /speech/transcribe to get text, then POST /chat with that text —
never a single combined "voice chat" endpoint.

Run it (CLAUDE.md > Running locally — read that section before changing
this command; the port is pinned deliberately, not a stylistic choice):

    uvicorn api.main:app --reload --port 8004

— or `run.bat` / `run.sh`, which run this exact command. Do not omit
--port 8004: Laravel's config/services.php hardcodes
http://127.0.0.1:8004 as services.healix.url's default, and a real
integration bug already shipped from this service coming up on
whatever port uvicorn's bare default happened to be instead — see
CLAUDE.md > Running locally for the full incident.

Then open http://127.0.0.1:8004/ for the dev chat page, or POST to
/chat directly (with the header) to exercise the real route. Needs the
same .env configuration as every other real-LLM entry point in this
project (HEALIX_LLM_PROVIDER_*/HEALIX_MODEL_*/API keys — see
.env.example) plus HEALIX_INTERNAL_TOKEN (new — see below).
"""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Must run before ANY import that could transitively load ml/'s XGBoost
# stack (numpy/scikit-learn/xgboost) or speech_client.py's Whisper stack
# (faster-whisper/ctranslate2) — both bundle their own private OpenMP
# runtime, and this process is the one place both can end up loaded
# together (nodes/ml_corroborate.py and the /speech/* routes below).
# Reproduced directly during this feature's integration: whichever one
# initializes ITS OpenMP runtime first in a given process works fine;
# whichever loads second crashes the whole process with "OMP: Error #15:
# Initializing libiomp5md.dll, but found libiomp5md.dll already
# initialized" — not a Python exception either lazy-load's own try/except
# can catch, since it aborts the process before either speech_client.py's
# SpeechError or ml/model_loader.py's MLModelError has a chance to run.
# Real risk, not hypothetical: neither library is loaded eagerly at
# import time (both lazy-load on first real use, deliberately, to avoid
# paying their startup cost for callers who never need them) — so which
# one loads "first" depends purely on which kind of request a given
# process happens to receive first, e.g. a voice message reaching
# POST /speech/transcribe before any turn has ever reached
# nodes/ml_corroborate.py in that process's lifetime. This env var is the
# documented, standard workaround (also suggested by the OMP error
# message itself); the residual risk it accepts (two OpenMP thread pools
# coexisting without coordination) does not touch either library's
# correctness here — xgboost's predict_proba() and Whisper's transcription
# never run inside the same thread at the same instant in this codebase's
# request flow, and nodes/ml_corroborate.py's own signal is gated on
# argmax agreement, not a numeric value sensitive to floating-point
# thread-scheduling variance.
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
    """One turn: invoke the real compiled graph, shape the result as ChatResponse.

    thread_id is passed on every call, not just a detected "first" one —
    harmless on later turns (last-value-wins on a field with no reducer,
    state.py), and means this endpoint doesn't need to track per-thread
    turn number itself; the checkpointer (keyed by thread_id, CLAUDE.md >
    State) is what actually remembers everything before this turn.

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
                # Every node reads this via state.get("medical_record_summary",
                # "") — that default only kicks in when the KEY is absent,
                # not when it's present with value None. ChatRequest's own
                # field defaults to None when the caller omits it
                # (api/contracts.py), so forwarding it unchanged would plant
                # an explicit None into state and break the first node that
                # calls rules.crisis.normalize() on it (rules/red_flags.py's
                # mentions_chronic_condition).
                "medical_record_summary": request.medical_record_summary or "",
                # No equivalent None-vs-absent hazard here: every reader
                # (nodes/rag_retrieve.py) uses state.get("patient_sex")
                # with no default, so an absent key and an explicit None
                # already produce the identical None either way — verified
                # against that node's own code before assuming so.
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

    # diagnosis/specialty/reports are gated on stage == "diagnosis" exactly,
    # matching CLAUDE.md > Laravel <-> service contract's documented
    # ChatResponse shape verbatim ("populated only when stage ==
    # 'diagnosis'"). None of the three fields has a reducer (state.py), so
    # once a thread has ever gone through a real diagnosis turn they persist
    # in the checkpointed state indefinitely — a LATER turn that doesn't
    # reach diagnose/route_specialty/generate_reports again (an emergency,
    # a crisis, or a followup) would otherwise silently carry that stale
    # differential forward in the response. Observed for real: a live
    # emergency-stage request returned the previous turn's Migraine
    # differential and specialty alongside the "go to the ER now" reply,
    # before this gate existed. See
    # test_emergency_stage_response_does_not_carry_a_stale_diagnosis_from_an_earlier_turn.
    is_diagnosis_stage = stage == "diagnosis"

    return ChatResponse(
        thread_id=thread_id,
        conversation_id=thread_id,
        reply=_latest_assistant_reply(result.get("messages", [])),
        stage=stage,
        is_crisis=stage == "crisis",
        severity=result.get("severity"),
        # ChatResponse.red_flags is list[str] (api/contracts.py) — ids
        # only. "reason" is doctor-facing (state.py's own field comment:
        # explaining why reads as a clinical explanation bordering on
        # diagnosis, safety rule 1) and belongs in reports["doctor"]
        # instead, not duplicated onto this summary field.
        red_flags=[flag["id"] for flag in red_flags],
        diagnosis=result.get("diagnosis") if is_diagnosis_stage else None,
        # specialty_laravel, not the bare "specialty" state field: the KB's
        # own specialty strings don't match Laravel's real specializations
        # table (nodes/route_specialty.py's own docstring, CLAUDE.md > Known
        # limitations) — specialty_laravel is the SPECIALTY_MAP-translated
        # value a future doctor-matching lookup would actually need.
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
    (CLAUDE.md-equivalent: docs/AHD_DATA_PROVENANCE.md). Shares no graph
    state with /chat; a failure here cannot affect it. See
    rag/health_education/service.py for the safety-gate + retrieval +
    LLM-summary pipeline this wraps.
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
    """Speech-to-text only — does not itself invoke the graph. The
    caller sends the returned text through POST /chat as an ordinary
    turn afterward (speech_client.py's module docstring: this module is
    the single entry point for speech I/O, kept separate from the graph
    the same way llm_client.py stays separate from nodes/).

    Plain `def`, not `async def`, same as POST /chat above —
    speech_client.transcribe() is a blocking call (faster-whisper has no
    async API), so FastAPI runs this route in its threadpool rather than
    blocking the event loop.
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
    """Unauthenticated liveness check. See module docstring — this
    service's own POST /health/GET /health expectations on the Laravel
    side (LabClient) could not be verified against source, since that
    file lives in the separate Laravel codebase, not here. This is the
    smallest conventional shape; adjust if Laravel's real client expects
    something else."""
    return {"status": "ok"}


def _inject_dev_token(html: str) -> str:
    """Splice the real shared-secret token into the dev chat page's own
    JS so its fetch("/chat", ...) calls succeed against the now-
    authenticated route, without ever hardcoding a secret into the
    static file itself (api/static/index.html has no real value in it —
    only the placeholder below, safe to commit).

    This is the one deliberate, minimal edit the dev page needed to keep
    working once /chat stopped being unauthenticated — everything else
    about it (UI, RTL handling, banner) is untouched.
    """
    token = os.getenv(_INTERNAL_TOKEN_ENV_VAR, "").strip()
    return html.replace("__HEALIX_INTERNAL_TOKEN__", token)


@app.get("/")
def index() -> HTMLResponse:
    html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(_inject_dev_token(html))
