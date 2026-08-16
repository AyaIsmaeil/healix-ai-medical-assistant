"""Request/response contract between Laravel and this service.

CLAUDE.md > Laravel <-> service contract is the authoritative documentation
of this interface — keep the two in sync on any change here. CLAUDE.md >
Architecture: Laravel is the main backend (auth, users, medical records,
doctors, bookings, saved reports); this service owns the graph,
conversation state, RAG retrieval, and LLM calls. They talk over internal
HTTP with a shared secret in the header — a transport-level concern for the
future FastAPI route/dependency, not a body field, so it is not modeled
here.

Models only. No FastAPI routes yet (CLAUDE.md > Working style: one piece at
a time) — these are the shapes every future route must use.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from state import PatientSex, Severity, Stage

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn in a conversation, as accumulated in HealixState["messages"].

    The formal version of the {"role": ..., "content": ...} shape
    nodes/crisis_check.py first used ad hoc, fixed here before more nodes
    depend on it (CLAUDE.md > Laravel <-> service contract).

    state.py deliberately keeps state["messages"] as list[dict[str, str]],
    not list[Message] — same reasoning as state.Symptom: a graph reducer
    must tolerate a malformed entry without raising mid-conversation (see
    state.merge_symptoms's docstring), which a strict Pydantic field would
    not. Message is still the one definition every *producer* of an entry —
    the future request-handling route, and any node that appends one —
    should build against; the runtime list just stays dict-typed.
    """

    role: Role
    content: str


class ChatRequest(BaseModel):
    """One incoming turn from Laravel.

    thread_id: generated and owned by Laravel (CLAUDE.md > Architecture) —
        this service never mints its own.
    message: the patient's new message this turn, raw text. Not a full
        history: this service owns conversation state (CLAUDE.md >
        Architecture) and accumulates it per thread_id across turns, so
        Laravel never resends prior turns.
    medical_record_summary: a *filtered* summary (CLAUDE.md > Architecture
        — never a raw record dump). Optional: only load_record's
        first-turn run uses it (CLAUDE.md > Graph flow), so Laravel does
        not need to keep resending it once a thread is underway.
    patient_sex: structured, Laravel's own account data — never inferred
        from conversation text (see state.PatientSex's own comment).
        Optional, same reasoning as medical_record_summary: Laravel may
        not always send it, and this service must not treat its absence
        as an error. Consumed by nodes.rag_retrieve to gate sex-specific
        rag/knowledge_base/ entries; see that node's module docstring for
        what happens when it's absent and a sex-specific candidate would
        otherwise qualify (a follow-up question, not a guess).

    No other metadata is currently required.
    """

    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    medical_record_summary: str | None = None
    patient_sex: PatientSex | None = None


class ChatResponse(BaseModel):
    """One outgoing turn to Laravel, once a graph run reaches END.

    thread_id: echoed back so a response is traceable on its own.
    reply: the patient-facing message for this turn — Syrian colloquial
        Arabic (CLAUDE.md > Language) — taken from the last assistant-role
        Message appended to state["messages"] by whichever terminal node
        ran.
    stage: which terminal node produced this response (CLAUDE.md > Graph
        flow: crisis_node, emergency_node, ask_followup, or
        generate_reports, mapped 1:1 onto state.Stage's four values —
        imported from there, not redefined here, so a node setting
        state["stage"] and this field can never quietly drift apart). An
        explicit field rather than something Laravel infers from which
        other fields happen to be populated — that inference would
        silently re-derive the graph's own routing logic on the Laravel
        side, and the two could drift.
    is_crisis: duplicates what stage == "crisis" already implies. Kept as
        its own field anyway because it is safety-critical (CLAUDE.md >
        Non-negotiable safety rules) — the same reasoning that has crisis
        detection itself combine two redundant layers rather than trust
        one (safety rule 3). Laravel can act on this directly without
        parsing `stage`.
    severity, red_flags: mirror HealixState as of this turn's end.
    diagnosis, specialty, reports: populated only when stage ==
        "diagnosis" (generate_reports is the only path that reaches it).
        Left as loosely-typed dicts, matching state.py's own typing for
        these fields — the diagnose/report-generation nodes don't exist
        yet (CLAUDE.md > Working style: one node at a time), so there is
        nothing concrete to model more strictly yet.
    """

    thread_id: str
    reply: str
    stage: Stage
    is_crisis: bool
    severity: Severity | None = None
    red_flags: list[str] = Field(default_factory=list)
    diagnosis: dict | None = None
    specialty: str | None = None
    reports: dict | None = None


class SpeechTranscribeResponse(BaseModel):
    """POST /speech/transcribe's response — speech_client.transcribe()'s
    output, wrapped for the HTTP boundary. Callers (Laravel, or the dev
    chat page) still send the transcribed text through POST /chat as an
    ordinary ChatRequest.message afterward — this endpoint only does
    speech-to-text, never triggers a graph turn itself."""

    text: str


class SpeechSynthesizeRequest(BaseModel):
    """POST /speech/synthesize's request body. Typically the `reply` from
    a prior ChatResponse, but not constrained to that — any Arabic text
    the caller wants read aloud."""

    text: str = Field(min_length=1)
