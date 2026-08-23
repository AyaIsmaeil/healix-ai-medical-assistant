"""Request/response contract between Laravel and this service.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from state import PatientSex, Severity, Stage

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn in a conversation, as accumulated in HealixState["messages"].
    """

    role: Role
    content: str


class ChatRequest(BaseModel):
    thread_id: str | None = Field(default=None, min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1)
    message: str = Field(min_length=1)
    medical_record_summary: str | None = None
    patient_sex: PatientSex | None = None

    @model_validator(mode="after")
    def _same_id_for_conversation_and_thread(self) -> Self:
        thread_id = self.thread_id
        conversation_id = self.conversation_id
        if thread_id is None and conversation_id is None:
            raise ValueError("thread_id or conversation_id is required")
        if (
            thread_id is not None
            and conversation_id is not None
            and thread_id != conversation_id
        ):
            raise ValueError(
                "thread_id and conversation_id must identify the same conversation"
            )
        resolved = conversation_id or thread_id
        assert resolved is not None
        self.thread_id = resolved
        self.conversation_id = resolved
        return self


class ChatResponse(BaseModel):
    """The response to a POST /chat request.""" 

    thread_id: str
    conversation_id: str | None = None
    reply: str
    stage: Stage    # one of "followup", "diagnosis", "crisis"
    is_crisis: bool
    severity: Severity | None = None
    red_flags: list[str] = Field(default_factory=list)
    diagnosis: dict | None = None
    specialty: str | None = None
    reports: dict | None = None

    """The conversation_id is optional in the response, but if provided, it must match the thread_id. This is to ensure that both identifiers refer to the same conversation."""
    @model_validator(mode="after") #
    def _echo_conversation_id(self) -> Self:
        if self.conversation_id is None:
            self.conversation_id = self.thread_id
        elif self.conversation_id != self.thread_id:
            raise ValueError(
                "thread_id and conversation_id must identify the same conversation"
            )
        return self


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
