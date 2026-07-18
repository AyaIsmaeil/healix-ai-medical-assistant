"""
Healix - Interview Schemas
نماذج الطلب والاستجابة لمحرك المحادثة (طبقة الـ API).
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SymptomOut(BaseModel):
    text: str
    negated: bool


class InterviewTurnRequest(BaseModel):
    """رسالة مريض في مقابلة أخذ التاريخ المرضي."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="رسالة المريض بالعربية",
        examples=["أعاني من صداع وحرارة"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="معرّف الجلسة (يُترك فارغاً في أول رسالة ويُعاد استخدامه بعدها)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"text": "أعاني من صداع وحرارة", "session_id": None}
        }
    }


class InterviewTurnResponse(BaseModel):
    """استجابة الدور: سؤال عربي واحد أو إشارة الانتهاء."""

    session_id: str
    finished: bool
    next_slot: Optional[str] = None
    question: Optional[str] = None
    turn: int
    status: str
    symptoms: List[SymptomOut] = Field(default_factory=list)
