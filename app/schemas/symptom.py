"""
Healix - Symptom Schemas
نماذج الطلب والاستجابة لنقطة استخراج الأعراض (طبقة الـ API).
"""

from typing import List

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    """طلب استخراج الأعراض من نص."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="النص الطبي المراد استخراج الأعراض منه",
        examples=["أعاني من صداع شديد وحرارة ولا يوجد كحة"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {"text": "أعاني من صداع شديد وحرارة ولا يوجد كحة"}
        }
    }


class SymptomOut(BaseModel):
    """عرض واحد مستخرج."""

    text: str = Field(..., description="نص العرض")
    negated: bool = Field(..., description="هل العرض منفي؟")
    confidence: float = Field(..., ge=0.0, le=1.0, description="نسبة الثقة")


class ExtractResponse(BaseModel):
    """استجابة استخراج الأعراض."""

    symptoms: List[SymptomOut] = Field(default_factory=list)
