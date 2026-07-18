"""
Healix - Speech Schemas
نماذج الطلب/الاستجابة لتفريغ الصوت (Whisper Speech-to-Text).
"""

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class SpeechToTextRequest(BaseModel):
    """طلب تفريغ صوت — يُرسَل مسار محلي أو رابط، أحدهما حصراً."""

    audio_path: Optional[str] = Field(
        default=None, description="مسار مطلق لملف صوتي على نفس الخادم"
    )
    audio_url: Optional[HttpUrl] = Field(
        default=None, description="رابط لملف صوتي يُنزّل ثم يُفرَّغ"
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "SpeechToTextRequest":
        if bool(self.audio_path) == bool(self.audio_url):
            raise ValueError("يجب تحديد audio_path أو audio_url (واحد فقط لا كلاهما).")
        return self


class SpeechToTextResponse(BaseModel):
    """استجابة التفريغ."""

    success: bool = True
    text: str
