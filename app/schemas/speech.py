from pydantic import BaseModel, Field, HttpUrl, model_validator


class SpeechToTextRequest(BaseModel):
    audio_path: str | None = Field(
        default=None,
        description="Absolute path to an audio file on the shared filesystem.",
    )
    audio_url: HttpUrl | None = Field(
        default=None,
        description="Public or signed URL to download the audio file.",
    )

    @model_validator(mode="after")
    def validate_audio_source(self) -> "SpeechToTextRequest":
        has_path = bool(self.audio_path and self.audio_path.strip())
        has_url = self.audio_url is not None

        if not has_path and not has_url:
            raise ValueError("Either audio_path or audio_url must be provided.")
        if has_path and has_url:
            raise ValueError("Provide either audio_path or audio_url, not both.")
        return self


class SpeechToTextResponse(BaseModel):
    success: bool = True
    text: str
