from typing import Literal

from pydantic import BaseModel, ConfigDict


TranscriptionLanguageHint = Literal["auto", "es", "en"]
DetectedLanguage = Literal["es", "en", "unknown"]
FailureCode = Literal[
    "EMPTY_AUDIO",
    "UNSUPPORTED_MEDIA_TYPE",
    "FILE_TOO_LARGE",
    "UNAUTHORIZED",
    "TRANSCRIPTION_FAILED",
    "SERVICE_UNAVAILABLE",
    "INVALID_REQUEST",
]


class TranscribeAudioForm(BaseModel):
    languageHint: TranscriptionLanguageHint = "auto"
    source: Literal["pending-app"] | None = None


class TranscribeAudioSuccess(BaseModel):
    ok: Literal[True]
    transcript: str
    language: DetectedLanguage


class TranscribeAudioFailure(BaseModel):
    ok: Literal[False]
    code: FailureCode
    message: str


TranscribeAudioResponse = TranscribeAudioSuccess | TranscribeAudioFailure


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    status: Literal["live", "ready"]
    model: str | None = None
    detail: str | None = None
