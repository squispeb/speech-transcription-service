from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TRANSCRIPTION_SERVICE_TOKEN: str = Field(min_length=1)
    PARAKEET_MODEL_NAME: str = "nvidia/parakeet-tdt-0.6b-v3"
    LANGID_MODEL_NAME: str = "langid_ambernet"
    MODEL_DEVICE: str = "cuda"
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_AUDIO_SECONDS: int = 120
    TRANSCRIPTION_TIMEOUT_SECONDS: int = 30
    LANGID_CONFIDENCE_THRESHOLD: float = 0.55
    LANGID_MIN_MARGIN: float = 0.15
    FFMPEG_BINARY: str = "ffmpeg"
    TEMP_DIRECTORY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
