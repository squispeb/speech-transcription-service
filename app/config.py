from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def _coalesce(*values: str | None) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def _require(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(slots=True)
class Settings:
    TRANSCRIPTION_SERVICE_TOKEN: str
    PARAKEET_MODEL_NAME: str = "nvidia/parakeet-tdt-0.6b-v3"
    LANGID_MODEL_NAME: str = "langid_ambernet"
    MODEL_DEVICE: str = "cuda"
    MODEL_IDLE_UNLOAD_SECONDS: float = 600.0
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_AUDIO_SECONDS: int = 120
    TRANSCRIPTION_TIMEOUT_SECONDS: int = 30
    LANGID_CONFIDENCE_THRESHOLD: float = 0.55
    LANGID_MIN_MARGIN: float = 0.15
    FFMPEG_BINARY: str = "ffmpeg"
    TEMP_DIRECTORY: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        dotenv_values = _load_dotenv(Path(".env"))
        env = os.environ

        temp_directory = _coalesce(env.get("TEMP_DIRECTORY"), dotenv_values.get("TEMP_DIRECTORY"))
        if temp_directory is not None and not temp_directory.strip():
            temp_directory = None

        return cls(
            TRANSCRIPTION_SERVICE_TOKEN=_require(
                _coalesce(
                    env.get("TRANSCRIPTION_SERVICE_TOKEN"),
                    dotenv_values.get("TRANSCRIPTION_SERVICE_TOKEN"),
                ),
                "TRANSCRIPTION_SERVICE_TOKEN",
            ),
            PARAKEET_MODEL_NAME=_coalesce(
                env.get("PARAKEET_MODEL_NAME"), dotenv_values.get("PARAKEET_MODEL_NAME")
            )
            or "nvidia/parakeet-tdt-0.6b-v3",
            LANGID_MODEL_NAME=_coalesce(
                env.get("LANGID_MODEL_NAME"), dotenv_values.get("LANGID_MODEL_NAME")
            )
            or "langid_ambernet",
            MODEL_DEVICE=_coalesce(env.get("MODEL_DEVICE"), dotenv_values.get("MODEL_DEVICE"))
            or "cuda",
            MODEL_IDLE_UNLOAD_SECONDS=_float(
                _coalesce(
                    env.get("MODEL_IDLE_UNLOAD_SECONDS"),
                    dotenv_values.get("MODEL_IDLE_UNLOAD_SECONDS"),
                ),
                600.0,
            ),
            MAX_UPLOAD_BYTES=_int(
                _coalesce(env.get("MAX_UPLOAD_BYTES"), dotenv_values.get("MAX_UPLOAD_BYTES")),
                10 * 1024 * 1024,
            ),
            MAX_AUDIO_SECONDS=_int(
                _coalesce(env.get("MAX_AUDIO_SECONDS"), dotenv_values.get("MAX_AUDIO_SECONDS")),
                120,
            ),
            TRANSCRIPTION_TIMEOUT_SECONDS=_int(
                _coalesce(
                    env.get("TRANSCRIPTION_TIMEOUT_SECONDS"),
                    dotenv_values.get("TRANSCRIPTION_TIMEOUT_SECONDS"),
                ),
                30,
            ),
            LANGID_CONFIDENCE_THRESHOLD=_float(
                _coalesce(
                    env.get("LANGID_CONFIDENCE_THRESHOLD"),
                    dotenv_values.get("LANGID_CONFIDENCE_THRESHOLD"),
                ),
                0.55,
            ),
            LANGID_MIN_MARGIN=_float(
                _coalesce(
                    env.get("LANGID_MIN_MARGIN"), dotenv_values.get("LANGID_MIN_MARGIN")
                ),
                0.15,
            ),
            FFMPEG_BINARY=_coalesce(env.get("FFMPEG_BINARY"), dotenv_values.get("FFMPEG_BINARY"))
            or "ffmpeg",
            TEMP_DIRECTORY=temp_directory,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
