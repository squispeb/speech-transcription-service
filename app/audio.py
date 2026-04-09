import shutil
import subprocess
import wave
from pathlib import Path

from fastapi import UploadFile

from .errors import ServiceError

ALLOWED_AUDIO_MIME_TYPES = {
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/mpeg",
}

ALLOWED_AUDIO_SUFFIXES = {".webm", ".wav", ".mp4", ".mpeg", ".mp3", ".m4a"}
DEFAULT_UPLOAD_SUFFIX = ".bin"


def _run_ffmpeg_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise ServiceError(
            "SERVICE_UNAVAILABLE",
            "Audio normalization is unavailable on the server.",
            503,
        ) from exc


def verify_ffmpeg_available(ffmpeg_binary: str) -> None:
    result = _run_ffmpeg_command([ffmpeg_binary, "-version"])
    if result.returncode != 0:
        raise ServiceError(
            "SERVICE_UNAVAILABLE",
            "Audio normalization is unavailable on the server.",
            503,
        )


def ensure_supported_upload(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    suffix = Path(upload.filename or "audio").suffix.lower()

    if content_type in ALLOWED_AUDIO_MIME_TYPES:
        return suffix or DEFAULT_UPLOAD_SUFFIX

    if suffix in ALLOWED_AUDIO_SUFFIXES:
        return suffix

    raise ServiceError(
        "UNSUPPORTED_MEDIA_TYPE", "Uploaded audio format is not supported.", 415
    )


async def write_upload_to_path(
    upload: UploadFile, target_path: Path, max_upload_bytes: int
) -> int:
    total_bytes = 0

    with target_path.open("wb") as output_file:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise ServiceError(
                    "FILE_TOO_LARGE",
                    "Uploaded audio exceeds the configured size limit.",
                    413,
                )

            output_file.write(chunk)

    await upload.close()

    if total_bytes == 0:
        raise ServiceError("EMPTY_AUDIO", "Uploaded audio file is empty.", 400)

    return total_bytes


def normalize_audio_file(
    input_path: Path, output_path: Path, ffmpeg_binary: str
) -> None:
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]

    result = _run_ffmpeg_command(command)

    if result.returncode != 0:
        raise ServiceError(
            "INVALID_REQUEST",
            "Uploaded audio could not be decoded into a supported format.",
            400,
        )


def get_wav_duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
    except wave.Error as exc:
        raise ServiceError(
            "INVALID_REQUEST", "Normalized audio could not be inspected.", 400
        ) from exc

    if frame_rate <= 0:
        raise ServiceError(
            "INVALID_REQUEST", "Normalized audio has an invalid sample rate.", 400
        )

    return frames / frame_rate


def copy_audio_file(source_path: Path, target_path: Path) -> None:
    shutil.copyfile(source_path, target_path)
