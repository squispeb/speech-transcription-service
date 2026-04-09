from contextlib import asynccontextmanager
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import audio as audio_utils
from .auth import require_bearer_token
from .config import Settings, get_settings
from .errors import ServiceError
from .runtime import NeMoParakeetRuntime, TranscriptionRuntime
from .schemas import (
    HealthResponse,
    TranscribeAudioFailure,
    TranscribeAudioForm,
    TranscribeAudioResponse,
    TranscribeAudioSuccess,
)


logger = logging.getLogger("uvicorn.error")


def create_app(
    settings: Settings | None = None, runtime: TranscriptionRuntime | None = None
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_runtime = runtime or NeMoParakeetRuntime(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.runtime = resolved_runtime
        await resolved_runtime.startup()
        yield

    app = FastAPI(
        title="Pending App Transcription Service", version="0.1.0", lifespan=lifespan
    )
    app.state.settings = resolved_settings
    app.state.runtime = resolved_runtime

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        payload = TranscribeAudioFailure(ok=False, code=exc.code, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    def get_runtime(request: Request) -> TranscriptionRuntime:
        return request.app.state.runtime

    @app.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        return HealthResponse(ok=True, status="live")

    @app.get("/health/ready", response_model=HealthResponse)
    async def health_ready(
        runtime: TranscriptionRuntime = Depends(get_runtime),
    ) -> JSONResponse | HealthResponse:
        ready = runtime.ready
        detail = runtime.startup_error

        if ready:
            try:
                audio_utils.verify_ffmpeg_available(resolved_settings.FFMPEG_BINARY)
            except ServiceError as exc:
                ready = False
                detail = exc.message

        payload = HealthResponse(
            ok=ready,
            status="ready",
            model=runtime.model_name,
            detail=detail,
        )

        if ready:
            return payload

        return JSONResponse(status_code=503, content=payload.model_dump())

    @app.post("/transcribe", response_model=TranscribeAudioResponse)
    async def transcribe_audio(
        _: None = Depends(require_bearer_token),
        audio: UploadFile = File(...),
        languageHint: str = Form("auto"),
        source: str | None = Form(None),
        runtime: TranscriptionRuntime = Depends(get_runtime),
    ) -> TranscribeAudioSuccess:
        request_started_at = perf_counter()

        try:
            form_data = TranscribeAudioForm(languageHint=languageHint, source=source)
        except ValidationError as exc:
            raise ServiceError("INVALID_REQUEST", exc.errors()[0]["msg"], 400) from exc

        upload_suffix = audio_utils.ensure_supported_upload(audio)

        with TemporaryDirectory(dir=resolved_settings.TEMP_DIRECTORY) as temp_dir:
            temp_directory = Path(temp_dir)
            uploaded_path = temp_directory / f"upload{upload_suffix}"
            normalized_path = temp_directory / "normalized.wav"

            await audio_utils.write_upload_to_path(
                upload=audio,
                target_path=uploaded_path,
                max_upload_bytes=resolved_settings.MAX_UPLOAD_BYTES,
            )
            audio_utils.normalize_audio_file(
                input_path=uploaded_path,
                output_path=normalized_path,
                ffmpeg_binary=resolved_settings.FFMPEG_BINARY,
            )

            duration_seconds = audio_utils.get_wav_duration_seconds(normalized_path)
            if duration_seconds > resolved_settings.MAX_AUDIO_SECONDS:
                raise ServiceError(
                    "FILE_TOO_LARGE",
                    "Uploaded audio exceeds the configured duration limit.",
                    413,
                )

            result = await runtime.transcribe(normalized_path, form_data.languageHint)

        request_duration_ms = (perf_counter() - request_started_at) * 1000
        logger.info(
            "transcribe_request_completed source=%s language_hint=%s audio_duration_s=%.2f final_language=%s request_duration_ms=%.2f",
            form_data.source or "unknown",
            form_data.languageHint,
            duration_seconds,
            result.language,
            request_duration_ms,
        )

        return TranscribeAudioSuccess(
            ok=True, transcript=result.transcript, language=result.language
        )

    return app
