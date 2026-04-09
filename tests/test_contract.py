import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from app import audio as audio_utils
from app.config import Settings
from app.errors import ServiceError
from app.main import create_app
from app.runtime import TranscriptResult


@dataclass
class FakeRuntime:
    ready: bool = True
    model_name: str = "fake-parakeet"
    startup_error: str | None = None

    async def startup(self) -> None:
        return None

    async def transcribe(
        self, audio_path: Path, language_hint: str
    ) -> TranscriptResult:
        return TranscriptResult(transcript="Tengo que llamar a mamá.", language="es")


def make_test_client() -> TestClient:
    settings = Settings(TRANSCRIPTION_SERVICE_TOKEN="test-token", TEMP_DIRECTORY=None)
    app = create_app(settings=settings, runtime=FakeRuntime())
    return TestClient(app)


def make_wav_bytes(duration_seconds: float = 1.0) -> bytes:
    sample_rate = 16000
    frame_count = int(sample_rate * duration_seconds)
    samples = b"\x00\x00" * frame_count
    output_path = Path(__file__).parent / "sample.wav"

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)

    content = output_path.read_bytes()
    output_path.unlink()
    return content


def patch_normalization(monkeypatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "normalize_audio_file",
        lambda input_path, output_path, ffmpeg_binary: audio_utils.copy_audio_file(
            input_path, output_path
        ),
    )


def test_rejects_missing_bearer_token() -> None:
    client = make_test_client()

    response = client.post(
        "/transcribe",
        files={"audio": ("sample.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_rejects_invalid_source(monkeypatch) -> None:
    patch_normalization(monkeypatch)
    client = make_test_client()

    response = client.post(
        "/transcribe",
        headers={"Authorization": "Bearer test-token"},
        data={"source": "mobile-app"},
        files={"audio": ("sample.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_rejects_unsupported_media_type() -> None:
    client = make_test_client()

    response = client.post(
        "/transcribe",
        headers={"Authorization": "Bearer test-token"},
        files={"audio": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_rejects_empty_audio(monkeypatch) -> None:
    patch_normalization(monkeypatch)
    client = make_test_client()

    response = client.post(
        "/transcribe",
        headers={"Authorization": "Bearer test-token"},
        files={"audio": ("sample.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "EMPTY_AUDIO"


def test_health_ready_reports_ffmpeg_unavailable(monkeypatch) -> None:
    client = make_test_client()
    monkeypatch.setattr(
        audio_utils,
        "verify_ffmpeg_available",
        lambda ffmpeg_binary: (_ for _ in ()).throw(
            ServiceError(
                "SERVICE_UNAVAILABLE",
                "Audio normalization is unavailable on the server.",
                503,
            )
        ),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "ok": False,
        "status": "ready",
        "model": "fake-parakeet",
        "detail": "Audio normalization is unavailable on the server.",
    }


def test_transcribes_audio(monkeypatch) -> None:
    patch_normalization(monkeypatch)
    client = make_test_client()

    response = client.post(
        "/transcribe",
        headers={"Authorization": "Bearer test-token"},
        data={"languageHint": "es", "source": "pending-app"},
        files={"audio": ("sample.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "transcript": "Tengo que llamar a mamá.",
        "language": "es",
    }


def test_normalize_audio_file_reports_invalid_ffmpeg_binary(tmp_path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"test")

    try:
        audio_utils.normalize_audio_file(
            input_path=input_path,
            output_path=output_path,
            ffmpeg_binary="/definitely/missing/ffmpeg",
        )
    except ServiceError as exc:
        assert exc.code == "SERVICE_UNAVAILABLE"
        assert exc.message == "Audio normalization is unavailable on the server."
        assert exc.status_code == 503
    else:
        raise AssertionError("expected ServiceError for unavailable ffmpeg")
