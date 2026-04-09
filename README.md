# Transcription Service

FastAPI service for the Pending App voice-capture transcription boundary.

## Stack

- FastAPI
- `uv` for dependency and virtualenv management
- NVIDIA NeMo + `nvidia/parakeet-tdt-0.6b-v3` for inference
- `ffmpeg` for audio normalization

## Setup

1. Install `uv` and `ffmpeg` on the host machine.
2. Create a local environment and install dependencies:

```bash
uv sync --extra inference --group dev
```

3. Copy the example env file and fill in the service token:

```bash
cp .env.example .env
```

4. Start the service:

```bash
uv run uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Environment

- `TRANSCRIPTION_SERVICE_TOKEN`: shared bearer token required by all requests
- `PARAKEET_MODEL_NAME`: defaults to `nvidia/parakeet-tdt-0.6b-v3`
- `LANGID_MODEL_NAME`: defaults to `langid_ambernet`
- `MODEL_DEVICE`: defaults to `cuda`
- `MAX_UPLOAD_BYTES`: defaults to `10485760` (10 MB)
- `MAX_AUDIO_SECONDS`: defaults to `120`
- `TRANSCRIPTION_TIMEOUT_SECONDS`: defaults to `30`
- `LANGID_CONFIDENCE_THRESHOLD`: defaults to `0.55`
- `LANGID_MIN_MARGIN`: defaults to `0.15`
- `FFMPEG_BINARY`: defaults to `ffmpeg`

## Endpoints

- `GET /health/live`
- `GET /health/ready`
- `POST /transcribe`

## Observability

- service startup logs the loaded ASR and LangID models
- each transcription logs LangID label, confidence, margin, final language source, and ASR/LangID timing
- each request logs normalized audio duration and total request latency

## Tests

```bash
uv run pytest
```

## Deployment

See `DEPLOYMENT.md` for the recommended remote GPU-host deployment, HTTPS exposure, and app-side configuration.
