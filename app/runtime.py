import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from .config import Settings
from .errors import ServiceError


logger = logging.getLogger("uvicorn.error")


@dataclass(slots=True)
class TranscriptResult:
    transcript: str
    language: str


@dataclass(slots=True)
class LangIdDecision:
    language: str
    is_decisive: bool
    predicted_label: str
    probability: float
    margin: float


class TranscriptionRuntime(Protocol):
    ready: bool
    model_name: str
    startup_error: str | None

    async def startup(self) -> None: ...

    async def transcribe(
        self, audio_path: Path, language_hint: str
    ) -> TranscriptResult: ...


class NeMoParakeetRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._langid_model: Any | None = None
        self._semaphore = asyncio.Semaphore(1)
        self.model_name = settings.PARAKEET_MODEL_NAME
        self.ready = False
        self.startup_error: str | None = None

    async def startup(self) -> None:
        try:
            await asyncio.to_thread(self._load_model)
            self.ready = True
            self.startup_error = None
            logger.info(
                "transcription_runtime_ready asr_model=%s langid_model=%s device=%s",
                self.model_name,
                self._settings.LANGID_MODEL_NAME,
                self._settings.MODEL_DEVICE,
            )
        except Exception as exc:
            self.ready = False
            self.startup_error = str(exc)
            logger.exception("transcription_runtime_startup_failed")

    async def transcribe(
        self, audio_path: Path, language_hint: str
    ) -> TranscriptResult:
        if not self.ready or self._model is None or self._langid_model is None:
            raise ServiceError(
                "SERVICE_UNAVAILABLE", "Transcription model is not ready.", 503
            )

        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._transcribe_sync, audio_path, language_hint),
                    timeout=self._settings.TRANSCRIPTION_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise ServiceError(
                    "SERVICE_UNAVAILABLE",
                    "Transcription service timed out while processing the audio.",
                    503,
                ) from exc

    def _load_model(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError(
                "NeMo ASR dependencies are not installed. Run `uv sync --extra inference`."
            ) from exc

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        langid_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self._settings.LANGID_MODEL_NAME
        )

        self._model = _maybe_move_model_to_device(model, self._settings.MODEL_DEVICE)
        self._langid_model = _maybe_move_model_to_device(
            langid_model, self._settings.MODEL_DEVICE
        )

    def _transcribe_sync(
        self, audio_path: Path, language_hint: str
    ) -> TranscriptResult:
        if self._model is None or self._langid_model is None:
            raise ServiceError(
                "SERVICE_UNAVAILABLE", "Transcription model is not ready.", 503
            )

        asr_started_at = perf_counter()
        hypotheses = self._model.transcribe([str(audio_path)], return_hypotheses=True)
        asr_duration_ms = (perf_counter() - asr_started_at) * 1000

        hypothesis = hypotheses[0] if hypotheses else None
        transcript = _extract_transcript(hypothesis)

        langid_started_at = perf_counter()
        langid_decision = _detect_language_with_langid(
            langid_model=self._langid_model,
            audio_path=audio_path,
            confidence_threshold=self._settings.LANGID_CONFIDENCE_THRESHOLD,
            min_margin=self._settings.LANGID_MIN_MARGIN,
        )
        langid_duration_ms = (perf_counter() - langid_started_at) * 1000

        language = langid_decision.language
        language_source = "langid"
        if not langid_decision.is_decisive and language == "unknown":
            language = _extract_detected_language(hypothesis)
            language_source = "asr_metadata" if language != "unknown" else "unknown"

        if not transcript:
            raise ServiceError(
                "TRANSCRIPTION_FAILED",
                "No usable transcript was produced from the audio.",
                422,
            )

        logger.info(
            "transcription_completed language_hint=%s final_language=%s language_source=%s langid_label=%s langid_probability=%.4f langid_margin=%.4f langid_decisive=%s asr_duration_ms=%.2f langid_duration_ms=%.2f transcript_chars=%s",
            language_hint,
            language,
            language_source,
            langid_decision.predicted_label,
            langid_decision.probability,
            langid_decision.margin,
            langid_decision.is_decisive,
            asr_duration_ms,
            langid_duration_ms,
            len(transcript),
        )

        return TranscriptResult(transcript=transcript, language=language)


def _extract_transcript(hypothesis: Any) -> str:
    if hypothesis is None:
        return ""

    if isinstance(hypothesis, str):
        return hypothesis.strip()

    text = getattr(hypothesis, "text", None)
    if isinstance(text, str):
        return text.strip()

    if isinstance(hypothesis, dict):
        value = hypothesis.get("text")
        if isinstance(value, str):
            return value.strip()

    return ""


def _extract_detected_language(hypothesis: Any) -> str:
    if hypothesis is None:
        return "unknown"

    for field_name in ("language", "language_code", "lang"):
        value = getattr(hypothesis, field_name, None)
        normalized = _normalize_language_code(value)
        if normalized != "unknown":
            return normalized

    if isinstance(hypothesis, dict):
        for field_name in ("language", "language_code", "lang"):
            normalized = _normalize_language_code(hypothesis.get(field_name))
            if normalized != "unknown":
                return normalized

    multi_value = getattr(hypothesis, "langs", None)
    normalized = _normalize_language_code(
        multi_value[0] if isinstance(multi_value, list) and multi_value else None
    )
    if normalized != "unknown":
        return normalized

    return "unknown"


def _detect_language_with_langid(
    langid_model: Any,
    audio_path: Path,
    confidence_threshold: float,
    min_margin: float,
) -> LangIdDecision:
    predicted_label, probability, margin = _infer_langid_prediction(
        langid_model, audio_path
    )

    if probability < confidence_threshold or margin < min_margin:
        return LangIdDecision(
            language="unknown",
            is_decisive=False,
            predicted_label=predicted_label,
            probability=probability,
            margin=margin,
        )

    return LangIdDecision(
        language=_normalize_language_code(predicted_label),
        is_decisive=True,
        predicted_label=predicted_label,
        probability=probability,
        margin=margin,
    )


def _infer_langid_prediction(
    langid_model: Any, audio_path: Path
) -> tuple[str, float, float]:
    try:
        import torch
    except ImportError as exc:
        raise ServiceError(
            "SERVICE_UNAVAILABLE",
            "Torch is required for language identification inference.",
            503,
        ) from exc

    try:
        _, logits = langid_model.infer_file(str(audio_path))
    except Exception as exc:
        raise ServiceError(
            "SERVICE_UNAVAILABLE",
            "Language identification failed while processing the audio.",
            503,
        ) from exc

    probabilities = torch.softmax(logits, dim=1)
    top_probabilities, top_indices = torch.topk(probabilities, k=2, dim=1)
    labels = _get_langid_labels(langid_model)

    top_index = int(top_indices[0][0].item())
    top_probability = float(top_probabilities[0][0].item())
    runner_up_probability = (
        float(top_probabilities[0][1].item()) if top_probabilities.shape[1] > 1 else 0.0
    )

    predicted_label = labels[top_index] if top_index < len(labels) else "unknown"
    return predicted_label, top_probability, top_probability - runner_up_probability


def _get_langid_labels(langid_model: Any) -> list[str]:
    labels = getattr(langid_model, "labels", None)
    if isinstance(labels, list) and labels:
        return labels

    train_ds = getattr(getattr(langid_model, "_cfg", None), "train_ds", None)
    cfg_labels = getattr(train_ds, "labels", None)
    if cfg_labels is not None:
        return list(cfg_labels)

    raise ServiceError(
        "SERVICE_UNAVAILABLE",
        "Language identification model labels are unavailable.",
        503,
    )


def _maybe_move_model_to_device(model: Any, device: str) -> Any:
    if device == "cpu":
        return model

    try:
        import torch

        if torch.cuda.is_available() and hasattr(model, "to"):
            return model.to(device)
    except Exception:
        return model

    return model


def _normalize_language_code(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"

    normalized = value.strip().lower()
    if normalized.startswith("es"):
        return "es"
    if normalized.startswith("en"):
        return "en"
    return "unknown"
