"""WhisperLiveKit streaming STT backend (AlignAtt / SimulStreaming on MLX)."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from math import gcd
from typing import Any, cast

import numpy as np
from scipy.signal import resample_poly

from wyoming_mlx.backends.base import STTUpdate

try:
    from whisperlivekit import (  # pyright: ignore[reportMissingImports]
        AudioProcessor as _AudioProcessor,
    )
    from whisperlivekit import (  # pyright: ignore[reportMissingImports]
        TranscriptionEngine as _TranscriptionEngine,
    )

    _WLK_AVAILABLE = True
except ImportError:
    _AudioProcessor = None  # type: ignore[assignment]
    _TranscriptionEngine = None  # type: ignore[assignment]
    _WLK_AVAILABLE = False

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def _release_mlx_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx  # pyright: ignore[reportMissingImports]
    except ImportError:
        return
    mx.clear_cache()


def _resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample float32 mono audio to Whisper's required 16 kHz.

    Whisper produces empty/garbled transcripts when fed audio at the wrong
    rate, since its mel spectrogram is computed assuming 16 kHz input.
    """
    if sample_rate == WHISPER_SAMPLE_RATE:
        return pcm
    g = gcd(sample_rate, WHISPER_SAMPLE_RATE)
    up = WHISPER_SAMPLE_RATE // g
    down = sample_rate // g
    return resample_poly(pcm, up, down).astype(np.float32)


def resample_pcm16(audio: bytes, sample_rate: int) -> bytes:
    """Resample int16 mono PCM bytes to 16 kHz int16 mono PCM bytes."""
    if sample_rate == WHISPER_SAMPLE_RATE:
        return audio
    pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    pcm16k = _resample_to_16k(pcm, sample_rate)
    return (np.clip(pcm16k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _joined_text(lines: list[Any]) -> str:
    """Join WLK confirmed segments into a single transcript string."""
    return " ".join(seg.text.strip() for seg in lines if seg.text and seg.text.strip())


def _delta(emitted: str, confirmed: str) -> str:
    """Newly confirmed suffix, or "" if `confirmed` is not an extension of `emitted`.

    WLK confirmed lines only grow, so the prefix property normally holds; if a
    text revision ever breaks it we emit nothing and let the final transcript
    (which is authoritative) carry the correction.
    """
    if confirmed.startswith(emitted):
        return confirmed[len(emitted) :]
    return ""


class _WLKSession:
    """One utterance: wraps a WhisperLiveKit AudioProcessor."""

    def __init__(self, engine: Any, on_close: Callable[[], None]) -> None:
        assert _AudioProcessor is not None
        self._processor: Any | None = _AudioProcessor(transcription_engine=engine)
        self._results: AsyncIterator[Any] | None = None
        self._emitted = ""
        self._on_close = on_close
        self._closed = False

    async def _ensure_started(self) -> AsyncIterator[Any]:
        if self._results is None:
            if self._processor is None:
                raise RuntimeError("STT session is closed")
            self._results = cast(AsyncIterator[Any], await self._processor.create_tasks())
        return self._results

    async def feed(self, audio: bytes, sample_rate: int) -> None:
        await self._ensure_started()
        assert self._processor is not None
        await self._processor.process_audio(resample_pcm16(audio, sample_rate))

    async def finish(self) -> None:
        await self._ensure_started()
        assert self._processor is not None
        # Empty message triggers WLK's flush + stop sequence; the results
        # generator terminates once all processing tasks finish.
        await self._processor.process_audio(None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._processor is not None:
                await self._processor.cleanup()
        finally:
            self._results = None
            self._processor = None
            self._on_close()

    async def updates(self) -> AsyncGenerator[STTUpdate, None]:
        results = await self._ensure_started()
        confirmed = ""
        buffer_tail = ""
        async for front in results:
            if front.status == "error":
                raise RuntimeError(f"WhisperLiveKit error: {front.error}")
            confirmed = _joined_text(front.lines)
            buffer_tail = front.buffer_transcription or ""
            new_text = _delta(self._emitted, confirmed)
            if new_text:
                self._emitted = confirmed
                yield STTUpdate(text=new_text)
        # Anything still unconfirmed when the stream ends belongs in the final.
        final = f"{confirmed} {buffer_tail.strip()}".strip() if buffer_tail.strip() else confirmed
        yield STTUpdate(final=final)


class WhisperLiveKitBackend:
    """STT backend powered by WhisperLiveKit LocalAgreement and MLX Whisper.

    The TranscriptionEngine is shared by active sessions and unloaded after an
    idle period. Each session gets its own AudioProcessor and may run concurrently.
    """

    def __init__(self, model: str = "large-v3-turbo", idle_unload_seconds: float = 60) -> None:
        if "/" in model:
            raise ValueError(
                f"model must be a WhisperLiveKit size name (e.g. 'large-v3-turbo'), "
                f"not a Hugging Face repo id: {model!r}"
            )
        if not _WLK_AVAILABLE or _TranscriptionEngine is None:
            raise ImportError("whisperlivekit is required but not installed")
        if idle_unload_seconds < 0:
            raise ValueError("idle_unload_seconds must be non-negative")
        self._model = model
        self._idle_unload_seconds = idle_unload_seconds
        self._engine: Any | None = None
        self._active_sessions = 0
        self._unload_handle: asyncio.TimerHandle | None = None

    def _load_engine(self) -> Any:
        if self._engine is None:
            log.info("Loading WhisperLiveKit engine (model=%s) …", self._model)
            assert _TranscriptionEngine is not None
            self._engine = _TranscriptionEngine(
                model_size=self._model,
                backend="mlx-whisper",
                backend_policy="localagreement",
                pcm_input=True,
            )
            log.info("[STT] WhisperLiveKit engine ready")
        return self._engine

    def _cancel_unload(self) -> None:
        if self._unload_handle is not None:
            self._unload_handle.cancel()
            self._unload_handle = None

    def _session_closed(self) -> None:
        self._active_sessions -= 1
        if self._active_sessions == 0:
            self._schedule_unload()

    def _schedule_unload(self) -> None:
        self._unload_handle = asyncio.get_running_loop().call_later(
            self._idle_unload_seconds, self._unload_engine
        )

    def _unload_engine(self) -> None:
        self._unload_handle = None
        if self._active_sessions or self._engine is None:
            return
        log.info("[STT] Unloading WhisperLiveKit engine after idle timeout")
        self._engine = None
        _release_mlx_memory()

    def start_session(self) -> _WLKSession:
        self._cancel_unload()
        engine = self._load_engine()
        try:
            session = _WLKSession(engine, self._session_closed)
        except Exception:
            if self._active_sessions == 0:
                self._schedule_unload()
            raise
        self._active_sessions += 1
        return session
