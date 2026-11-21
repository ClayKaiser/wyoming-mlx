"""Kokoro TTS backend using MLX-Audio on Apple Silicon."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import suppress
from typing import Any

import numpy as np

_load_model: Callable[..., Any] | None = None
with suppress(ImportError):
    from mlx_audio.tts.utils import load_model as _load_model

log = logging.getLogger(__name__)


def _release_mlx_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx  # pyright: ignore[reportMissingImports]
    except ImportError:
        return
    mx.clear_cache()


class KokoroBackend:
    """TTS backend powered by Kokoro-82M via MLX-Audio."""

    def __init__(
        self,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        idle_unload_seconds: float = 60,
    ) -> None:
        if _load_model is None:
            raise ImportError("mlx-audio is required for KokoroBackend")
        if idle_unload_seconds < 0:
            raise ValueError("idle_unload_seconds must be non-negative")
        self._voice = voice
        self._model_id = model_id
        self._idle_unload_seconds = idle_unload_seconds
        self.sample_rate = 24000
        self.voices: list[str] = [
            # American English
            "af_heart",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            # British English
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
            # Japanese
            "jf_alpha",
            "jf_gongitsune",
            "jf_nezumi",
            "jf_tebukuro",
            "jm_kumo",
            # Mandarin Chinese
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
            # Spanish
            "ef_dora",
            "em_alex",
            "em_santa",
            # French
            "ff_siwis",
            # Hindi
            "hf_alpha",
            "hf_beta",
            "hm_omega",
            "hm_psi",
            # Italian
            "if_sara",
            "im_nicola",
            # Brazilian Portuguese
            "pf_dora",
            "pm_alex",
            "pm_santa",
        ]
        self._model: Any | None = None
        self._lock = asyncio.Lock()
        self._unload_handle: asyncio.TimerHandle | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            if _load_model is None:
                raise ImportError("mlx-audio is required for KokoroBackend")
            log.info("Loading Kokoro MLX model (model=%s) …", self._model_id)
            self._model = _load_model(self._model_id)
        return self._model

    def _cancel_unload(self) -> None:
        if self._unload_handle is not None:
            self._unload_handle.cancel()
            self._unload_handle = None

    def _schedule_unload(self) -> None:
        self._unload_handle = asyncio.get_running_loop().call_later(
            self._idle_unload_seconds, self._unload_model
        )

    def _unload_model(self) -> None:
        self._unload_handle = None
        if self._model is None:
            return
        log.info("[TTS] Unloading Kokoro MLX model after idle timeout")
        self._model = None
        _release_mlx_memory()

    def _synthesize_locked(self, text: str, voice: str) -> bytes:
        model = self._ensure_model()
        results: Iterator[Any] = model.generate(text, voice=voice, lang_code=voice[0])
        pcm_chunks: list[bytes] = []
        for result in results:
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                continue
            pcm_chunks.append((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        return b"".join(pcm_chunks)

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncGenerator[bytes, None]:
        v = voice or self._voice
        if v not in self.voices:
            raise ValueError(f"unknown voice: {v!r}, expected one of {self.voices}")
        async with self._lock:
            self._cancel_unload()
            # Serialize inference; response chunks are yielded after releasing
            # the lock so slow clients do not block other requests.
            try:
                data = self._synthesize_locked(text, v)
            finally:
                self._schedule_unload()
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
