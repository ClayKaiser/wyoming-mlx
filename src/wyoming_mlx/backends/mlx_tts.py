"""Kokoro TTS backend using PyTorch MPS on Apple Silicon."""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import warnings
from collections.abc import AsyncGenerator

import numpy as np

_kokoro_ok = False
try:
    from kokoro import KPipeline as _KPipeline  # type: ignore[import-not-found, unused-ignore]

    _kokoro_ok = True
except ImportError:
    _KPipeline = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class KokoroBackend:
    """TTS backend powered by Kokoro-82M via PyTorch MPS."""

    def __init__(
        self,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        idle_unload_seconds: float = 60,
    ) -> None:
        if not _kokoro_ok:
            raise ImportError("kokoro is required for KokoroBackend")
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
        self._pipeline: object = None
        self._lock = asyncio.Lock()
        self._unload_handle: asyncio.TimerHandle | None = None

    def _ensure_pipeline(self) -> object:
        if self._pipeline is None:
            assert _KPipeline is not None, "kokoro is not installed"
            self._pipeline = _KPipeline(lang_code="a", repo_id=self._model_id, device="mps")
        return self._pipeline

    def _cancel_unload(self) -> None:
        if self._unload_handle is not None:
            self._unload_handle.cancel()
            self._unload_handle = None

    def _schedule_unload(self) -> None:
        self._unload_handle = asyncio.get_running_loop().call_later(
            self._idle_unload_seconds, self._unload_pipeline
        )

    def _unload_pipeline(self) -> None:
        self._unload_handle = None
        if self._pipeline is None:
            return
        log.info("[TTS] Unloading Kokoro pipeline after idle timeout")
        self._pipeline = None
        _release_torch_memory()

    def _synthesize_locked(self, text: str, voice: str) -> bytes:
        pipeline = self._ensure_pipeline()
        generator = pipeline(text, voice=voice)  # pyright: ignore
        buf = io.BytesIO()
        # Suppress Kokoro internal STFT resize warnings (PyTorch bug with
        # empty/half-empty harmonic source tensors — harmless, ignored).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*was resized since it had shape.*")
            for result in generator:
                audio = result.audio
                assert audio is not None
                if audio.numel() == 0:
                    continue
                arr = audio.detach().cpu().numpy().astype(np.float32)
                pcm = (arr * 32767).astype(np.int16).tobytes()
                buf.write(pcm)
        return buf.getvalue()

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
        # Yield in chunks
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
