"""Unit tests for WhisperLiveKit backend helpers (no whisperlivekit install required)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import wyoming_mlx.backends.wlk_stt as wlk_stt
from wyoming_mlx.backends.wlk_stt import (
    WHISPER_SAMPLE_RATE,
    WhisperLiveKitBackend,
    _delta,
    _joined_text,
    _resample_to_16k,
    _WLKSession,
    resample_pcm16,
)


def test_backend_rejects_hf_repo_id():
    """A Hugging Face repo id is rejected up front, before any model load."""
    with pytest.raises(ValueError, match="repo id"):
        WhisperLiveKitBackend(model="mlx-community/distil-whisper-large-v3")


def test_resample_passthrough_at_16k():
    pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    out = _resample_to_16k(pcm, 16000)
    assert out is pcm or np.array_equal(out, pcm)
    assert out.dtype == np.float32


def test_resample_24k_to_16k_changes_length_correctly():
    pcm = np.zeros(24000, dtype=np.float32)
    out = _resample_to_16k(pcm, 24000)
    assert out.dtype == np.float32
    assert out.shape == (WHISPER_SAMPLE_RATE,)


def test_resample_preserves_sinusoid_frequency():
    sr_in = 24000
    duration = 0.5
    freq = 440.0
    t = np.arange(int(sr_in * duration), dtype=np.float32) / sr_in
    pcm = np.sin(2 * np.pi * freq * t).astype(np.float32)
    out = _resample_to_16k(pcm, sr_in)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), d=1 / 16000)[int(np.argmax(spectrum))]
    assert abs(peak_hz - freq) < 5


def test_resample_pcm16_passthrough_at_16k_is_lossless():
    audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16).tobytes()
    assert resample_pcm16(audio, 16000) == audio


def test_resample_pcm16_halves_48k_input():
    audio = np.zeros(4800, dtype=np.int16).tobytes()
    out = resample_pcm16(audio, 48000)
    assert len(out) == 1600 * 2  # 1600 samples of int16


def test_joined_text_skips_empty_segments():
    lines = [
        SimpleNamespace(text="hello"),
        SimpleNamespace(text=None),
        SimpleNamespace(text="  "),
        SimpleNamespace(text="world"),
    ]
    assert _joined_text(lines) == "hello world"


def test_delta_returns_new_suffix():
    assert _delta("hello ", "hello world") == "world"


def test_delta_empty_when_not_a_prefix():
    assert _delta("hello there", "hello world") == ""


def test_delta_from_empty_emits_everything():
    assert _delta("", "hello") == "hello"


def _session_with_results(fronts: list[SimpleNamespace]) -> _WLKSession:
    """Build a _WLKSession without an AudioProcessor, injecting fabricated results."""
    session = object.__new__(_WLKSession)
    session._emitted = ""

    async def results():
        for front in fronts:
            yield front

    session._results = results()
    return session


def _front(texts: list[str | None], buffer: str = "", status: str = "active_transcription"):
    return SimpleNamespace(
        status=status,
        error="",
        lines=[SimpleNamespace(text=t) for t in texts],
        buffer_transcription=buffer,
    )


async def test_updates_emits_deltas_then_final_with_buffer_tail():
    session = _session_with_results(
        [
            _front(["hello"]),
            _front(["hello", "world"], buffer="tail "),
        ]
    )

    updates = [u async for u in session.updates()]

    assert [u.text for u in updates[:-1]] == ["hello", " world"]
    assert updates[-1].final == "hello world tail"


async def test_updates_yields_only_final_when_nothing_confirmed():
    session = _session_with_results([_front([], status="no_audio_detected")])

    updates = [u async for u in session.updates()]

    assert len(updates) == 1
    assert updates[0].final == ""


async def test_updates_raises_on_error_status():
    session = _session_with_results(
        [SimpleNamespace(status="error", error="boom", lines=[], buffer_transcription="")]
    )

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in session.updates():
            pass


async def test_backend_loads_on_demand_and_unloads_after_idle(monkeypatch: pytest.MonkeyPatch):
    class FakeProcessor:
        def __init__(self, *, transcription_engine: object) -> None:
            self.engine = transcription_engine

        async def cleanup(self) -> None:
            pass

    engines: list[object] = []

    def create_engine(**_kwargs: object) -> object:
        engine = object()
        engines.append(engine)
        return engine

    monkeypatch.setattr(wlk_stt, "_WLK_AVAILABLE", True)
    monkeypatch.setattr(wlk_stt, "_TranscriptionEngine", create_engine)
    monkeypatch.setattr(wlk_stt, "_AudioProcessor", FakeProcessor)
    monkeypatch.setattr(wlk_stt, "_release_mlx_memory", lambda: None)

    backend = WhisperLiveKitBackend(idle_unload_seconds=0.01)
    assert backend._engine is None

    session = backend.start_session()
    assert len(engines) == 1
    await session.close()
    await asyncio.sleep(0.03)
    assert backend._engine is None

    next_session = backend.start_session()
    assert len(engines) == 2
    await next_session.close()


async def test_backend_keeps_engine_until_all_sessions_close(monkeypatch: pytest.MonkeyPatch):
    class FakeProcessor:
        def __init__(self, *, transcription_engine: object) -> None:
            self.engine = transcription_engine

        async def cleanup(self) -> None:
            pass

    monkeypatch.setattr(wlk_stt, "_WLK_AVAILABLE", True)
    monkeypatch.setattr(wlk_stt, "_TranscriptionEngine", lambda **_kwargs: object())
    monkeypatch.setattr(wlk_stt, "_AudioProcessor", FakeProcessor)
    monkeypatch.setattr(wlk_stt, "_release_mlx_memory", lambda: None)

    backend = WhisperLiveKitBackend(idle_unload_seconds=0.01)
    first = backend.start_session()
    second = backend.start_session()

    await first.close()
    await asyncio.sleep(0.03)
    assert backend._engine is not None

    await second.close()
    await asyncio.sleep(0.03)
    assert backend._engine is None


async def test_new_session_cancels_pending_unload(monkeypatch: pytest.MonkeyPatch):
    class FakeProcessor:
        def __init__(self, *, transcription_engine: object) -> None:
            self.engine = transcription_engine

        async def cleanup(self) -> None:
            pass

    monkeypatch.setattr(wlk_stt, "_WLK_AVAILABLE", True)
    monkeypatch.setattr(wlk_stt, "_TranscriptionEngine", lambda **_kwargs: object())
    monkeypatch.setattr(wlk_stt, "_AudioProcessor", FakeProcessor)
    monkeypatch.setattr(wlk_stt, "_release_mlx_memory", lambda: None)

    backend = WhisperLiveKitBackend(idle_unload_seconds=0.05)
    first = backend.start_session()
    await first.close()
    await asyncio.sleep(0.02)

    second = backend.start_session()
    await asyncio.sleep(0.04)
    assert backend._engine is not None

    await second.close()
    await asyncio.sleep(0.06)
    assert backend._engine is None
