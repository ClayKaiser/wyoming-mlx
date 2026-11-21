"""Unit tests for Kokoro MLX model memory lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

import wyoming_mlx.backends.mlx_tts as mlx_tts
from wyoming_mlx.backends.mlx_tts import KokoroBackend


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def generate(self, text: str, *, voice: str, lang_code: str) -> list[object]:
        self.calls.append((text, voice, lang_code))
        return [type("Result", (), {"audio": np.array([0.25, 1.5, -1.5])})()]


def _backend(
    monkeypatch: pytest.MonkeyPatch,
    models: list[_FakeModel],
    idle_unload_seconds: float,
) -> KokoroBackend:
    def load_model(model_id: str) -> _FakeModel:
        assert model_id == "mlx-community/Kokoro-82M-bf16"
        model = _FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(mlx_tts, "_load_model", load_model)
    monkeypatch.setattr(mlx_tts, "_release_mlx_memory", lambda: None)
    return KokoroBackend(idle_unload_seconds=idle_unload_seconds)


async def _synthesize(backend: KokoroBackend, text: str, voice: str = "af_heart") -> list[bytes]:
    synthesize: Callable[[str, str | None], AsyncIterator[bytes]] = backend.synthesize
    return [chunk async for chunk in synthesize(text, voice)]


async def test_model_loads_on_demand_and_unloads_after_idle(monkeypatch: pytest.MonkeyPatch):
    models: list[_FakeModel] = []
    backend = _backend(monkeypatch, models, idle_unload_seconds=0.01)
    assert backend._model is None

    chunks = await _synthesize(backend, "hello")
    assert len(models) == 1
    assert models[0].calls == [("hello", "af_heart", "a")]
    assert np.frombuffer(b"".join(chunks), dtype="<i2").tolist() == [8191, 32767, -32767]

    await asyncio.sleep(0.03)
    assert backend._model is None


async def test_new_request_reuses_model_and_resets_idle_timeout(monkeypatch: pytest.MonkeyPatch):
    models: list[_FakeModel] = []
    backend = _backend(monkeypatch, models, idle_unload_seconds=0.2)
    _ = await _synthesize(backend, "first")
    await asyncio.sleep(0.02)
    _ = await _synthesize(backend, "second", "jf_alpha")

    assert len(models) == 1
    assert models[0].calls == [("first", "af_heart", "a"), ("second", "jf_alpha", "j")]
    await asyncio.sleep(0.1)
    assert backend._model is not None
    await asyncio.sleep(0.12)
    assert backend._model is None
