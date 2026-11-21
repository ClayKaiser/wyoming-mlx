"""Unit tests for Kokoro model memory lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

import wyoming_mlx.backends.mlx_tts as mlx_tts
from wyoming_mlx.backends.mlx_tts import KokoroBackend


class _FakeAudio:
    def numel(self) -> int:
        return 1

    def detach(self) -> _FakeAudio:
        return self

    def cpu(self) -> _FakeAudio:
        return self

    def numpy(self) -> np.ndarray:
        return np.array([0.25], dtype=np.float32)


class _FakePipeline:
    def __call__(self, _text: str, *, voice: str) -> list[object]:
        assert voice == "af_heart"
        return [type("Result", (), {"audio": _FakeAudio()})()]


def _backend(
    monkeypatch: pytest.MonkeyPatch,
    pipelines: list[_FakePipeline],
    idle_unload_seconds: float,
) -> KokoroBackend:
    def create_pipeline(**_kwargs: object) -> _FakePipeline:
        pipeline = _FakePipeline()
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(mlx_tts, "_kokoro_ok", True)
    monkeypatch.setattr(mlx_tts, "_KPipeline", create_pipeline)
    monkeypatch.setattr(mlx_tts, "_release_torch_memory", lambda: None)
    return KokoroBackend(idle_unload_seconds=idle_unload_seconds)


async def _synthesize(backend: KokoroBackend, text: str) -> list[bytes]:
    # Kokoro is optional and mocked in this unit test.
    synthesize: Callable[[str, str | None], AsyncIterator[bytes]] = backend.synthesize
    return [chunk async for chunk in synthesize(text, "af_heart")]


async def test_pipeline_loads_on_demand_and_unloads_after_idle(
    monkeypatch: pytest.MonkeyPatch,
):
    pipelines: list[_FakePipeline] = []
    backend = _backend(monkeypatch, pipelines, idle_unload_seconds=0.01)
    assert backend._pipeline is None

    chunks = await _synthesize(backend, "hello")
    assert chunks
    assert len(pipelines) == 1
    await asyncio.sleep(0.03)
    assert backend._pipeline is None


async def test_new_request_reuses_pipeline_and_resets_idle_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    pipelines: list[_FakePipeline] = []
    backend = _backend(monkeypatch, pipelines, idle_unload_seconds=0.2)
    _ = await _synthesize(backend, "first")
    await asyncio.sleep(0.02)
    _ = await _synthesize(backend, "second")

    assert len(pipelines) == 1
    await asyncio.sleep(0.1)
    assert backend._pipeline is not None
    await asyncio.sleep(0.12)
    assert backend._pipeline is None
