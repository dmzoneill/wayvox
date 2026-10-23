"""Shared helpers for the push-to-talk tests."""

import math
import os
import struct
import sys
from unittest.mock import (
    MagicMock,
    patch,
)

import dictation_output

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_silent_raw(duration_s=1.0, sample_rate=16000):
    """Generate silent raw PCM."""
    count = int(sample_rate * duration_s)
    return struct.pack(f"<{count}h", *([0] * count))


def _sine_samples(count, amplitude, sample_rate):
    """Return a 440 Hz sine wave as 16-bit sample values."""
    cycles = 2 * math.pi * 440 / sample_rate
    return [int(amplitude * math.sin(cycles * index)) for index in range(count)]


def fake_glib_modules(mock_glib):
    """sys.modules map that exposes *mock_glib* as gi.repository.GLib."""
    repository = MagicMock(GLib=mock_glib)
    return {"gi": MagicMock(), "gi.repository": repository}


def call_dbus(action, return_value=None, side_effect=None):
    """Run *action* with a fake GLib and a stubbed try_dbus_call."""
    mock_glib = MagicMock()
    options = {"side_effect": side_effect} if side_effect is not None else {}
    if side_effect is None:
        options["return_value"] = return_value
    with (
        patch.dict("sys.modules", fake_glib_modules(mock_glib)),
        patch.object(dictation_output, "try_dbus_call", **options),
    ):
        return action(), mock_glib


def _loud_raw(duration_s=0.5, amplitude=16000, sample_rate=16000):
    """Raw PCM with a loud sine wave (well above any silence threshold)."""
    count = int(sample_rate * duration_s)
    samples = _sine_samples(count, amplitude, sample_rate)
    return struct.pack(f"<{count}h", *samples)


def _quiet_raw(duration_s=0.5, amplitude=5, sample_rate=16000):
    """Raw PCM that is extremely quiet (below typical thresholds)."""
    count = int(sample_rate * duration_s)
    samples = _sine_samples(count, amplitude, sample_rate)
    return struct.pack(f"<{count}h", *samples)


class _AsyncCtx:
    """Minimal async context manager for mocking ``async with``."""

    def __init__(self, return_value):
        self._rv = return_value

    async def __aenter__(self):
        return self._rv

    async def __aexit__(self, *a):
        return False


def _make_aiohttp_mocks(resp):
    """Build mock aiohttp module where ClientSession().post() returns *resp*.

    Usage::

        resp = AsyncMock(status=200, ...)
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)
        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            ...
    """
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=_AsyncCtx(resp))
    mock_session.get = MagicMock(return_value=_AsyncCtx(resp))

    mock_aiohttp = MagicMock()
    mock_aiohttp.ClientSession = MagicMock(return_value=_AsyncCtx(mock_session))
    mock_aiohttp.ClientTimeout = MagicMock()
    mock_aiohttp.ClientError = Exception
    return mock_aiohttp, mock_session


# ── Constants ────────────────────────────────────────────────────────────────


class _AsyncIterLines:
    """Async iterator over a list of byte lines — mocks ``resp.content``."""

    def __init__(self, lines):
        self._it = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
