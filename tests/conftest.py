"""Shared fixtures for Wayvox tests."""

import io
import json
import math
import os
import struct
import sys
import wave
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Keep server tests off the user's real vocabulary file.
os.environ.setdefault(
    "WHISPER_VOCAB_PATH",
    os.path.join(os.path.expanduser("~"), ".cache", "wayvox-pytest-vocab-absent.json"),
)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _sine_samples(count, amplitude, sample_rate):
    """Return a 440 Hz sine wave as 16-bit sample values."""
    samples = []
    for index in range(count):
        phase = 2 * math.pi * 440 * index / sample_rate
        samples.append(int(amplitude * math.sin(phase)))
    return samples


def make_wav_bytes(
    samples=None,
    duration_s=1.0,
    sample_rate=16000,
    channels=1,
    sample_width=2,
):
    """Generate a valid WAV file in memory.

    If *samples* is None, generates a sine wave of *duration_s* seconds.
    """
    if samples is None:
        n = int(sample_rate * duration_s)
        samples = _sine_samples(n, 16000, sample_rate)

    buf = io.BytesIO()
    with wave.Wave_write(buf) as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        raw = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(raw)
    return buf.getvalue()


def make_raw_audio(samples=None, duration_s=1.0, sample_rate=16000):
    """Generate raw 16-bit LE PCM bytes (no WAV header)."""
    if samples is None:
        n = int(sample_rate * duration_s)
        samples = _sine_samples(n, 16000, sample_rate)
    return struct.pack(f"<{len(samples)}h", *samples)


def make_silent_raw(duration_s=1.0, sample_rate=16000):
    """Generate silent raw PCM."""
    n = int(sample_rate * duration_s)
    return struct.pack(f"<{n}h", *([0] * n))


@pytest.fixture
def wav_bytes():
    """Return a one-second in-memory WAV fixture."""
    return make_wav_bytes(duration_s=1.0)


@pytest.fixture
def raw_audio():
    """Return one second of raw 16-bit audio."""
    return make_raw_audio(duration_s=1.0)


@pytest.fixture
def silent_raw():
    """Return one second of silent raw PCM."""
    return make_silent_raw(duration_s=1.0)


# ---------------------------------------------------------------------------
# Temporary config
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary Wayvox config directory."""
    config_dir = tmp_path / "wayvox"
    config_dir.mkdir()
    settings = {
        "server-host": "127.0.0.1",
        "server-port": 5000,
        "language-buddy-enabled": True,
        "language-buddy-bypass": False,
        "language-buddy-timeout": 30,
    }
    (config_dir / "settings.json").write_text(json.dumps(settings))
    return config_dir


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary models directory with some fake models."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in ["whisper-small-en", "whisper-medium-en", "whisper-large-v3"]:
        (models_dir / name).mkdir()
    return models_dir


@pytest.fixture
def tmp_llm_models_dir(tmp_path):
    """Create a temporary LLM models directory."""
    llm_dir = tmp_path / "llm-models"
    llm_dir.mkdir()
    (llm_dir / "llama-3-int4-ov").mkdir()
    return llm_dir


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_subprocess():
    """Patch subprocess.run to succeed silently."""
    with patch("subprocess.run") as m:
        m.return_value = MagicMock(returncode=0)
        yield m


@pytest.fixture
def mock_popen():
    """Patch subprocess.Popen."""
    with patch("subprocess.Popen") as m:
        proc = MagicMock()
        proc.stdout.read.return_value = b""
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        m.return_value = proc
        yield m, proc
