"""Mock the transcription server once and share fixtures between test modules."""

import importlib
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock optional backends before importing the server. The module builds global
# managers at import time, and CI has no NPU or OpenVINO install.
os.environ["WHISPER_DEVICE"] = "NPU"

_mock_ov = MagicMock()
_mock_librosa = MagicMock()
_mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

sys.modules["openvino_genai"] = _mock_ov
sys.modules["librosa"] = _mock_librosa

_TEST_HOME = tempfile.mkdtemp(prefix="wayvox-test-home-")

_patches = [
    patch("os.path.exists", return_value=True),
    patch("os.listdir", return_value=["whisper-small-en"]),
    patch("os.path.isdir", return_value=True),
    patch(
        "os.path.expanduser",
        side_effect=lambda path: path.replace("~", _TEST_HOME),
    ),
]
for _patch in _patches:
    _patch.start()

server_mod = importlib.import_module("server_native")
server_mod.ENGINES.openvino_genai = _mock_ov
server_mod.ENGINES.openvino_genai_for_llm = _mock_ov

for _patch in _patches:
    _patch.stop()

mock_ov = _mock_ov
mock_librosa = _mock_librosa

MetricsCollector = server_mod.MetricsCollector
ModelManager = server_mod.ModelManager
LLMManager = server_mod.LLMManager
extract_perf_metrics = server_mod.extract_perf_metrics
DEFAULT_TONES = server_mod.DEFAULT_TONES
app = server_mod.app


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset shared mocks so one test cannot leak state into the next."""
    _mock_ov.reset_mock()
    _mock_librosa.reset_mock()
    _mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    server_mod.ENGINES.openvino_genai = _mock_ov
    server_mod.ENGINES.openvino_genai_for_llm = _mock_ov
    server_mod.ENGINES.faster_whisper_cls = None
    server_mod.ENGINE = "openvino"
    server_mod.DEVICE = "NPU"
    server_mod.LLM_DEVICE = "NPU"
    metrics = server_mod.metrics
    metrics.transcription_count = 0
    metrics.transcription_errors = 0
    metrics.total_latency = 0.0
    metrics.total_audio_duration = 0.0
    metrics.model_load_times = {}
    metrics.last_request = {}
    yield


def make_mock_result(text="hello world"):
    """Return a transcription result whose perf metrics stay serializable."""
    mock_result = MagicMock()
    mock_result.__str__ = lambda self: text
    mock_result.perf_metrics = MagicMock(spec=[])
    return mock_result


@pytest.fixture
def client():
    """Flask test client for the transcription server."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def fresh_metrics():
    """Return a new metrics collector with empty counters."""
    return MetricsCollector()


@pytest.fixture
def model_mgr(tmp_path):
    """Model manager pointed at a temporary models directory."""
    manager = ModelManager.__new__(ModelManager)
    manager.models_dir = str(tmp_path / "models")
    manager.ct2_models_dir = str(tmp_path / "ct2-models")
    manager.pipelines = {}
    manager.default_model = "whisper-small-en"
    os.makedirs(manager.models_dir, exist_ok=True)
    os.makedirs(manager.ct2_models_dir, exist_ok=True)
    return manager


@pytest.fixture
def llm_mgr(tmp_path):
    """LLM manager pointed at a temporary model directory."""
    manager = LLMManager.__new__(LLMManager)
    manager.models_dir = str(tmp_path / "llm-models")
    manager.pipeline = None
    manager.current_model = ""
    os.makedirs(manager.models_dir, exist_ok=True)
    return manager
