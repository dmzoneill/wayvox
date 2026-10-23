"""Tests for the Wayvox transcription server managers."""

import json
import os
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from server_support import (
    LLMManager,
    extract_perf_metrics,
    make_mock_result,
    mock_librosa,
    mock_ov,
    server_mod,
)

pytest_plugins = ["server_support"]


class TestMetricsCollector:
    """Tests covering metrics collector."""

    def test_initial_snapshot(self, fresh_metrics):
        """Initial snapshot."""
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 0
        assert snap["transcription_errors"] == 0
        assert snap["average_latency_seconds"] == 0.0
        assert snap["error_rate"] == 0.0
        assert snap["total_audio_seconds"] == 0.0
        assert snap["model_load_times"] == {}
        assert "uptime_seconds" in snap
        assert "last_request" not in snap  # empty → omitted

    def test_record_transcription_basic(self, fresh_metrics):
        """Record transcription basic."""
        fresh_metrics.record_transcription(0.5, 2.0)
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 1
        assert snap["average_latency_seconds"] == 0.5
        assert snap["total_audio_seconds"] == 2.0
        assert "last_request" not in snap  # no perf_metrics passed

    def test_record_transcription_with_perf_metrics(self, fresh_metrics):
        """Record transcription with perf metrics."""
        perf = {"inference_ms": 123.4}
        fresh_metrics.record_transcription(1.0, 3.0, perf_metrics=perf)
        snap = fresh_metrics.snapshot()
        assert snap["last_request"] == perf

    def test_record_multiple_transcriptions_averages(self, fresh_metrics):
        """Record multiple transcriptions averages."""
        fresh_metrics.record_transcription(1.0, 4.0)
        fresh_metrics.record_transcription(3.0, 6.0)
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 2
        assert snap["average_latency_seconds"] == 2.0
        assert snap["total_audio_seconds"] == 10.0

    def test_record_error(self, fresh_metrics):
        """Record error."""
        fresh_metrics.record_error()
        fresh_metrics.record_error()
        snap = fresh_metrics.snapshot()
        assert snap["transcription_errors"] == 2
        assert snap["error_rate"] == 1.0  # 0 successes, 2 errors

    def test_error_rate_mixed(self, fresh_metrics):
        """Error rate mixed."""
        fresh_metrics.record_transcription(1.0, 1.0)
        fresh_metrics.record_error()
        snap = fresh_metrics.snapshot()
        # 1 success + 1 error = error_rate 0.5
        assert snap["error_rate"] == 0.5

    def test_record_model_load(self, fresh_metrics):
        """Record model load."""
        fresh_metrics.record_model_load("whisper-small-en", 2.567)
        snap = fresh_metrics.snapshot()
        assert snap["model_load_times"] == {"whisper-small-en": 2.57}

    def test_thread_safety(self, fresh_metrics):
        """Concurrent writers should not corrupt data."""

        def record_many():
            """Record many."""
            for _ in range(100):
                fresh_metrics.record_transcription(0.01, 0.1)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = fresh_metrics.snapshot()
        assert snap["transcription_count"] == 400


# ===================================================================
#  2. ModelManager
# ===================================================================


class TestModelManager:
    """Tests covering model manager."""

    def test_list_models_with_models(self, model_mgr):
        """List models with models."""
        os.makedirs(os.path.join(model_mgr.models_dir, "whisper-small-en"))
        os.makedirs(os.path.join(model_mgr.models_dir, "whisper-large-v3"))
        models = model_mgr.list_models()
        assert set(models) == {"whisper-small-en", "whisper-large-v3"}

    def test_list_models_empty(self, model_mgr):
        """List models empty."""
        assert model_mgr.list_models() == []

    def test_list_models_hidden_dirs_filtered(self, model_mgr):
        """List models hidden dirs filtered."""
        os.makedirs(os.path.join(model_mgr.models_dir, ".hidden"))
        os.makedirs(os.path.join(model_mgr.models_dir, "visible"))
        models = model_mgr.list_models()
        assert models == ["visible"]

    def test_list_models_files_filtered(self, model_mgr):
        """Regular files (not directories) should not appear."""
        note = os.path.join(model_mgr.models_dir, "not-a-dir.txt")
        with open(note, "w", encoding="utf-8") as handle:
            handle.write("")
        assert model_mgr.list_models() == []

    def test_load_model_cache_miss(self, model_mgr, monkeypatch):
        """Load model cache miss."""
        monkeypatch.setenv("WHISPER_NPU_WORD_TIMESTAMPS", "1")
        monkeypatch.setenv("WHISPER_CACHE_DIR", "/tmp/wayvox-test-cache")
        monkeypatch.setattr(server_mod, "ENGINE", "openvino")
        monkeypatch.setattr(server_mod, "DEVICE", "NPU")
        monkeypatch.setattr(server_mod.ENGINES, "openvino_genai", mock_ov)
        model_dir = os.path.join(model_mgr.models_dir, "whisper-small-en")
        os.makedirs(model_dir, exist_ok=True)
        model_mgr.load_model("whisper-small-en")
        mock_ov.WhisperPipeline.assert_called_once_with(
            model_dir,
            device="NPU",
            STATIC_PIPELINE=False,
            word_timestamps=True,
            CACHE_DIR="/tmp/wayvox-test-cache",
        )
        assert "whisper-small-en" in model_mgr.pipelines

    def test_load_model_cache_hit(self, model_mgr):
        """Load model cache hit."""
        sentinel = MagicMock(name="cached_pipeline")
        model_mgr.pipelines["whisper-small-en"] = sentinel
        result = model_mgr.load_model("whisper-small-en")
        assert result is sentinel
        mock_ov.WhisperPipeline.assert_not_called()

    def test_load_model_not_found(self, model_mgr):
        """Load model not found."""
        with pytest.raises(FileNotFoundError, match="not found"):
            model_mgr.load_model("nonexistent-model")


# ===================================================================
#  3. LLMManager
# ===================================================================


class TestLLMManager:
    """Tests covering l lm manager."""

    def test_init_with_env_model(self, tmp_path):
        """When LLM_MODEL env is set, current_model should reflect it."""
        mgr = LLMManager.__new__(LLMManager)
        mgr.models_dir = str(tmp_path / "llm")
        mgr.pipeline = None
        mgr.current_model = "my-llm"
        assert mgr.current_model == "my-llm"

    def test_init_auto_select_first(self, llm_mgr):
        """list_models returns models; auto-select picks first."""
        os.makedirs(os.path.join(llm_mgr.models_dir, "alpha-model"))
        os.makedirs(os.path.join(llm_mgr.models_dir, "beta-model"))
        models = llm_mgr.list_models()
        assert len(models) >= 1
        # Simulate auto-select behavior
        llm_mgr.current_model = models[0]
        assert llm_mgr.current_model in models

    def test_list_models_empty_dir(self, llm_mgr):
        """List models empty dir."""
        assert llm_mgr.list_models() == []

    def test_list_models_no_dir(self, tmp_path):
        """List models no dir."""
        mgr = LLMManager.__new__(LLMManager)
        mgr.models_dir = str(tmp_path / "does-not-exist")
        mgr.pipeline = None
        mgr.current_model = ""
        assert mgr.list_models() == []

    def test_list_models_hidden_filtered(self, llm_mgr):
        """List models hidden filtered."""
        os.makedirs(os.path.join(llm_mgr.models_dir, ".cache"))
        os.makedirs(os.path.join(llm_mgr.models_dir, "llama-3"))
        assert llm_mgr.list_models() == ["llama-3"]

    def test_load_model_no_model_configured(self, llm_mgr):
        """Load model no model configured."""
        llm_mgr.current_model = ""
        with pytest.raises(ValueError, match="No LLM model configured"):
            llm_mgr.load_model()

    def test_load_model_not_found(self, llm_mgr):
        """Load model not found."""
        llm_mgr.current_model = "nonexistent"
        with pytest.raises(FileNotFoundError, match="not found"):
            llm_mgr.load_model()

    def test_load_model_success(self, llm_mgr):
        """Load model success."""
        model_name = "llama-3"
        model_dir = os.path.join(llm_mgr.models_dir, model_name)
        os.makedirs(model_dir)
        llm_mgr.load_model(model_name)
        mock_ov.LLMPipeline.assert_called_once()
        assert llm_mgr.current_model == model_name
        assert llm_mgr.pipeline is not None

    def test_load_model_already_loaded_same_model(self, llm_mgr):
        """If model is already loaded, don't reload."""
        model_name = "llama-3"
        model_dir = os.path.join(llm_mgr.models_dir, model_name)
        os.makedirs(model_dir)
        sentinel = MagicMock(name="existing_pipeline")
        llm_mgr.pipeline = sentinel
        llm_mgr.current_model = model_name
        result = llm_mgr.load_model(model_name)
        assert result is sentinel
        mock_ov.LLMPipeline.assert_not_called()

    def test_load_model_switch_model(self, llm_mgr):
        """Switching to a different model should reload."""
        for name in ("model-a", "model-b"):
            os.makedirs(os.path.join(llm_mgr.models_dir, name))
        llm_mgr.current_model = "model-a"
        llm_mgr.pipeline = MagicMock()
        llm_mgr.load_model("model-b")
        mock_ov.LLMPipeline.assert_called_once()
        assert llm_mgr.current_model == "model-b"

    def test_rewrite(self, llm_mgr):
        """Rewrite."""
        model_dir = os.path.join(llm_mgr.models_dir, "llm")
        os.makedirs(model_dir)
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = "  Rewritten text  "
        llm_mgr.pipeline = mock_pipe
        llm_mgr.current_model = "llm"
        result = llm_mgr.rewrite("hello world", "diplomatic", "Be diplomatic")
        assert result == "Rewritten text"
        mock_pipe.generate.assert_called_once()
        call_args = mock_pipe.generate.call_args
        assert "hello world" in call_args[0][0]
        assert call_args[1]["max_new_tokens"] == 512

    def test_punctuate(self, llm_mgr):
        """Punctuate."""
        model_dir = os.path.join(llm_mgr.models_dir, "llm")
        os.makedirs(model_dir)
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = "  Hello, world.  "
        llm_mgr.pipeline = mock_pipe
        llm_mgr.current_model = "llm"
        result = llm_mgr.punctuate("hello world")
        assert result == "Hello, world."
        call_args = mock_pipe.generate.call_args
        assert call_args[1]["max_new_tokens"] == 256
        assert call_args[1]["temperature"] == 0.1

    def test_translate(self, llm_mgr):
        """Translate."""
        model_dir = os.path.join(llm_mgr.models_dir, "llm")
        os.makedirs(model_dir)
        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = "  Hola mundo  "
        llm_mgr.pipeline = mock_pipe
        llm_mgr.current_model = "llm"
        result = llm_mgr.translate("hello world", "Spanish")
        assert result == "Hola mundo"
        call_args = mock_pipe.generate.call_args
        assert "Spanish" in call_args[0][0]
        assert call_args[1]["max_new_tokens"] == 512


# ===================================================================
#  4. extract_perf_metrics
# ===================================================================


class TestExtractPerfMetrics:
    """Tests covering extract perf metrics."""

    def test_success(self):
        """Success."""
        mock_pm = MagicMock()
        for attr in (
            "get_features_extraction_duration",
            "get_inference_duration",
            "get_generate_duration",
            "get_detokenization_duration",
            "get_throughput",
        ):
            dur = MagicMock()
            dur.mean = 42.123
            getattr(mock_pm, attr).return_value = dur
        mock_pm.get_num_generated_tokens.return_value = 10

        result_obj = MagicMock()
        result_obj.perf_metrics = mock_pm
        out = extract_perf_metrics(result_obj)
        assert out["features_extraction_ms"] == 42.12
        assert out["inference_ms"] == 42.12
        assert out["generate_ms"] == 42.12
        assert out["detokenization_ms"] == 42.12
        assert out["throughput_tokens_per_sec"] == 42.12
        assert out["num_generated_tokens"] == 10

    def test_exception_returns_empty(self):
        """Exception returns empty."""
        result_obj = MagicMock(spec=[])  # spec=[] means no attributes
        assert extract_perf_metrics(result_obj) == {}


# ===================================================================
#  5-20. Flask route tests
# ===================================================================


def _npu_report():
    """OpenVINO section used to render the system report."""
    device = {
        "id": "NPU",
        "properties": {"FULL_DEVICE_NAME": "Intel AI Boost"},
        "plugins": {},
    }
    return {
        "importable": True,
        "version": "2026.4.1",
        "devices": [device],
    }


class TestHealthRoute:
    """Tests covering health route."""

    def test_health(self, client):
        """Health."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_system_report(self, client, monkeypatch):
        """System report."""
        monkeypatch.setattr(
            server_mod.diagnostics,
            "collect",
            lambda: {
                "app": {"version": "0.0.2"},
                "python": "3.14.0",
                "python_executable": "/venv/bin/python",
                "platform": "Linux",
                "cpu": "test-cpu",
                "libraries": {"openvino": "2026.4.1", "openvino-genai": "2026.4.1"},
                "openvino": _npu_report(),
                "pci": [
                    {
                        "kind": "NPU",
                        "slot": "00:0b.0",
                        "vendor": "Intel",
                        "device": "Meteor Lake NPU",
                    }
                ],
                "modules": {"intel_vpu": {"loaded": True, "version": "1"}},
                "nvidia": [],
            },
        )
        resp = client.get("/system")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["source"] == "wayvox-server"
        assert data["runtime"]["engine"]
        assert data["runtime"]["device"]
        assert "OpenVINO GenAI WhisperPipeline" in data["runtime"]["interface"]
        assert "Intel AI Boost" in data["text"]
        assert "Meteor Lake NPU" in data["text"]
        assert "Action:" in data["text"]


def test_list_models(client):
    """List models."""
    server_mod.model_manager.list_models = MagicMock(
        return_value=["whisper-small-en", "whisper-large-v3"],
    )
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "whisper-small-en" in data["models"]


class TestDefaultModelRoute:
    """Tests covering default model route."""

    def test_get_default_model(self, client):
        """Get default model."""
        resp = client.get("/model/default")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model" in data

    def test_set_default_model_missing_body(self, client):
        """Set default model missing body."""
        resp = client.put(
            "/model/default",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "model" in resp.get_json()["error"]

    def test_set_default_model_not_found(self, client):
        """Set default model not found."""
        server_mod.model_manager.list_models = MagicMock(return_value=["a"])
        resp = client.put(
            "/model/default",
            data=json.dumps({"model": "nonexistent"}),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_set_default_model_success(self, client):
        """Set default model success."""
        server_mod.model_manager.list_models = MagicMock(
            return_value=["whisper-small-en"],
        )
        server_mod.model_manager.load_model = MagicMock()
        resp = client.put(
            "/model/default",
            data=json.dumps({"model": "whisper-small-en"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["model"] == "whisper-small-en"


class TestTranscribeRoute:
    """Tests covering transcribe route."""

    def test_no_audio(self, client):
        """No audio."""
        server_mod.model_manager.load_model = MagicMock()
        resp = client.post("/transcribe", data=b"")
        assert resp.status_code == 400
        assert "No audio" in resp.get_json()["error"]

    def test_success(self, client, wav_bytes):
        """Success."""
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = make_mock_result("hello world")
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe", data=wav_bytes)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["text"] == "hello world"

    def test_vocabulary_replacements(self, client, wav_bytes):
        """Vocabulary replacements."""
        with patch.object(
            server_mod.model_manager,
            "transcribe",
            return_value=("open a polar quest", {}, 0.01),
        ):
            resp = client.post("/transcribe", data=wav_bytes)
        assert resp.status_code == 200
        assert resp.get_json()["text"] == "open a pull request"

    def test_exception(self, client, wav_bytes):
        """Exception."""
        server_mod.model_manager.load_model = MagicMock(
            side_effect=RuntimeError("device error"),
        )
        resp = client.post("/transcribe", data=wav_bytes)
        assert resp.status_code == 500
        assert "device error" in resp.get_json()["error"]


class TestTranscribeWithModelRoute:
    """Tests covering transcribe with model route."""

    def test_with_language(self, client, wav_bytes):
        """With language."""
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = make_mock_result("bonjour")
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe/whisper-small-en?language=fr", data=wav_bytes)
        assert resp.status_code == 200
        # Verify language kwarg was passed
        call_kwargs = mock_pipeline.generate.call_args[1]
        assert call_kwargs["language"] == "<|fr|>"

    def test_model_not_found(self, client, wav_bytes):
        """Model not found."""
        server_mod.model_manager.load_model = MagicMock(
            side_effect=FileNotFoundError("not found"),
        )
        resp = client.post("/transcribe/bad-model", data=wav_bytes)
        assert resp.status_code == 500
        assert "not found" in resp.get_json()["error"]


class TestTranscribeStreamRoute:
    """Tests covering transcribe stream route."""

    def test_no_audio(self, client):
        """No audio."""
        server_mod.model_manager.load_model = MagicMock()
        resp = client.post("/transcribe/stream", data=b"")
        assert resp.status_code == 400

    def test_duration_too_long(self, client, wav_bytes):
        """Duration too long."""
        mock_pipeline = MagicMock()
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        # 31 seconds of audio at 16kHz
        long_audio = np.zeros(16000 * 31, dtype=np.float32)
        mock_librosa.load.return_value = (long_audio, 16000)

        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 400
        assert "30 seconds" in resp.get_json()["error"]

    def test_success_sse(self, client, wav_bytes):
        """Success sse."""
        mock_pipeline = MagicMock()

        # Simulate streaming: the pipeline.generate call invokes the streamer
        # callback. The server runs pipeline.generate in a background thread
        # and reads chunks from a queue, so the mock must call the streamer
        # that the server passes in.
        def fake_generate(_audio, **kwargs):
            """Fake generate."""
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("hello ")
                streamer("world")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds
        mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

        # Parse SSE events from response data
        raw = resp.get_data(as_text=True)
        events = []
        for line in raw.split("\n"):
            if line.startswith("data: "):
                events.append(line[len("data: ") :])
        assert len(events) >= 2  # chunk events + done event
        # Last event should have done: true
        last = json.loads(events[-1])
        assert last["done"] is True
        assert last["full_text"] == "hello world"

    def test_stream_with_model_name(self, client, wav_bytes):
        """Test the /transcribe/stream/<model_name> route."""
        mock_pipeline = MagicMock()

        def fake_generate(_audio, **kwargs):
            """Fake generate."""
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("test")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000, dtype=np.float32)
        mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream/whisper-small-en", data=wav_bytes)
        assert resp.status_code == 200
        server_mod.model_manager.load_model.assert_called_with("whisper-small-en")

    def test_stream_with_language(self, client, wav_bytes):
        """Verify that language query param is threaded through the streaming path."""
        mock_pipeline = MagicMock()
        captured_kwargs = {}

        def fake_generate(_audio, **kwargs):
            """Fake generate."""
            captured_kwargs.update(kwargs)
            streamer = kwargs.get("streamer")
            if streamer:
                streamer("bonjour")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000 * 2, dtype=np.float32)
        mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream?language=fr", data=wav_bytes)
        assert resp.status_code == 200
        # Consume the response to ensure the background thread runs
        resp.get_data()
        assert captured_kwargs.get("language") == "<|fr|>"

    def test_stream_inference_error(self, client, wav_bytes):
        """If pipeline.generate raises inside the thread, the stream still completes."""
        mock_pipeline = MagicMock()

        def fake_generate(_audio, **kwargs):
            """Fake generate."""
            raise RuntimeError("inference exploded")

        mock_pipeline.generate.side_effect = fake_generate
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        short_audio = np.zeros(16000, dtype=np.float32)
        mock_librosa.load.return_value = (short_audio, 16000)

        resp = client.post("/transcribe/stream", data=wav_bytes)
        assert resp.status_code == 200
        raw = resp.get_data(as_text=True)
        # The stream should still produce a done event (from the finally block)
        events = []
        for line in raw.split("\n"):
            if line.startswith("data: "):
                events.append(line[len("data: ") :])
        last = json.loads(events[-1])
        assert last["done"] is True
        assert last["full_text"] == ""
