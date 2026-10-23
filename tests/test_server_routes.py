"""Route tests for the Wayvox transcription server."""

import json
import os
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from server_support import (
    DEFAULT_TONES,
    make_mock_result,
    mock_librosa,
    server_mod,
)

pytest_plugins = ["server_support"]


class TestRewriteRoute:
    """Tests covering rewrite route."""

    def test_missing_text(self, client):
        """Missing text."""
        resp = client.post(
            "/rewrite",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_text(self, client):
        """Empty text."""
        resp = client.post(
            "/rewrite",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_success_default_tones(self, client):
        """Success default tones."""
        server_mod.llm_manager.rewrite = MagicMock(return_value="Rewritten")
        resp = client.post(
            "/rewrite",
            data=json.dumps({"text": "fix this"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        variants = data["variants"]
        # First variant is always original
        assert variants[0]["tone"] == "original"
        assert variants[0]["text"] == "fix this"
        # Should have one variant per default tone
        assert len(variants) == 1 + len(DEFAULT_TONES)

    def test_success_custom_tones(self, client):
        """Success custom tones."""
        server_mod.llm_manager.rewrite = MagicMock(return_value="Custom rewrite")
        resp = client.post(
            "/rewrite",
            data=json.dumps(
                {
                    "text": "hello",
                    "tones": ["my_tone"],
                    "custom_tones": {"my_tone": "Be very casual"},
                },
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        variants = resp.get_json()["variants"]
        assert any(v["tone"] == "my_tone" for v in variants)

    def test_rewrite_exception(self, client):
        """Rewrite exception."""
        server_mod.llm_manager.rewrite = MagicMock(
            side_effect=RuntimeError("LLM failed"),
        )
        resp = client.post(
            "/rewrite",
            data=json.dumps({"text": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 200  # route returns 200 with error in variant
        variants = resp.get_json()["variants"]
        errored = [v for v in variants if "error" in v]
        assert len(errored) > 0
        assert "LLM failed" in errored[0]["error"]


def test_list_llm_models(client):
    """List llm models."""
    server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
    server_mod.llm_manager.current_model = "llama-3"
    resp = client.get("/llm/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "llama-3" in data["models"]
    assert data["current"] == "llama-3"


class TestSetLLMModelRoute:
    """Tests covering set ll mmodel route."""

    def test_missing_body(self, client):
        """Missing body."""
        resp = client.put(
            "/llm/model",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "model" in resp.get_json()["error"]

    def test_not_found(self, client):
        """Not found."""
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        resp = client.put(
            "/llm/model",
            data=json.dumps({"model": "nonexistent"}),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_success(self, client):
        """Success."""
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        server_mod.llm_manager.load_model = MagicMock()
        resp = client.put(
            "/llm/model",
            data=json.dumps({"model": "llama-3"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["model"] == "llama-3"

    def test_load_exception(self, client):
        """Load exception."""
        server_mod.llm_manager.list_models = MagicMock(return_value=["llama-3"])
        server_mod.llm_manager.load_model = MagicMock(
            side_effect=RuntimeError("load failed"),
        )
        resp = client.put(
            "/llm/model",
            data=json.dumps({"model": "llama-3"}),
            content_type="application/json",
        )
        assert resp.status_code == 500
        assert "load failed" in resp.get_json()["error"]


def test_list_tones(client):
    """List tones."""
    resp = client.get("/llm/tones")
    assert resp.status_code == 200
    tones = resp.get_json()["tones"]
    assert "diplomatic" in tones
    assert "professional" in tones
    assert "summarize" in tones


def test_get_metrics(client):
    """Get metrics."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "transcription_count" in data
    assert "uptime_seconds" in data


class TestPunctuateRoute:
    """Tests covering punctuate route."""

    def test_missing_text(self, client):
        """Missing text."""
        resp = client.post(
            "/punctuate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_text(self, client):
        """Empty text."""
        resp = client.post(
            "/punctuate",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_success(self, client):
        """Success."""
        server_mod.llm_manager.punctuate = MagicMock(return_value="Hello, world.")
        resp = client.post(
            "/punctuate",
            data=json.dumps({"text": "hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["text"] == "Hello, world."

    def test_exception(self, client):
        """Exception."""
        server_mod.llm_manager.punctuate = MagicMock(
            side_effect=RuntimeError("inference crash"),
        )
        resp = client.post(
            "/punctuate",
            data=json.dumps({"text": "hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 500
        assert "inference crash" in resp.get_json()["error"]


class TestTranslateRoute:
    """Tests covering translate route."""

    def test_missing_text(self, client):
        """Missing text."""
        resp = client.post(
            "/translate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_text(self, client):
        """Empty text."""
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "   ", "target_language": "Spanish"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"]

    def test_missing_target_language(self, client):
        """Missing target language."""
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "target_language" in resp.get_json()["error"]

    def test_success(self, client):
        """Success."""
        server_mod.llm_manager.translate = MagicMock(return_value="Hola mundo")
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "hello world", "target_language": "Spanish"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["text"] == "Hola mundo"
        assert data["source"] == "hello world"
        assert data["target_language"] == "Spanish"

    def test_exception(self, client):
        """Exception."""
        server_mod.llm_manager.translate = MagicMock(
            side_effect=RuntimeError("translation error"),
        )
        resp = client.post(
            "/translate",
            data=json.dumps({"text": "hello", "target_language": "Spanish"}),
            content_type="application/json",
        )
        assert resp.status_code == 500
        assert "translation error" in resp.get_json()["error"]


class TestTranscribeTimestampsRoute:
    """Tests covering transcribe timestamps route."""

    def test_success(self, client, wav_bytes):
        """Success."""
        mock_pipeline = MagicMock()
        mock_result = make_mock_result("hello world")
        # Simulate chunks
        chunk1 = MagicMock()
        chunk1.text = "hello"
        chunk1.start_ts = 0.0
        chunk1.end_ts = 0.5
        chunk2 = MagicMock()
        chunk2.text = " world"
        chunk2.start_ts = 0.5
        chunk2.end_ts = 1.0
        mock_result.chunks = [chunk1, chunk2]
        mock_pipeline.generate.return_value = mock_result
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe/timestamps", data=wav_bytes)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["text"] == "hello world"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["text"] == "hello"
        assert data["chunks"][0]["start"] == 0.0
        assert data["chunks"][1]["end"] == 1.0
        assert "duration" in data
        assert "latency" in data

    def test_with_model_name(self, client, wav_bytes):
        """With model name."""
        mock_pipeline = MagicMock()
        mock_result = make_mock_result("test")
        mock_result.chunks = []
        mock_pipeline.generate.return_value = mock_result
        server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
        mock_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        resp = client.post("/transcribe/timestamps/whisper-large", data=wav_bytes)
        assert resp.status_code == 200
        server_mod.model_manager.load_model.assert_called_with("whisper-large")

    def test_exception(self, client, wav_bytes):
        """Exception."""
        server_mod.model_manager.load_model = MagicMock(
            side_effect=RuntimeError("crash"),
        )
        resp = client.post("/transcribe/timestamps", data=wav_bytes)
        assert resp.status_code == 500
        assert "crash" in resp.get_json()["error"]


class TestHistoryExportRoute:
    """Tests covering history export route."""

    def test_no_db(self, client, tmp_path):
        """No db."""
        with patch.dict(os.environ, {"WAYVOX_CONFIG_DIR": str(tmp_path)}, clear=False):
            resp = client.get("/history/export")
        assert resp.status_code == 404

    def test_json_format(self, client, tmp_path):
        """Json format."""
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)",
            )
            conn.execute(
                "INSERT INTO history (text, timestamp) VALUES (?, ?)",
                ("hello", 1700000000.0),
            )
            conn.execute(
                "INSERT INTO history (text, timestamp) VALUES (?, ?)",
                ("world", 1700000010.0),
            )

        with patch.dict(os.environ, {"WAYVOX_CONFIG_DIR": str(tmp_path)}, clear=False):
            resp = client.get("/history/export?format=json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert len(data["items"]) == 2
        # Items should be in chronological order (reversed from DESC)
        assert data["items"][0]["text"] == "hello"
        assert "datetime" in data["items"][0]

    def test_markdown_format(self, client, tmp_path):
        """Markdown format."""
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)",
            )
            conn.execute(
                "INSERT INTO history (text, timestamp) VALUES (?, ?)",
                ("test entry", 1700000000.0),
            )

        with patch.dict(os.environ, {"WAYVOX_CONFIG_DIR": str(tmp_path)}, clear=False):
            resp = client.get("/history/export?format=markdown")
        assert resp.status_code == 200
        assert "text/markdown" in resp.content_type
        body = resp.get_data(as_text=True)
        assert "Transcription History" in body
        assert "test entry" in body

    def test_srt_format(self, client, tmp_path):
        """Srt format."""
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)",
            )
            conn.execute(
                "INSERT INTO history (text, timestamp) VALUES (?, ?)",
                ("subtitle line", 1700000000.0),
            )

        with patch.dict(os.environ, {"WAYVOX_CONFIG_DIR": str(tmp_path)}, clear=False):
            resp = client.get("/history/export?format=srt")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "1\n" in body
        assert "-->" in body
        assert "subtitle line" in body

    def test_unknown_format(self, client, tmp_path):
        """Unknown format."""
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)",
            )

        with patch.dict(os.environ, {"WAYVOX_CONFIG_DIR": str(tmp_path)}, clear=False):
            resp = client.get("/history/export?format=xml")
        assert resp.status_code == 400
        assert "Unknown format" in resp.get_json()["error"]

    def test_limit_parameter(self, client, tmp_path):
        """Limit parameter."""
        db_path = str(tmp_path / "history.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE history (id INTEGER PRIMARY KEY, text TEXT, timestamp REAL)",
            )
            for i in range(10):
                conn.execute(
                    "INSERT INTO history (text, timestamp) VALUES (?, ?)",
                    (f"entry {i}", 1700000000.0 + i),
                )

        with patch.dict(os.environ, {"WAYVOX_CONFIG_DIR": str(tmp_path)}, clear=False):
            resp = client.get("/history/export?format=json&limit=3")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3


# ===================================================================
#  Edge cases and integration-style tests
# ===================================================================


def test_transcribe_with_model_no_audio(client):
    """Transcribe with model no audio."""
    mock_pipeline = MagicMock()
    server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
    resp = client.post("/transcribe/whisper-small-en", data=b"")
    assert resp.status_code == 400


class TestRewriteToneResolution:
    """Verify that tone resolution picks custom_tones over DEFAULT_TONES."""

    def test_custom_tone_overrides_default(self, client):
        """Custom tone overrides default."""
        calls = []

        def capture_rewrite(_text, tone_name, tone_prompt):
            """Capture rewrite."""
            calls.append((tone_name, tone_prompt))
            return "rewritten"

        server_mod.llm_manager.rewrite = capture_rewrite
        resp = client.post(
            "/rewrite",
            data=json.dumps(
                {
                    "text": "hello",
                    "tones": ["diplomatic"],
                    "custom_tones": {"diplomatic": "My custom prompt"},
                },
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert calls[0][1] == "My custom prompt"

    def test_unknown_tone_ignored(self, client):
        """A tone not in DEFAULT_TONES and not in custom_tones is skipped."""
        server_mod.llm_manager.rewrite = MagicMock(return_value="done")
        resp = client.post(
            "/rewrite",
            data=json.dumps(
                {
                    "text": "hello",
                    "tones": ["nonexistent_tone"],
                },
            ),
            content_type="application/json",
        )
        assert resp.status_code == 200
        variants = resp.get_json()["variants"]
        # Only "original" — the unknown tone is not in tone_prompts
        assert len(variants) == 1
        assert variants[0]["tone"] == "original"


def test_stream_load_model_failure(client, wav_bytes):
    """Stream load model failure."""
    server_mod.model_manager.load_model = MagicMock(
        side_effect=RuntimeError("NPU unavailable"),
    )
    resp = client.post("/transcribe/stream", data=wav_bytes)
    assert resp.status_code == 500
    assert "NPU unavailable" in resp.get_json()["error"]


def test_no_audio_data(client):
    """No audio data."""
    mock_pipeline = MagicMock()
    server_mod.model_manager.load_model = MagicMock(return_value=mock_pipeline)
    resp = client.post("/transcribe/timestamps", data=b"")
    assert resp.status_code == 400
    assert "No audio" in resp.get_json()["error"]


def _fw_segment(text, start=0.0, end=1.0):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


class TestDetectEngineAndDevice:
    """Device selection must be deterministic without an NPU or GPU."""

    def test_explicit_cuda(self, monkeypatch):
        """Explicit cuda."""
        monkeypatch.setenv("WHISPER_DEVICE", "CUDA:1")
        assert server_mod.detect_engine_and_device() == ("faster-whisper", "cuda:1")

    def test_explicit_openvino_devices(self, monkeypatch):
        """Explicit openvino devices."""
        for raw, device in (("gpu", "GPU"), ("CPU", "CPU"), ("npu", "NPU")):
            monkeypatch.setenv("WHISPER_DEVICE", raw)
            assert server_mod.detect_engine_and_device() == ("openvino", device)

    def test_unknown_device_falls_back(self, monkeypatch):
        """Unknown device falls back."""
        monkeypatch.setenv("WHISPER_DEVICE", "tpu")
        with (
            patch.object(server_mod, "openvino", None),
            patch.object(server_mod, "ctranslate2", None),
        ):
            engine, device = server_mod.detect_engine_and_device()
        assert (engine, device) == ("faster-whisper", "cpu")

    def test_auto_prefers_npu(self, monkeypatch):
        """Auto prefers npu."""
        monkeypatch.setenv("WHISPER_DEVICE", "auto")
        ov = MagicMock()
        ov.Core.return_value.available_devices = ["CPU", "GPU", "NPU"]
        with patch.object(server_mod, "openvino", ov):
            assert server_mod.detect_engine_and_device() == ("openvino", "NPU")

    def test_auto_cuda_when_no_npu(self, monkeypatch):
        """Auto cuda when no npu."""
        monkeypatch.setenv("WHISPER_DEVICE", "auto")
        ov = MagicMock()
        ov.Core.return_value.available_devices = ["CPU"]
        ct2 = MagicMock()
        ct2.get_cuda_device_count.return_value = 1
        with (
            patch.object(server_mod, "openvino", ov),
            patch.object(server_mod, "ctranslate2", ct2),
        ):
            assert server_mod.detect_engine_and_device() == ("faster-whisper", "cuda")

    def test_auto_discrete_gpu(self, monkeypatch):
        """Auto discrete gpu."""
        monkeypatch.setenv("WHISPER_DEVICE", "auto")
        ov = MagicMock()
        ov.Core.return_value.available_devices = ["CPU", "GPU", "GPU.1"]
        ct2 = MagicMock()
        ct2.get_cuda_device_count.return_value = 0
        with (
            patch.object(server_mod, "openvino", ov),
            patch.object(server_mod, "ctranslate2", ct2),
        ):
            assert server_mod.detect_engine_and_device() == ("openvino", "GPU.1")

    def test_auto_integrated_gpu(self, monkeypatch):
        """Auto integrated gpu."""
        monkeypatch.setenv("WHISPER_DEVICE", "auto")
        ov = MagicMock()
        ov.Core.return_value.available_devices = ["CPU", "GPU"]
        ct2 = MagicMock()
        ct2.get_cuda_device_count.return_value = 0
        with (
            patch.object(server_mod, "openvino", ov),
            patch.object(server_mod, "ctranslate2", ct2),
        ):
            assert server_mod.detect_engine_and_device() == ("openvino", "GPU")

    def test_auto_openvino_cpu(self, monkeypatch):
        """Auto openvino cpu."""
        monkeypatch.setenv("WHISPER_DEVICE", "auto")
        ov = MagicMock()
        ov.Core.return_value.available_devices = ["CPU"]
        ct2 = MagicMock()
        ct2.get_cuda_device_count.return_value = 0
        with (
            patch.object(server_mod, "openvino", ov),
            patch.object(server_mod, "ctranslate2", ct2),
        ):
            assert server_mod.detect_engine_and_device() == ("openvino", "CPU")

    def test_auto_cpu_faster_whisper_when_nothing_installed(self, monkeypatch):
        """Auto cpu faster whisper when nothing installed."""
        monkeypatch.setenv("WHISPER_DEVICE", "auto")
        with (
            patch.object(server_mod, "openvino", None),
            patch.object(server_mod, "ctranslate2", None),
        ):
            assert server_mod.detect_engine_and_device() == ("faster-whisper", "cpu")


class TestFasterWhisperEngine:
    """faster-whisper path, mocked so CI does not need CUDA or a model download."""

    def _use_faster_whisper(self, device="cpu"):
        server_mod.ENGINE = "faster-whisper"
        server_mod.DEVICE = device
        whisper = MagicMock(name="WhisperModel")
        server_mod.ENGINES.faster_whisper_cls = whisper
        return whisper

    def test_load_local_cpu_model(self, model_mgr):
        """Load local cpu model."""
        whisper = self._use_faster_whisper("cpu")
        local = os.path.join(model_mgr.ct2_models_dir, "tiny.en")
        os.makedirs(local)
        pipe = model_mgr.load_model("tiny.en")
        assert pipe is whisper.return_value
        whisper.assert_called_once_with(
            local,
            device="cpu",
            device_index=0,
            compute_type="int8",
            num_workers=1,
            cpu_threads=1,
        )

    def test_load_remote_name_when_not_cached(self, model_mgr):
        """Load remote name when not cached."""
        whisper = self._use_faster_whisper("cpu")
        model_mgr.load_model("tiny.en")
        assert whisper.call_args.args[0] == "tiny.en"

    def test_load_cuda_picks_supported_compute_type(self, model_mgr):
        """Load cuda picks supported compute type."""
        whisper = self._use_faster_whisper("cuda:1")
        ct2 = MagicMock()
        ct2.get_supported_compute_types.return_value = [
            "int8",
            "float32",
            "int8_float16",
        ]
        with patch.object(server_mod, "ctranslate2", ct2):
            model_mgr.load_model("tiny.en")
        kwargs = whisper.call_args.kwargs
        assert kwargs["device"] == "cuda"
        assert kwargs["device_index"] == 1
        assert kwargs["compute_type"] == "int8_float16"

    def test_load_cuda_falls_back_when_probe_fails(self, model_mgr):
        """Load cuda falls back when probe fails."""
        whisper = self._use_faster_whisper("cuda")
        ct2 = MagicMock()
        ct2.get_supported_compute_types.side_effect = RuntimeError("no cuda")
        with patch.object(server_mod, "ctranslate2", ct2):
            model_mgr.load_model("tiny.en")
        assert whisper.call_args.kwargs["compute_type"] == "float32"

    def test_load_cuda_falls_back_when_no_preferred_type(self, model_mgr):
        """Load cuda falls back when no preferred type."""
        whisper = self._use_faster_whisper("cuda")
        ct2 = MagicMock()
        ct2.get_supported_compute_types.return_value = ["bfloat16"]
        with patch.object(server_mod, "ctranslate2", ct2):
            model_mgr.load_model("tiny.en")
        assert whisper.call_args.kwargs["compute_type"] == "float32"

    def test_transcribe(self, model_mgr):
        """Transcribe."""
        self._use_faster_whisper()
        pipe = MagicMock()
        info = MagicMock(language="en", language_probability=0.9123)
        pipe.transcribe.return_value = (
            [_fw_segment("hello"), _fw_segment(" world")],
            info,
        )
        model_mgr.pipelines["tiny.en"] = pipe
        text, perf, _elapsed = model_mgr.transcribe(
            "tiny.en",
            np.zeros(16),
            language="en",
        )
        assert text == "hello world"
        assert perf == {"language": "en", "language_probability": 0.912}
        assert pipe.transcribe.call_args.kwargs["language"] == "en"
        assert pipe.transcribe.call_args.kwargs["beam_size"] == 5

    def test_transcribe_stream(self, model_mgr):
        """Transcribe stream."""
        self._use_faster_whisper()
        pipe = MagicMock()
        pipe.transcribe.return_value = (
            [_fw_segment("a"), _fw_segment("b")],
            MagicMock(),
        )
        model_mgr.pipelines["tiny.en"] = pipe
        chunks = list(model_mgr.transcribe_stream("tiny.en", np.zeros(16)))
        assert chunks == ["a", "b"]
        assert pipe.transcribe.call_args.kwargs["language"] is None

    def test_transcribe_timestamps(self, model_mgr):
        """Transcribe timestamps."""
        self._use_faster_whisper()
        pipe = MagicMock()
        pipe.transcribe.return_value = (
            [_fw_segment("hello", 0.0, 0.25), _fw_segment(" world", 0.25, 0.8)],
            MagicMock(),
        )
        model_mgr.pipelines["tiny.en"] = pipe
        text, chunks, perf = model_mgr.transcribe_timestamps(
            "tiny.en",
            np.zeros(16),
            language="de",
        )
        assert text == "hello world"
        assert chunks == [
            {"text": "hello", "start": 0.0, "end": 0.25},
            {"text": " world", "start": 0.25, "end": 0.8},
        ]
        assert perf == {}
        assert pipe.transcribe.call_args.kwargs["word_timestamps"] is False
        assert pipe.transcribe.call_args.kwargs["language"] == "de"

    def test_list_models_inserts_default_and_hides_dotfiles(self, model_mgr):
        """List models inserts default and hides dotfiles."""
        server_mod.ENGINE = "faster-whisper"
        os.makedirs(os.path.join(model_mgr.ct2_models_dir, "tiny.en"))
        os.makedirs(os.path.join(model_mgr.ct2_models_dir, ".cache"))
        models = model_mgr.list_models()
        assert models[0] == server_mod.CT2_MODEL_NAME
        assert "tiny.en" in models
        assert all(not name.startswith(".") for name in models)

    def test_list_models_does_not_duplicate_default(self, model_mgr):
        """List models does not duplicate default."""
        server_mod.ENGINE = "faster-whisper"
        os.makedirs(os.path.join(model_mgr.ct2_models_dir, server_mod.CT2_MODEL_NAME))
        models = model_mgr.list_models()
        assert models.count(server_mod.CT2_MODEL_NAME) == 1

    def test_list_models_missing_dir_still_offers_default(self, model_mgr):
        """List models missing dir still offers default."""
        server_mod.ENGINE = "faster-whisper"
        model_mgr.ct2_models_dir = os.path.join(model_mgr.ct2_models_dir, "missing")
        assert model_mgr.list_models() == [server_mod.CT2_MODEL_NAME]


class TestOpenVinoBranches:
    """Tests covering open vino branches."""

    def test_list_models_missing_directory(self, model_mgr):
        """List models missing directory."""
        model_mgr.models_dir = os.path.join(model_mgr.models_dir, "missing")
        assert model_mgr.list_models() == []

    def test_timestamps_passes_language(self, model_mgr):
        """Timestamps passes language."""
        pipe = MagicMock()
        result = make_mock_result("bonjour")
        result.chunks = []
        pipe.generate.return_value = result
        model_mgr.pipelines["m"] = pipe
        text, chunks, _perf = model_mgr.transcribe_timestamps(
            "m",
            np.zeros(8),
            language="fr",
        )
        assert text == "bonjour"
        assert chunks == []
        assert pipe.generate.call_args.kwargs["language"] == "<|fr|>"
        assert pipe.generate.call_args.kwargs["return_timestamps"] is True

    def test_llm_requires_openvino_genai(self, llm_mgr):
        """Llm requires openvino genai."""
        server_mod.ENGINES.openvino_genai_for_llm = None
        with pytest.raises(RuntimeError, match="OpenVINO GenAI"):
            llm_mgr.load_model("llama-3")
