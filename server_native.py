"""Flask server that transcribes speech with OpenVINO GenAI or faster-whisper."""

import datetime
import io
import json
import logging
import os
import queue
import sqlite3
import threading
import time

import librosa
from flask import Flask, Response, jsonify, request

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None

try:
    from faster_whisper import WhisperModel as faster_whisper_cls

    _FASTER_WHISPER_IMPORT_ERROR = None
except ImportError as import_error:
    faster_whisper_cls = None
    _FASTER_WHISPER_IMPORT_ERROR = import_error

try:
    import openvino
except ImportError:
    openvino = None

try:
    import openvino_genai

    _OPENVINO_GENAI_IMPORT_ERROR = None
except ImportError as import_error:
    openvino_genai = None
    _OPENVINO_GENAI_IMPORT_ERROR = import_error

import diagnostics
import server_security
import vocab
from wayvox_paths import history_file


class EngineBindings:
    """Optional inference libraries. Tests replace attributes on ENGINES."""

    def __init__(self):
        self.openvino_genai = None
        self.openvino_genai_for_llm = None
        self.faster_whisper_cls = None

    def bind_openvino_genai(self, module):
        """Store the OpenVINO GenAI module for whisper and LLM calls."""
        self.openvino_genai = module
        self.openvino_genai_for_llm = module

    def bind_faster_whisper(self, whisper_cls):
        """Store the faster-whisper model class used to load CT2 weights."""
        self.faster_whisper_cls = whisper_cls


ENGINES = EngineBindings()
ENGINES.bind_openvino_genai(openvino_genai)
ENGINES.bind_faster_whisper(faster_whisper_cls)

SERVER_START_TIME = time.time()

app = server_security.install_csrf_guard(Flask(__name__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("WHISPER_MODEL", "whisper-small.en-fp16-ov")
CT2_MODEL_NAME = os.environ.get("WHISPER_CT2_MODEL", "small.en")
LLM_MODEL = os.environ.get("WHISPER_LLM_MODEL", "")

_SUMMARIZE_TASK = "Summarize the following text in one concise sentence."
_SUMMARIZE_ONLY = "Return only the summary, nothing else."

DEFAULT_TONES = {
    "diplomatic": (
        "Rewrite the following text to sound warmer, more considerate, "
        "and diplomatically phrased. Preserve the original meaning exactly. "
        "Return only the rewritten text, nothing else."
    ),
    "professional": (
        "Rewrite the following text in a formal, professional business tone. "
        "Preserve the original meaning exactly. "
        "Return only the rewritten text, nothing else."
    ),
    "summarize": _SUMMARIZE_TASK + " " + _SUMMARIZE_ONLY,
}


def _model_dir(base_dir, model_name):
    """Return base_dir/model_name, refusing names that escape base_dir."""
    base = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(base, model_name))
    if not path.startswith(base + os.sep):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return path


def detect_engine_and_device():
    """Return the engine and device, preferring NPU, CUDA, Intel GPU, then CPU."""
    raw = os.environ.get("WHISPER_DEVICE", "auto").strip()
    if raw.lower().startswith("cuda"):
        return "faster-whisper", raw.lower()
    if raw.upper() in ("NPU", "GPU", "CPU"):
        return "openvino", raw.upper()
    if raw.upper() != "AUTO":
        warned = "Unknown WHISPER_DEVICE '%s', falling back to auto-detect"
        logger.warning(warned, raw)

    if openvino is not None and "NPU" in openvino.Core().available_devices:
        logger.info("Auto-detected device: NPU (OpenVINO)")
        return "openvino", "NPU"
    cuda_count = 0
    if ctranslate2 is not None:
        cuda_count = ctranslate2.get_cuda_device_count()
    if cuda_count > 0:
        logger.info("Auto-detected device: CUDA (faster-whisper)")
        return "faster-whisper", "cuda"

    chosen = None
    if openvino is not None:
        devices = openvino.Core().available_devices
        gpu_names = [device for device in devices if device.startswith("GPU")]
        # Higher-indexed GPU is typically discrete (ARC) vs integrated (iGPU).
        best_gpus = sorted(gpu_names, reverse=True)
        if best_gpus:
            best_gpu = best_gpus[0]
            label = "Intel ARC (discrete)" if "." in best_gpu else "Intel GPU"
            found = "Auto-detected device: %s (%s, OpenVINO)"
            logger.info(found, best_gpu, label)
            chosen = ("openvino", best_gpu)
    if chosen is None and openvino is not None:
        logger.info("Auto-detected device: CPU (OpenVINO)")
        chosen = ("openvino", "CPU")
    if chosen is None:
        logger.info("Auto-detected device: CPU (faster-whisper)")
        chosen = ("faster-whisper", "cpu")
    return chosen


ENGINE, DEVICE = detect_engine_and_device()
logger.info("Engine: %s  Device: %s", ENGINE, DEVICE)
if ENGINE == "openvino":
    vocab.configure_backend(DEVICE)

LLM_DEVICE = os.environ.get(
    "WHISPER_LLM_DEVICE",
    DEVICE if ENGINE == "openvino" else "CPU",
)


def _require_selected_engine():
    """Abort startup when the selected whisper backend is not installed."""
    if ENGINE == "openvino" and ENGINES.openvino_genai is None:
        missing = "OpenVINO engine selected but openvino-genai not installed"
        logger.error(missing)
        raise _OPENVINO_GENAI_IMPORT_ERROR
    if ENGINE == "openvino":
        version = getattr(ENGINES.openvino_genai, "__version__", "unknown")
        logger.info("OpenVINO GenAI %s", version)
    if ENGINE == "faster-whisper" and ENGINES.faster_whisper_cls is None:
        missing = "faster-whisper engine selected but faster-whisper not installed"
        logger.error(missing)
        raise _FASTER_WHISPER_IMPORT_ERROR


def _cuda_compute_type(cuda_device):
    """Return the compute type faster-whisper should use on this device."""
    if cuda_device != "cuda":
        return "int8"
    if ctranslate2 is None:
        return "float32"
    try:
        supported = ctranslate2.get_supported_compute_types("cuda")
    except (OSError, RuntimeError, ValueError, TypeError):
        return "float32"
    preference = ("float16", "int8_float16", "int8", "float32")
    for candidate in preference:
        if candidate in supported:
            return candidate
    return "float32"


def extract_perf_metrics(result):
    """Return OpenVINO timing fields, or an empty dict when they are missing."""
    try:
        perf = result.perf_metrics
        features = perf.get_features_extraction_duration().mean
        inference = perf.get_inference_duration().mean
        generate = perf.get_generate_duration().mean
        detoken = perf.get_detokenization_duration().mean
        throughput = perf.get_throughput().mean
        tokens = perf.get_num_generated_tokens()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return {}
    return {
        "features_extraction_ms": round(features, 2),
        "inference_ms": round(inference, 2),
        "generate_ms": round(generate, 2),
        "detokenization_ms": round(detoken, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "num_generated_tokens": tokens,
    }


class MetricsCollector:
    """Count transcriptions, errors, latency, and time spent in audio."""

    def __init__(self):
        self.transcription_count = 0
        self.transcription_errors = 0
        self.total_latency = 0.0
        self.total_audio_duration = 0.0
        self.model_load_times = {}
        self.last_request = {}
        self._lock = threading.Lock()

    def record_transcription(self, latency, audio_duration, perf_metrics=None):
        """Add one successful transcription to the running totals."""
        with self._lock:
            self.transcription_count += 1
            self.total_latency += latency
            self.total_audio_duration += audio_duration
            if perf_metrics:
                self.last_request = perf_metrics

    def record_error(self):
        """Add one failed transcription to the error total."""
        with self._lock:
            self.transcription_errors += 1

    def record_model_load(self, model_name, load_time):
        """Remember how long a model took to load."""
        with self._lock:
            self.model_load_times[model_name] = round(load_time, 2)

    def snapshot(self):
        """Return a JSON-ready copy of the counters and the latest timings."""
        with self._lock:
            total = self.transcription_count + self.transcription_errors
            if self.transcription_count:
                latency_total = self.total_latency
                heard = self.transcription_count
                average_latency = round(latency_total / heard, 3)
            else:
                average_latency = 0.0
            if total:
                error_rate = round(self.transcription_errors / total, 4)
            else:
                error_rate = 0.0
            data = {
                "transcription_count": self.transcription_count,
                "transcription_errors": self.transcription_errors,
                "average_latency_seconds": average_latency,
                "error_rate": error_rate,
                "total_audio_seconds": round(self.total_audio_duration, 1),
                "model_load_times": dict(self.model_load_times),
                "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
                "engine": ENGINE,
                "device": DEVICE,
            }
            if self.last_request:
                data["last_request"] = dict(self.last_request)
            return data


metrics = MetricsCollector()


class ModelManager:
    """Load whisper models and run batch, stream, and timestamp transcription."""

    def __init__(self):
        self.models_dir = os.path.expanduser("~/.whisper/models")
        self.ct2_models_dir = os.path.expanduser("~/.whisper/ct2-models")
        self.pipelines = {}
        self.default_model = MODEL_NAME if ENGINE == "openvino" else CT2_MODEL_NAME

    def _load_openvino(self, model_name):
        """Build an OpenVINO WhisperPipeline for one installed model."""
        assert ENGINES.openvino_genai is not None
        model_path = _model_dir(self.models_dir, model_name)
        if not os.path.exists(model_path):
            missing = f"Model {model_name} not found at {model_path}"
            raise FileNotFoundError(missing)
        logger.info("Loading %s on %s (OpenVINO)", model_name, DEVICE)
        # Stateful on NPU so GenAI 2025.4 does not stay on the static
        # pipeline. word_timestamps=True (WHISPER_NPU_WORD_TIMESTAMPS=1)
        # compiles the 447 token prompt slot. CACHE_DIR keeps that blob.
        return vocab.load_whisper_pipeline(
            ENGINES.openvino_genai.WhisperPipeline,
            model_path,
            DEVICE,
        )

    def _load_faster_whisper(self, model_name):
        """Build a faster-whisper model for the selected device."""
        assert ENGINES.faster_whisper_cls is not None
        cuda_index = 0
        if ":" in DEVICE:
            cuda_index = int(DEVICE.split(":")[1])
        cuda_device = DEVICE.split(":")[0]
        try:
            local_path = _model_dir(self.ct2_models_dir, model_name)
        except ValueError:
            local_path = ""  # not a local directory name, e.g. a Hugging Face repo id
        if local_path and os.path.isdir(local_path):
            source = local_path
        else:
            source = model_name
        compute_type = _cuda_compute_type(cuda_device)
        loaded = "Loading %s on %s (faster-whisper, compute=%s)"
        logger.info(loaded, model_name, DEVICE, compute_type)
        return ENGINES.faster_whisper_cls(
            source,
            device=cuda_device,
            device_index=cuda_index,
            compute_type=compute_type,
            num_workers=1,
            cpu_threads=1,
        )

    def load_model(self, model_name):
        """Load a whisper model into the cache and return it."""
        if model_name in self.pipelines:
            return self.pipelines[model_name]
        started = time.time()
        if ENGINE == "openvino":
            pipeline = self._load_openvino(model_name)
        else:
            pipeline = self._load_faster_whisper(model_name)
        self.pipelines[model_name] = pipeline
        elapsed = time.time() - started
        logger.info("Model loaded in %.1fs", elapsed)
        metrics.record_model_load(model_name, elapsed)
        return pipeline

    def transcribe(self, model_name, audio_np, language=None):
        """Return recognized text, performance metrics, and elapsed seconds."""
        pipeline = self.load_model(model_name)
        started = time.time()
        if ENGINE == "openvino":
            gen_kwargs: dict = {}
            if language:
                gen_kwargs["language"] = f"<|{language}|>"
            result = vocab.generate_biased(
                pipeline.generate,
                audio_np,
                **gen_kwargs,
            )
            elapsed = time.time() - started
            perf = extract_perf_metrics(result)
            return str(result), perf, elapsed
        segments, info = vocab.generate_biased(
            pipeline.transcribe,
            audio_np,
            language=language or None,
            beam_size=5,
        )
        text = "".join(segment.text for segment in segments)
        elapsed = time.time() - started
        perf = {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }
        return text, perf, elapsed

    def transcribe_stream(self, model_name, audio_np, language=None):
        """Yield recognized text chunks as the model produces them."""
        pipeline = self.load_model(model_name)
        if ENGINE == "openvino":
            chunks: queue.Queue = queue.Queue()

            def cb(chunk):
                chunks.put(chunk)
                return 0

            def run():
                try:
                    gen_kwargs: dict = {
                        "streamer": cb,
                        "return_timestamps": False,
                    }
                    if language:
                        gen_kwargs["language"] = f"<|{language}|>"
                    vocab.generate_biased(pipeline.generate, audio_np, **gen_kwargs)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    logger.error("Streaming error: %s", exc)
                finally:
                    chunks.put(None)

            threading.Thread(target=run, daemon=True).start()
            while True:
                chunk = chunks.get()
                if chunk is None:
                    break
                yield chunk
            return
        segments, _ = vocab.generate_biased(
            pipeline.transcribe,
            audio_np,
            language=language or None,
            beam_size=5,
        )
        for segment in segments:
            yield segment.text

    def transcribe_timestamps(self, model_name, audio_np, language=None):
        """Return full text, timed chunks, and performance metrics."""
        pipeline = self.load_model(model_name)
        if ENGINE == "openvino":
            gen_kwargs: dict = {"return_timestamps": True}
            if language:
                gen_kwargs["language"] = f"<|{language}|>"
            result = vocab.generate_biased(
                pipeline.generate,
                audio_np,
                **gen_kwargs,
            )
            chunks = [
                {
                    "text": chunk.text,
                    "start": round(chunk.start_ts, 3),
                    "end": round(chunk.end_ts, 3),
                }
                for chunk in result.chunks
            ]
            perf = extract_perf_metrics(result)
            return str(result), chunks, perf
        segments, _ = vocab.generate_biased(
            pipeline.transcribe,
            audio_np,
            language=language or None,
            word_timestamps=False,
            beam_size=5,
        )
        segs = list(segments)
        text = "".join(segment.text for segment in segs)
        chunks = []
        for segment in segs:
            chunks.append(
                {
                    "text": segment.text,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                },
            )
        return text, chunks, {}

    def list_models(self):
        """Return installed model names for the active whisper engine."""
        if ENGINE == "openvino":
            if not os.path.isdir(self.models_dir):
                return []
            names = []
            for name in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, name)
                if os.path.isdir(model_path) and not name.startswith("."):
                    names.append(name)
            return names
        models = []
        if os.path.isdir(self.ct2_models_dir):
            for name in os.listdir(self.ct2_models_dir):
                if not name.startswith("."):
                    models.append(name)
        if CT2_MODEL_NAME not in models:
            models.insert(0, CT2_MODEL_NAME)
        return models


class LLMManager:
    """Load an OpenVINO LLM and rewrite, punctuate, or translate text."""

    def __init__(self):
        self.models_dir = os.path.expanduser("~/.whisper/llm-models")
        self.pipeline = None
        self.current_model = LLM_MODEL
        if not self.current_model:
            models = self.list_models()
            if models:
                self.current_model = models[0]
                logger.info("Auto-selected LLM: %s", self.current_model)

    def _require_openvino(self):
        """Raise when OpenVINO GenAI is not available for LLM calls."""
        if ENGINES.openvino_genai_for_llm is None:
            lead = "LLM features require OpenVINO GenAI"
            detail = "(not installed on this device path)"
            raise RuntimeError(f"{lead} {detail}")

    def load_model(self, model_name=None):
        """Load an LLM pipeline and return it."""
        self._require_openvino()
        assert ENGINES.openvino_genai_for_llm is not None
        model_name = model_name or self.current_model
        if not model_name:
            raise ValueError("No LLM model configured")
        model_path = _model_dir(self.models_dir, model_name)
        if not os.path.exists(model_path):
            missing = f"LLM model {model_name} not found at {model_path}"
            raise FileNotFoundError(missing)
        if self.current_model != model_name or self.pipeline is None:
            logger.info("Loading LLM: %s on %s", model_name, LLM_DEVICE)
            started = time.time()
            self.pipeline = ENGINES.openvino_genai_for_llm.LLMPipeline(
                str(model_path),
                device=LLM_DEVICE,
            )
            self.current_model = model_name
            logger.info("LLM loaded in %.1fs", time.time() - started)
        return self.pipeline

    def list_models(self):
        """Return installed LLM directory names."""
        if not os.path.isdir(self.models_dir):
            return []
        names = []
        for name in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, name)
            if os.path.isdir(model_path) and not name.startswith("."):
                names.append(name)
        return names

    def rewrite(self, text, tone_name, tone_prompt):
        """Rewrite text in one tone and return the model's text."""
        pipeline = self.load_model()
        prompt = f"{tone_prompt}\n\n{text}"
        started = time.time()
        result = pipeline.generate(prompt, max_new_tokens=512, temperature=0.3)
        logger.info("Rewrote (%s) in %.2fs", tone_name, time.time() - started)
        return str(result).strip()

    def punctuate(self, text):
        """Add punctuation and capitalization and return the model's text."""
        pipeline = self.load_model()
        lead = "Add proper punctuation and capitalization to this text."
        closing = "Return only the corrected text, nothing else."
        prompt = f"{lead} {closing}\n\n{text}"
        started = time.time()
        result = pipeline.generate(prompt, max_new_tokens=256, temperature=0.1)
        logger.info("Punctuated in %.2fs", time.time() - started)
        return str(result).strip()

    def translate(self, text, target_language):
        """Translate text into the target language and return the model's text."""
        pipeline = self.load_model()
        lead = f"Translate the following text to {target_language}."
        instruction = "Return only the translation, nothing else."
        prompt = f"{lead} {instruction}\n\n{text}"
        started = time.time()
        result = pipeline.generate(prompt, max_new_tokens=512, temperature=0.3)
        elapsed = time.time() - started
        logger.info("Translated to %s in %.2fs", target_language, elapsed)
        return str(result).strip()


def _load_audio(req):
    """Decode the request body as 16 kHz mono audio samples."""
    audio_data = req.get_data()
    if not audio_data:
        raise ValueError("No audio data")
    audio_np, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
    return audio_np


def _runtime_interface():
    """Describe the active whisper backend and device for the system report."""
    if ENGINE == "openvino":
        return f"OpenVINO GenAI WhisperPipeline, device plugin {DEVICE}"
    if ENGINE == "faster-whisper":
        return f"faster-whisper through CTranslate2, device {DEVICE}"
    return ENGINE


def system_report():
    """Return process facts for the tray About dialog, including versions."""
    report = diagnostics.collect()
    report["source"] = "wayvox-server"
    report["runtime"] = {
        "engine": ENGINE,
        "device": DEVICE,
        "requested_device": os.environ.get("WHISPER_DEVICE", "auto"),
        "model": model_manager.default_model,
        "llm_model": LLM_MODEL,
        "llm_device": LLM_DEVICE,
        "interface": _runtime_interface(),
        "pid": os.getpid(),
        "bias": vocab.bias_status(),
        "vocab": vocab.describe(),
    }
    report["text"] = diagnostics.format_report(report)
    return report


@app.route("/health", methods=["GET"])
def health():
    """Report that the server is up and which engine is loaded."""
    return jsonify(
        {
            "status": "ok",
            "model": model_manager.default_model,
            "engine": ENGINE,
            "device": DEVICE,
            "bias": vocab.bias_status(),
        },
    )


@app.route("/system", methods=["GET"])
def system_info():
    """Return the diagnostics report for the tray About dialog."""
    return jsonify(system_report())


@app.route("/models", methods=["GET"])
def list_models():
    """Return the model names installed for the active engine."""
    return jsonify({"models": model_manager.list_models()})


@app.route("/model/default", methods=["GET"])
def get_default_model():
    """Return the model used when a request does not name one."""
    return jsonify({"model": model_manager.default_model})


@app.route("/model/default", methods=["PUT"])
def set_default_model():
    """Load the requested model and make it the default."""
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"error": "model name required"}), 400
    model_name = data["model"]
    available = model_manager.list_models()
    if model_name not in available:
        missing = f"model {model_name} not found"
        return jsonify({"error": missing}), 404
    model_manager.load_model(model_name)
    model_manager.default_model = model_name
    logger.info("Default model changed to: %s", model_name)
    return jsonify({"model": model_name})


@app.route("/transcribe/<model_name>", methods=["POST"])
def transcribe_with_model(model_name):
    """Transcribe an upload with one model and return JSON text."""
    try:
        audio_np = _load_audio(request)
        language = request.args.get("language")
        text, perf, elapsed = model_manager.transcribe(
            model_name,
            audio_np,
            language,
        )
        text = vocab.apply(text)
        duration = len(audio_np) / 16000
        template = "Transcribed %.1fs audio in %.2fs (%.1fx realtime)"
        logger.info(template, duration, elapsed, duration / elapsed)
        metrics.record_transcription(elapsed, duration, perf)
        return jsonify({"text": text})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, RuntimeError, TypeError, ZeroDivisionError) as exc:
        logger.error("Error: %s", exc)
        metrics.record_error()
        return jsonify({"error": str(exc)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe an upload with the default model."""
    return transcribe_with_model(model_manager.default_model)


@app.route("/transcribe/stream", methods=["POST"])
def transcribe_stream():
    """Stream a short upload using the default model."""
    return transcribe_stream_with_model(model_manager.default_model)


@app.route("/transcribe/stream/<model_name>", methods=["POST"])
def transcribe_stream_with_model(model_name=None):
    """Stream a short upload as server-sent recognition events."""
    model_name = model_name or model_manager.default_model
    try:
        audio_np = _load_audio(request)
        duration = len(audio_np) / 16000
        if duration > 30:
            return jsonify({"error": "Streaming requires audio < 30 seconds"}), 400
        language = request.args.get("language")
        # Load before the stream starts. Failures inside the generator cannot
        # be turned into a JSON error response once iteration has begun.
        model_manager.load_model(model_name)

        def generate():
            started = time.time()
            full_text = []
            for chunk in model_manager.transcribe_stream(
                model_name,
                audio_np,
                language,
            ):
                piece = chunk if isinstance(chunk, str) else str(chunk)
                full_text.append(piece)
                payload = json.dumps({"text": vocab.apply(piece)})
                yield f"data: {payload}\n\n"
            elapsed = time.time() - started
            template = "Streamed %.1fs audio in %.2fs (%.1fx realtime)"
            logger.info(template, duration, elapsed, duration / elapsed)
            metrics.record_transcription(elapsed, duration)
            corrected = vocab.apply("".join(full_text))
            done = json.dumps({"done": True, "full_text": corrected})
            yield f"data: {done}\n\n"

        return Response(generate(), mimetype="text/event-stream")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, RuntimeError, TypeError, ZeroDivisionError) as exc:
        logger.error("Error: %s", exc)
        metrics.record_error()
        return jsonify({"error": str(exc)}), 500


@app.route("/transcribe/timestamps/<model_name>", methods=["POST"])
def transcribe_timestamps_with_model(model_name):
    """Transcribe an upload and return timed text chunks."""
    try:
        audio_np = _load_audio(request)
        language = request.args.get("language")
        started = time.time()
        text, chunks, perf = model_manager.transcribe_timestamps(
            model_name,
            audio_np,
            language,
        )
        text = vocab.apply(text)
        for chunk in chunks:
            if isinstance(chunk.get("text"), str):
                chunk["text"] = vocab.apply(chunk["text"])
        elapsed = time.time() - started
        duration = len(audio_np) / 16000
        template = "Timestamped %.1fs audio in %.2fs (%s chunks)"
        logger.info(template, duration, elapsed, len(chunks))
        metrics.record_transcription(elapsed, duration, perf)
        return jsonify(
            {
                "text": text,
                "chunks": chunks,
                "duration": round(duration, 2),
                "latency": round(elapsed, 3),
            },
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, RuntimeError, TypeError) as exc:
        logger.error("Error: %s", exc)
        metrics.record_error()
        return jsonify({"error": str(exc)}), 500


@app.route("/transcribe/timestamps", methods=["POST"])
def transcribe_timestamps():
    """Transcribe an upload with the default model and return timestamps."""
    return transcribe_timestamps_with_model(model_manager.default_model)


@app.route("/rewrite", methods=["POST"])
def rewrite():
    """Rewrite text into the requested tones and return each variant."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    tones = data.get("tones", list(DEFAULT_TONES.keys()))
    custom_tones = data.get("custom_tones", {})
    tone_prompts = {}
    for tone in tones:
        if tone in custom_tones:
            tone_prompts[tone] = custom_tones[tone]
        elif tone in DEFAULT_TONES:
            tone_prompts[tone] = DEFAULT_TONES[tone]
    variants = [{"tone": "original", "text": text}]
    for tone_name, tone_prompt in tone_prompts.items():
        try:
            rewritten = llm_manager.rewrite(text, tone_name, tone_prompt)
            variants.append({"tone": tone_name, "text": rewritten})
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Rewrite (%s) failed: %s", tone_name, exc)
            variants.append({"tone": tone_name, "text": text, "error": str(exc)})
    return jsonify({"variants": variants})


@app.route("/llm/models", methods=["GET"])
def list_llm_models():
    """Return installed LLM directories and the model currently loaded."""
    return jsonify(
        {"models": llm_manager.list_models(), "current": llm_manager.current_model},
    )


@app.route("/llm/model", methods=["PUT"])
def set_llm_model():
    """Load an installed LLM and make it the current model."""
    data = request.get_json()
    if not data or "model" not in data:
        return jsonify({"error": "model name required"}), 400
    model_name = data["model"]
    available = llm_manager.list_models()
    if model_name not in available:
        missing = f"LLM model {model_name} not found"
        return jsonify({"error": missing}), 404
    try:
        llm_manager.load_model(model_name)
        return jsonify({"model": model_name})
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/llm/tones", methods=["GET"])
def list_tones():
    """Return the built-in rewrite tone names."""
    return jsonify({"tones": list(DEFAULT_TONES.keys())})


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return transcription counters for this server process."""
    return jsonify(metrics.snapshot())


@app.route("/punctuate", methods=["POST"])
def punctuate():
    """Add punctuation to text with the loaded LLM."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    try:
        result = llm_manager.punctuate(text)
        return jsonify({"text": result})
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Punctuate failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    """Translate text into the requested language with the loaded LLM."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    text = data["text"].strip()
    target_language = data.get("target_language", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    if not target_language:
        return jsonify({"error": "target_language required"}), 400
    try:
        result = llm_manager.translate(text, target_language)
        return jsonify(
            {"text": result, "source": text, "target_language": target_language},
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Translate failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


def _history_rows(limit):
    """Return history rows oldest-first, or None when the database is absent."""
    db_path = str(history_file())
    if not os.path.exists(db_path):
        return None
    query = "SELECT text, timestamp FROM history ORDER BY id DESC LIMIT ?"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    rows.reverse()
    return rows


def _local_time(stamp):
    """Return a timezone-aware local datetime for a Unix timestamp."""
    return datetime.datetime.fromtimestamp(stamp, tz=datetime.timezone.utc).astimezone()


def _history_json(rows):
    """Return history rows as JSON items with ISO timestamps."""
    items = []
    for text, stamp in rows:
        when = _local_time(stamp).isoformat()
        items.append({"text": text, "timestamp": stamp, "datetime": when})
    return jsonify({"items": items, "count": len(items)})


def _history_markdown(rows):
    """Return history rows as a markdown document."""
    lines = ["# Transcription History\n"]
    for text, stamp in rows:
        when = _local_time(stamp)
        heading = when.strftime("%H:%M — %Y-%m-%d")
        lines.append(f"## {heading}\n\n{text}\n")
    return Response("\n".join(lines), mimetype="text/markdown")


def _history_srt(rows):
    """Return history rows as an SRT subtitle file."""
    lines = []
    for index, (text, stamp) in enumerate(rows, 1):
        when = _local_time(stamp)
        start = when.strftime("%H:%M:%S,000")
        seconds = max(3, len(text) * 0.05)
        end_at = when + datetime.timedelta(seconds=seconds)
        end = end_at.strftime("%H:%M:%S,000")
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    return Response("\n".join(lines), mimetype="text/plain")


@app.route("/history/export", methods=["GET"])
def export_history():
    """Download recent transcription history as JSON, Markdown, or SRT."""
    fmt = request.args.get("format", "json")
    limit = request.args.get("limit", "50", type=int)
    rows = _history_rows(limit)
    if rows is None:
        return jsonify({"error": "No history database found"}), 404
    if fmt == "json":
        return _history_json(rows)
    if fmt == "markdown":
        return _history_markdown(rows)
    if fmt == "srt":
        return _history_srt(rows)
    message = f"Unknown format: {fmt}. Use json, markdown, or srt"
    return jsonify({"error": message}), 400


model_manager = ModelManager()
llm_manager = LLMManager()

if __name__ == "__main__":
    _require_selected_engine()
    _vocab_info = vocab.describe()
    logger.info(
        "Vocabulary: %d terms, %d replacements, prompt %d chars, bias %s (%s)",
        _vocab_info["terms"],
        _vocab_info["replacements"],
        _vocab_info["initial_prompt_chars"],
        _vocab_info["bias_mode"],
        _vocab_info["path"],
    )
    model_manager.load_model(model_manager.default_model)
    if LLM_MODEL:
        try:
            llm_manager.load_model()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("LLM model not loaded: %s", exc)
    app.run(host=server_security.bind_host(), port=5000)
