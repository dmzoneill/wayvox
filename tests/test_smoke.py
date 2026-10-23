"""Smoke tests — lightweight assembly checks that verify components load and
respond without requiring real models, hardware, or external services."""

import json
from unittest.mock import MagicMock, patch

try:
    from flask import Flask
except ImportError:  # server smoke tests report a missing Flask install
    Flask = None

import dictation_output
import push_to_talk

pytest_plugins = ["smoke_support"]


def _json_post(client, path, payload):
    """POST a JSON document and return the Flask response."""
    return client.post(
        path,
        data=json.dumps(payload),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# server_native.py smoke tests
# ---------------------------------------------------------------------------


class TestServerSmoke:
    """Verify the Flask app assembles and its routes respond."""

    def test_app_is_flask(self, server_smoke):
        """The assembled application is a Flask app."""

        server_mod, _client = server_smoke
        assert isinstance(server_mod.app, Flask)

    def test_health_returns_200(self, server_smoke):
        """Health returns 200."""

        _server_mod, client = server_smoke
        resp = client.get("/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"
        assert "model" in data

    def test_models_returns_200(self, server_smoke):
        """Models returns 200."""

        _server_mod, client = server_smoke
        with (
            patch("os.listdir", return_value=["whisper-small-en"]),
            patch(
                "os.path.isdir",
                return_value=True,
            ),
        ):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "models" in data

    def test_llm_tones_returns_200(self, server_smoke):
        """Llm tones returns 200."""

        _server_mod, client = server_smoke
        resp = client.get("/llm/tones")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "tones" in data
        assert isinstance(data["tones"], list)

    def test_llm_models_returns_200(self, server_smoke):
        """Llm models returns 200."""

        _server_mod, client = server_smoke
        resp = client.get("/llm/models")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "models" in data
        assert "current" in data

    def test_metrics_returns_200(self, server_smoke):
        """Metrics returns 200."""

        _server_mod, client = server_smoke
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "transcription_count" in data
        assert "uptime_seconds" in data

    def test_model_default_get(self, server_smoke):
        """Model default get."""

        _server_mod, client = server_smoke
        resp = client.get("/model/default")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "model" in data

    def test_all_expected_routes_registered(self, server_smoke):
        """All expected routes registered."""

        server_mod, _client = server_smoke
        rules = [r.rule for r in server_mod.app.url_map.iter_rules()]
        expected = [
            "/health",
            "/system",
            "/models",
            "/model/default",
            "/transcribe",
            "/transcribe/<model_name>",
            "/transcribe/stream",
            "/transcribe/stream/<model_name>",
            "/rewrite",
            "/llm/models",
            "/llm/model",
            "/llm/tones",
            "/metrics",
            "/punctuate",
            "/translate",
            "/transcribe/timestamps",
            "/transcribe/timestamps/<model_name>",
            "/history/export",
        ]
        for route in expected:
            assert route in rules, f"Route {route} not registered"

    def test_transcribe_rejects_empty_body(self, server_smoke):
        """Transcribe rejects empty body."""

        _server_mod, client = server_smoke
        resp = client.post("/transcribe", data=b"")
        assert resp.status_code == 400

    def test_rewrite_rejects_missing_text(self, server_smoke):
        """Rewrite rejects missing text."""

        _server_mod, client = server_smoke
        resp = _json_post(client, "/rewrite", {})
        assert resp.status_code == 400

    def test_punctuate_rejects_empty_text(self, server_smoke):
        """Punctuate rejects empty text."""

        _server_mod, client = server_smoke
        resp = _json_post(client, "/punctuate", {"text": "  "})
        assert resp.status_code == 400

    def test_translate_requires_target_language(self, server_smoke):
        """Translate requires target language."""

        _server_mod, client = server_smoke
        resp = _json_post(client, "/translate", {"text": "hello"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# push_to_talk.py smoke tests
# ---------------------------------------------------------------------------


class TestPushToTalkSmoke:
    """Verify the push-to-talk module imports and key components work."""

    def test_module_imports(self):
        """Module imports."""
        assert hasattr(push_to_talk, "trim_silence")
        assert hasattr(push_to_talk, "format_dictation")
        assert hasattr(push_to_talk, "type_text")

    def test_constants_are_sane(self):
        """Constants are sane."""
        assert push_to_talk.SAMPLE_RATE == 16000
        assert push_to_talk.CHANNELS == 1
        assert push_to_talk.SAMPLE_WIDTH == 2
        assert len(push_to_talk.VOICE_COMMANDS) > 10
        assert len(push_to_talk.DBUS_BUS_NAMES) == 2

    def test_dbus_bus_names(self):
        """Dbus bus names."""
        assert "org.gnome.Shell" in push_to_talk.DBUS_BUS_NAMES
        assert "com.whisper.LanguageBuddy" in push_to_talk.DBUS_BUS_NAMES

    def test_trim_silence_empty(self):
        """Trim silence empty."""
        assert push_to_talk.trim_silence(b"") == b""

    def test_format_dictation_passthrough(self):
        """Format dictation passthrough."""
        assert push_to_talk.format_dictation("Hello world") == "Hello world"

    def test_format_dictation_capitalizes(self):
        """Format dictation capitalizes."""
        assert push_to_talk.format_dictation("hello")[0] == "H"

    def test_type_text_empty_is_noop(self):
        """Type text empty is noop."""
        with patch("subprocess.run") as m:
            push_to_talk.type_text("")
            m.assert_not_called()

    def test_try_dbus_handoff_no_bus(self):
        """Try dbus handoff no bus."""
        with (
            patch.object(dictation_output, "GLib", MagicMock()),
            patch.object(dictation_output, "try_dbus_call", return_value=None),
        ):
            assert push_to_talk.try_dbus_handoff("test") is False

    def test_notify_graceful_failure(self):
        """Notify graceful failure."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            push_to_talk.notify("test")

    def test_play_sound_missing_file(self):
        """Play sound missing file."""
        with patch("os.path.exists", return_value=False):
            push_to_talk.play_sound("nonexistent")

    def test_load_app_contexts_returns_defaults(self):
        """Load app contexts returns defaults."""
        with patch("os.path.exists", return_value=False):
            ctx = push_to_talk.load_app_contexts()
            assert isinstance(ctx, dict)
            assert len(ctx) > 0

    def test_voice_commands_have_actions(self):
        """Voice commands have actions."""
        for cmd, actions in push_to_talk.VOICE_COMMANDS.items():
            assert len(actions) > 0, f"Command '{cmd}' has no actions"
            for action_type, _ in actions:
                assert action_type in (
                    "key",
                    "type",
                ), f"Command '{cmd}' has invalid action type '{action_type}'"


# ---------------------------------------------------------------------------
# KDE service smoke tests
# ---------------------------------------------------------------------------


class TestKdeServiceSmoke:
    """Verify the KDE D-Bus service module loads and key functions work."""

    def test_module_imports(self, kde_service):
        """Module imports."""
        assert hasattr(kde_service, "load_config")
        assert hasattr(kde_service, "server_url")
        assert hasattr(kde_service, "http_post")
        assert hasattr(kde_service, "type_text")
        assert hasattr(kde_service, "backspace_n")

    def test_load_config_missing_file(self, kde_service):
        """Load config missing file."""
        result = kde_service.load_config()
        assert isinstance(result, dict)

    def test_server_url_default(self, kde_service):
        """Server url default."""
        url = kde_service.server_url(
            "/health",
            {"server-host": "localhost", "server-port": 8080},
        )
        assert url == "http://localhost:8080/health"

    def test_server_url_default_config(self, kde_service):
        """Server url default config."""
        url = kde_service.server_url("/test")
        assert url.startswith("http://")
        assert "/test" in url

    def test_default_tones_constant(self, kde_service):
        """Default tones constant."""
        assert kde_service.DEFAULT_TONES == ["diplomatic", "professional"]

    def test_config_path_exists(self, kde_service):
        """Config path exists."""
        assert isinstance(kde_service.CONFIG_PATH, str)
        assert "wayvox" in kde_service.CONFIG_PATH

    def test_type_text_calls_ydotool(self, kde_service):
        """Type text calls ydotool."""
        with patch("subprocess.run") as m:
            kde_service.type_text("hello")
            m.assert_called_once()
            args = m.call_args[0][0]
            assert args[0] == "ydotool"

    def test_backspace_n_zero(self, kde_service):
        """Backspace n zero."""
        with patch("subprocess.run") as m:
            kde_service.backspace_n(0)
            m.assert_not_called()

    def test_http_post_handles_error(self, kde_service):
        """Http post handles error."""
        result = kde_service.http_post("http://127.0.0.1:1/bad", {"x": 1})
        assert result is None
