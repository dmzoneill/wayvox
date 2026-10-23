"""D-Bus service tests for kde-plasmoid/wayvox_kde_service.py.

tests.kde_support loads that module by file path and provides qapp.
"""

import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from tests.kde_support import kde_svc

pytest_plugins = ["tests.kde_support"]


def _record_finished(worker):
    """Return a list that collects the worker's finished signal."""
    results = []

    def _store(tone, text):
        """Append one finished tone and text pair."""
        results.append((tone, text))

    worker.finished.connect(_store)
    return results


@pytest.mark.usefixtures("qapp")
class TestRewriteWorker:
    """Checks that RewriteWorker emits the matching variant or the original."""

    def test_successful_rewrite(self):
        """A matching variant is emitted in place of the original text."""
        config = {"server-host": "localhost", "server-port": 5000}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)
        results = _record_finished(worker)
        api_response = {"variants": [{"tone": "diplomatic", "text": "Greetings"}]}

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        assert results == [("diplomatic", "Greetings")]

    def test_failed_http(self):
        """A failed HTTP call emits the original text."""
        config = {"server-host": "localhost", "server-port": 5000}
        worker = kde_svc.RewriteWorker("hello", "diplomatic", config)
        results = _record_finished(worker)

        with patch.object(kde_svc, "http_post", return_value=None):
            worker.run()

        assert results == [("diplomatic", "hello")]

    def test_missing_variants_key(self):
        """A response without variants emits the original text."""
        worker = kde_svc.RewriteWorker("hello", "diplomatic", {})
        results = _record_finished(worker)

        with patch.object(kde_svc, "http_post", return_value={"status": "ok"}):
            worker.run()

        assert results == [("diplomatic", "hello")]

    def test_wrong_tone_in_variants(self):
        """A variant for a different tone emits the original text."""
        worker = kde_svc.RewriteWorker("hello", "diplomatic", {})
        results = _record_finished(worker)
        api_response = {"variants": [{"tone": "casual", "text": "Hey there"}]}

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        assert results == [("diplomatic", "hello")]

    def test_variant_with_error_field(self):
        """A matching variant that carries an error emits the original text."""
        worker = kde_svc.RewriteWorker("hello", "diplomatic", {})
        results = _record_finished(worker)
        api_response = {
            "variants": [
                {"tone": "diplomatic", "text": "Greetings", "error": "LLM timeout"},
            ],
        }

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        assert results == [("diplomatic", "hello")]

    def test_server_url_called_correctly(self):
        """The worker posts the text and tone to the configured rewrite URL."""
        config = {"server-host": "myhost", "server-port": 9999}
        worker = kde_svc.RewriteWorker("test", "formal", config)
        _record_finished(worker)

        with patch.object(kde_svc, "http_post", return_value=None) as mock_post:
            worker.run()

        mock_post.assert_called_once_with(
            "http://myhost:9999/rewrite",
            {"text": "test", "tones": ["formal"]},
        )

    def test_multiple_variants_picks_matching(self):
        """When several variants arrive, the one with the worker tone is used."""
        worker = kde_svc.RewriteWorker("hello", "professional", {})
        results = _record_finished(worker)
        api_response = {
            "variants": [
                {"tone": "casual", "text": "Hey"},
                {"tone": "professional", "text": "Good day"},
                {"tone": "diplomatic", "text": "Greetings"},
            ],
        }

        with patch.object(kde_svc, "http_post", return_value=api_response):
            worker.run()

        assert results == [("professional", "Good day")]

    def test_variant_text_returns_match(self):
        """variant_text returns the matching rewrite without emitting."""
        worker = kde_svc.RewriteWorker("hello", "diplomatic", {})
        api_response = {"variants": [{"tone": "diplomatic", "text": "Greetings"}]}
        with patch.object(kde_svc, "http_post", return_value=api_response):
            text = worker.variant_text()
        assert text == "Greetings"


@pytest.mark.usefixtures("qapp")
class TestHandleTranscription:
    """Checks HandleTranscription when Language Buddy is on or off."""

    def test_handle_transcription_enabled(self, service):
        """Enabled settings accept the text and pass it to process."""
        config = {
            "language-buddy-enabled": True,
            "language-buddy-bypass": False,
            "language-buddy-timeout": 30,
        }
        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscription("hello world")
        assert result is True
        mock_proc.assert_called_once_with("hello world", config)

    def test_handle_transcription_disabled(self, service):
        """Disabled settings reject the text and do not call process."""
        config = {"language-buddy-enabled": False}
        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscription("hello world")
        assert result is False
        mock_proc.assert_not_called()

    def test_handle_transcription_missing_key(self, service):
        """Missing enabled settings default to off and do not call process."""
        with (
            patch.object(kde_svc, "load_config", return_value={}),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscription("hello")
        assert result is False
        mock_proc.assert_not_called()


@pytest.mark.usefixtures("qapp")
class TestHandleTranscriptionWithContext:
    """Checks HandleTranscriptionWithContext tone parsing and fallbacks."""

    def test_handle_with_context_enabled_valid(self, service):
        """Valid context tones are forwarded to process."""
        config = {
            "language-buddy-enabled": True,
            "language-buddy-bypass": False,
            "language-buddy-timeout": 30,
        }
        context = json.dumps({"tones": ["casual", "formal"]})

        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscriptionWithContext("hello", context)
        assert result is True
        mock_proc.assert_called_once_with("hello", config, ["casual", "formal"])

    def test_handle_with_context_invalid_json_fallback(self, service):
        """Invalid context JSON falls back to the default tones."""
        config = {
            "language-buddy-enabled": True,
            "language-buddy-bypass": False,
            "language-buddy-timeout": 30,
        }

        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscriptionWithContext("hello", "not json!!!")
        assert result is True
        mock_proc.assert_called_once_with("hello", config, kde_svc.DEFAULT_TONES)

    def test_handle_with_context_disabled(self, service):
        """Disabled settings reject contextual text and do not call process."""
        config = {"language-buddy-enabled": False}

        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscriptionWithContext("hello", "{}")
        assert result is False
        mock_proc.assert_not_called()

    def test_handle_with_context_empty_tones_uses_default(self, service):
        """An empty tone list is replaced with the default tones."""
        config = {
            "language-buddy-enabled": True,
            "language-buddy-bypass": False,
            "language-buddy-timeout": 30,
        }
        context = json.dumps({"tones": []})

        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscriptionWithContext("hello", context)
        assert result is True
        mock_proc.assert_called_once_with("hello", config, kde_svc.DEFAULT_TONES)

    def test_handle_with_context_none_tones_uses_default(self, service):
        """A null tone list is replaced with the default tones."""
        config = {
            "language-buddy-enabled": True,
            "language-buddy-bypass": False,
            "language-buddy-timeout": 30,
        }
        context = json.dumps({"tones": None})

        with (
            patch.object(kde_svc, "load_config", return_value=config),
            patch.object(service, "process") as mock_proc,
        ):
            result = service.HandleTranscriptionWithContext("hello", context)
        assert result is True
        mock_proc.assert_called_once_with("hello", config, kde_svc.DEFAULT_TONES)


@pytest.mark.usefixtures("qapp")
class TestGetFocusedApp:
    """Checks GetFocusedApp across KWin success, xdotool, and total failure."""

    def test_get_focused_app_kwin_success(self, service):
        """A KWin active-client reply returns the class and caption."""
        mock_bus = MagicMock()
        mock_result = {"resourceClass": "firefox", "caption": "Mozilla Firefox"}
        mock_obj = MagicMock()
        mock_obj.Get.return_value = mock_result
        mock_bus.get_object.return_value = mock_obj

        with (
            patch.object(kde_svc.dbus, "SessionBus", return_value=mock_bus),
            patch.object(kde_svc.dbus, "Interface", return_value=MagicMock()),
        ):
            result = service.GetFocusedApp()
        assert result == ("firefox", "Mozilla Firefox")

    def test_get_focused_app_kwin_fail_xdotool_success(self, service):
        """When KWin fails, xdotool supplies the class and window title."""
        mock_bus = MagicMock()
        mock_bus.get_object.side_effect = AttributeError("KWin not available")

        with (
            patch.object(kde_svc.dbus, "SessionBus", return_value=mock_bus),
            patch.object(kde_svc, "subprocess") as mock_sub,
        ):
            mock_sub.check_output.side_effect = [b"konsole\n", b"Terminal\n"]
            mock_sub.DEVNULL = -1
            result = service.GetFocusedApp()

        assert result == ("konsole", "Terminal")

    def test_get_focused_app_all_fail(self, service):
        """When KWin and xdotool both fail, the result is two empty strings."""
        mock_bus = MagicMock()
        mock_bus.get_object.side_effect = AttributeError("fail")

        with (
            patch.object(kde_svc.dbus, "SessionBus", return_value=mock_bus),
            patch.object(kde_svc, "subprocess") as mock_sub,
        ):
            mock_sub.check_output.side_effect = subprocess.CalledProcessError(
                1,
                "xdotool",
            )
            mock_sub.DEVNULL = -1
            result = service.GetFocusedApp()

        assert result == ("", "")


@pytest.mark.usefixtures("qapp")
class TestShowHistoryPicker:
    """Checks ShowHistoryPicker for valid, empty, and unusable payloads."""

    def test_show_history_picker_valid(self, service):
        """A non-empty history list is shown and reported as accepted."""
        items = [{"text": "hello", "ts": time.time()}]
        with patch.object(service.history_overlay, "show_history") as mock_show:
            result = service.ShowHistoryPicker(json.dumps(items))
        assert result is True
        mock_show.assert_called_once()

    def test_show_history_picker_empty_items(self, service):
        """An empty history list is rejected."""
        result = service.ShowHistoryPicker("[]")
        assert result is False

    def test_show_history_picker_invalid_json(self, service):
        """Invalid history JSON is rejected."""
        result = service.ShowHistoryPicker("not json")
        assert result is False

    def test_show_history_picker_null(self, service):
        """A JSON null payload is rejected."""
        result = service.ShowHistoryPicker("null")
        assert result is False


def _run_process(service, text, config, tones=None):
    """Run process with worker threads replaced by mocks."""
    mock_worker = MagicMock()
    with (
        patch.object(kde_svc, "RewriteWorker", return_value=mock_worker),
        patch.object(kde_svc, "QThread") as mock_thread_cls,
    ):
        mock_thread_cls.return_value = MagicMock()
        if tones is not None:
            service.process(text, config, tones)
        else:
            service.process(text, config)


def _capture_show(show_args):
    """Return a show_overlay stand-in that records tones, text, and timeout."""

    def _capture(tones, text, timeout_sec=30):
        """Store the arguments passed to show_overlay."""
        show_args["tones"] = list(tones)
        show_args["text"] = text
        show_args["timeout"] = timeout_sec

    return _capture


@pytest.mark.usefixtures("qapp")
class TestServiceProcess:
    """Checks process bypass typing, tone selection, and card choices."""

    def test_process_bypass_mode(self, service):
        """Bypass mode types the original text before the overlay is shown."""
        config = {"language-buddy-bypass": True, "language-buddy-timeout": 30}
        tones = ["diplomatic"]

        with patch.object(kde_svc, "type_text") as mock_type:
            _run_process(service, "hello", config, tones)

        mock_type.assert_called_once_with("hello")

    def test_process_non_bypass(self, service):
        """Without bypass, the original text is not typed immediately."""
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}
        tones = ["diplomatic"]

        with patch.object(kde_svc, "type_text") as mock_type:
            _run_process(service, "hello", config, tones)

        mock_type.assert_not_called()

    def test_process_text_over_50_adds_summarize(self, service):
        """Text longer than fifty characters adds the summarize tone."""
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}
        long_text = "x" * 51
        show_args = {}

        with patch.object(
            service.overlay,
            "show_overlay",
            side_effect=_capture_show(show_args),
        ):
            _run_process(service, long_text, config, tones=None)

        assert "summarize" in show_args["tones"]
        assert "diplomatic" in show_args["tones"]
        assert "professional" in show_args["tones"]

    def test_process_text_under_50_no_summarize(self, service):
        """Text of exactly fifty characters does not add the summarize tone."""
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}
        short_text = "x" * 50
        show_args = {}

        with patch.object(
            service.overlay,
            "show_overlay",
            side_effect=_capture_show(show_args),
        ):
            _run_process(service, short_text, config, tones=None)

        assert "summarize" not in show_args["tones"]

    def test_process_custom_tones(self, service):
        """Explicit tones and the configured timeout are passed to the overlay."""
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 15}
        custom_tones = ["casual", "formal", "pirate"]
        show_args = {}

        with patch.object(
            service.overlay,
            "show_overlay",
            side_effect=_capture_show(show_args),
        ):
            _run_process(service, "hello", config, tones=custom_tones)

        assert show_args["tones"] == ["casual", "formal", "pirate"]
        assert show_args["timeout"] == 15

    def test_process_bypass_on_select_different_text(self, service):
        """In bypass mode, a different choice backspaces the original and retypes."""
        config = {"language-buddy-bypass": True, "language-buddy-timeout": 30}

        with patch.object(kde_svc, "type_text"):
            _run_process(service, "hello", config, tones=["diplomatic"])

        with (
            patch.object(kde_svc, "type_text") as mock_type,
            patch.object(kde_svc, "backspace_n") as mock_bs,
        ):
            service.overlay.card_selected.emit("different text")
        mock_bs.assert_called_once_with(5)
        mock_type.assert_called_once_with("different text")

    def test_process_bypass_on_select_same_text(self, service):
        """In bypass mode, choosing the original text does not type again."""
        config = {"language-buddy-bypass": True, "language-buddy-timeout": 30}

        with patch.object(kde_svc, "type_text"):
            _run_process(service, "hello", config, tones=["diplomatic"])

        with (
            patch.object(kde_svc, "type_text") as mock_type,
            patch.object(kde_svc, "backspace_n") as mock_bs,
        ):
            service.overlay.card_selected.emit("hello")
        mock_bs.assert_not_called()
        mock_type.assert_not_called()

    def test_process_non_bypass_on_select(self, service):
        """Without bypass, choosing a card types the selected text."""
        config = {"language-buddy-bypass": False, "language-buddy-timeout": 30}

        _run_process(service, "hello", config, tones=["diplomatic"])

        with patch.object(kde_svc, "type_text") as mock_type:
            service.overlay.card_selected.emit("selected text")
        mock_type.assert_called_once_with("selected text")
