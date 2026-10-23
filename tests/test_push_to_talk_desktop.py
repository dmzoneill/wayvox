"""Desktop focus, app context, and history-picker tests."""

import json
from unittest.mock import MagicMock, patch

from ptt_support import call_dbus, fake_glib_modules

import dictation_output
import push_to_talk as ptt


class TestGetFocusedApp:
    """Tests covering getfocusedapp."""

    def test_first_bus_returns_result(self):
        """First bus returns result."""
        with patch.object(
            dictation_output,
            "try_dbus_call",
            return_value=("app.desktop", "Window Title"),
        ):
            result = ptt.get_focused_app()
        assert result == ("app.desktop", "Window Title")

    def test_falls_back_to_second_bus(self):
        """Falls back to second bus."""
        with patch.object(
            dictation_output,
            "try_dbus_call",
            side_effect=[None, ("kde.desktop", "KDE App")],
        ):
            result = ptt.get_focused_app()
        assert result == ("kde.desktop", "KDE App")

    def test_all_fail_returns_empty_tuple(self):
        """All fail returns empty tuple."""
        with patch.object(dictation_output, "try_dbus_call", return_value=None):
            result = ptt.get_focused_app()
        assert result == ("", "")


class TestLoadAppContexts:
    """Tests covering loadappcontexts."""

    def test_defaults_loaded(self):
        """Defaults loaded."""
        with patch("os.path.exists", return_value=False):
            contexts = ptt.load_app_contexts()
        assert "org.mozilla.Thunderbird.desktop" in contexts
        thunderbird = contexts["org.mozilla.Thunderbird.desktop"]
        assert thunderbird == {"tones": ["professional"]}

    def test_config_file_merges(self, tmp_path):
        """Config file merges."""
        config_dir = tmp_path / ".config" / "wayvox"
        config_dir.mkdir(parents=True)
        custom = {"custom.app.desktop": {"tones": ["casual"]}}
        (config_dir / "app-contexts.json").write_text(json.dumps(custom))

        with patch(
            "os.path.expanduser",
            return_value=str(config_dir / "app-contexts.json"),
        ):
            contexts = ptt.load_app_contexts()
        assert "custom.app.desktop" in contexts
        assert contexts["custom.app.desktop"] == {"tones": ["casual"]}
        # Defaults should still be present
        assert "org.mozilla.Thunderbird.desktop" in contexts

    def test_config_file_overrides_defaults(self, tmp_path):
        """Config file overrides defaults."""
        config_dir = tmp_path / ".config" / "wayvox"
        config_dir.mkdir(parents=True)
        override = {"org.mozilla.Thunderbird.desktop": {"tones": ["casual"]}}
        (config_dir / "app-contexts.json").write_text(json.dumps(override))

        with patch(
            "os.path.expanduser",
            return_value=str(config_dir / "app-contexts.json"),
        ):
            contexts = ptt.load_app_contexts()
        assert contexts["org.mozilla.Thunderbird.desktop"] == {"tones": ["casual"]}

    def test_invalid_json_silenced(self, tmp_path):
        """Invalid json silenced."""
        config_dir = tmp_path / ".config" / "wayvox"
        config_dir.mkdir(parents=True)
        (config_dir / "app-contexts.json").write_text("NOT JSON{{{")

        with patch(
            "os.path.expanduser",
            return_value=str(config_dir / "app-contexts.json"),
        ):
            contexts = ptt.load_app_contexts()
        # Should still have defaults
        assert "org.mozilla.Thunderbird.desktop" in contexts

    def test_missing_file_returns_defaults(self):
        """Missing file returns defaults."""
        with patch("os.path.expanduser", return_value="/nonexistent/path"):
            contexts = ptt.load_app_contexts()
        assert contexts == dict(ptt.DEFAULT_APP_CONTEXTS)


class TestTryDbusHandoffWithContext:
    """Tests covering trydbushandoffwithcontext."""

    def test_first_bus_succeeds(self):
        """The first bus accepts a context-aware handoff."""
        payload = '{"tones":["pro"]}'
        result, _glib = call_dbus(
            lambda: ptt.try_dbus_handoff_with_context("text", payload),
            return_value=(True,),
        )
        assert result is True

    def test_all_fail(self):
        """Context handoff fails when every bus call returns nothing."""
        result, _glib = call_dbus(
            lambda: ptt.try_dbus_handoff_with_context("text", "{}"),
        )
        assert result is False

    def test_false_result_not_handled(self):
        """A false context-handoff result is not treated as success."""
        result, _glib = call_dbus(
            lambda: ptt.try_dbus_handoff_with_context("text", "{}"),
            return_value=(False,),
        )
        assert result is False


class TestTryDbusHistoryPicker:
    """Tests covering trydbushistorypicker."""

    def test_empty_history_returns_false(self):
        """Empty history returns false."""
        history = MagicMock()
        history.recent.return_value = []
        mock_glib = MagicMock()
        with patch.dict("sys.modules", fake_glib_modules(mock_glib)):
            result = ptt.try_dbus_history_picker(history)
        assert result is False

    def test_success_with_items(self):
        """Success with items."""
        history = MagicMock()
        history.recent.return_value = [("hello", 1234567890.0), ("world", 1234567891.0)]
        mock_glib = MagicMock()

        with (
            patch.dict("sys.modules", fake_glib_modules(mock_glib)),
            patch.object(dictation_output, "try_dbus_call", return_value=(True,)),
        ):
            result = ptt.try_dbus_history_picker(history)
        assert result is True

    def test_dbus_call_receives_json(self):
        """Dbus call receives json."""
        history = MagicMock()
        history.recent.return_value = [("hello", 100.0)]
        mock_glib = MagicMock()

        with (
            patch.object(dictation_output, "GLib", mock_glib),
            patch.object(dictation_output, "try_dbus_call", return_value=(True,)),
        ):
            ptt.try_dbus_history_picker(history)
        # The GLib.Variant call gets the JSON string
        variant_call = mock_glib.Variant.call_args
        items_json = variant_call[0][1][0]
        parsed = json.loads(items_json)
        assert parsed == [{"text": "hello", "ts": 100.0}]

    def test_all_buses_fail(self):
        """All buses fail."""
        history = MagicMock()
        history.recent.return_value = [("test", 1.0)]
        mock_glib = MagicMock()

        with (
            patch.dict("sys.modules", fake_glib_modules(mock_glib)),
            patch.object(dictation_output, "try_dbus_call", return_value=None),
        ):
            result = ptt.try_dbus_history_picker(history)
        assert result is False
