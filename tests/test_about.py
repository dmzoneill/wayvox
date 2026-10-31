"""Tests for the About dialog in about.py."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from gtk_support import import_with_fake_gtk, import_without_gtk


@pytest.fixture(name="about_gtk")
def fixture_about_gtk():
    """Import about.py against fake GTK modules."""
    return import_with_fake_gtk("about")


def _response(payload):
    """Return a urlopen context manager that yields *payload* as JSON."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    return response


class TestReports:
    """Fetching the server report and the local fallback."""

    def test_fetch_report_fills_text(self, about_gtk):
        """Fetch report fills text."""
        about = about_gtk[0]
        with (
            patch.object(about.urllib.request, "urlopen", return_value=_response({})),
            patch.object(about.diagnostics, "format_report", return_value="TEXT"),
        ):
            report = about.fetch_report("http://h/system")
        assert report == {"source": "http://h/system", "text": "TEXT"}

    def test_fetch_report_keeps_server_text(self, about_gtk):
        """Fetch report keeps server text."""
        about = about_gtk[0]
        reply = _response({"text": "server"})
        with patch.object(about.urllib.request, "urlopen", return_value=reply):
            assert about.fetch_report("http://h")["text"] == "server"

    def test_fetch_report_rejects_list(self, about_gtk):
        """Fetch report rejects a non-object (list) payload with TypeError."""
        about = about_gtk[0]
        reply = _response([1])
        with (
            patch.object(about.urllib.request, "urlopen", return_value=reply),
            pytest.raises(TypeError),
        ):
            about.fetch_report("http://h")

    def test_report_for_falls_back_to_local(self, about_gtk):
        """Report for falls back to local."""
        about = about_gtk[0]
        down = urllib.error.URLError("refused")
        with (
            patch.object(about.urllib.request, "urlopen", side_effect=down),
            patch.object(about.diagnostics, "collect", return_value={}),
            patch.object(about.diagnostics, "format_report", return_value="LOCAL"),
        ):
            report = about.report_for("http://h/system")
        assert report["source"] == "local"
        assert "Server did not answer" in report["warning"]
        assert report["text"] == "LOCAL"


class TestDialog:
    """Building and running the GTK window."""

    def test_activate_builds_window(self, about_gtk):
        """Activate builds window."""
        about, gtk, _gdk = about_gtk
        app = about.About("report")
        app.do_activate()
        gtk.ApplicationWindow.return_value.present.assert_called_once()
        assert app.report_body() == "report"
        assert app.status is gtk.Label.return_value

    def test_copy_report(self, about_gtk):
        """Copy report."""
        about, _gtk, gdk = about_gtk
        app = about.About("report")
        app.status = MagicMock()
        app.copy_report(None)
        clipboard = gdk.Display.get_default.return_value.get_clipboard.return_value
        clipboard.set.assert_called_once_with("report")
        app.status.set_label.assert_called_with("Copied.")
        gdk.Display.get_default.return_value = None
        app.copy_report(None)
        app.status.set_label.assert_called_with("No display clipboard.")

    def test_run_dialog(self, about_gtk):
        """Run dialog."""
        about = about_gtk[0]
        assert about.run_dialog("text") == 0

    def test_run_dialog_without_gtk(self):
        """Run dialog without gtk."""
        about = import_without_gtk("about")
        with pytest.raises(RuntimeError):
            about.run_dialog("text")
        assert about.Gtk is None


class TestMain:
    """Command-line entry point."""

    def _main(self, about, argv, env):
        with (
            patch.object(about, "report_for", return_value={"text": "R"}),
            patch.dict(about.os.environ, env, clear=True),
        ):
            return about.main(argv)

    def test_print_only(self, about_gtk, capsys):
        """Print only."""
        about = about_gtk[0]
        assert self._main(about, ["--print"], {"DISPLAY": ":0"}) == 0
        assert capsys.readouterr().out == "R\n"

    def test_dialog(self, about_gtk):
        """Dialog."""
        about = about_gtk[0]
        with patch.object(about, "run_dialog", return_value=None) as dialog:
            assert self._main(about, ["--port", "6000"], {"DISPLAY": ":0"}) == 0
        dialog.assert_called_once_with("R")

    def test_dialog_failure_prints(self, about_gtk, capsys):
        """Dialog failure prints."""
        about = about_gtk[0]
        with patch.object(about, "run_dialog", side_effect=RuntimeError("no gtk")):
            assert self._main(about, [], {"WAYLAND_DISPLAY": "w"}) == 1
        assert "GTK dialog failed" in capsys.readouterr().err

    def test_glib_error_is_caught(self, about_gtk):
        """Glib error is caught."""
        about = about_gtk[0]
        failure = about.GLib.Error("display gone")
        with patch.object(about, "run_dialog", side_effect=failure):
            assert self._main(about, [], {"DISPLAY": ":0"}) == 1

    @pytest.mark.parametrize("glib", [None, MagicMock(Error="not a type")])
    def test_errors_without_glib_type(self, about_gtk, glib):
        """Errors without glib type."""
        about = about_gtk[0]
        with (
            patch.object(about, "GLib", glib),
            patch.object(about, "run_dialog", side_effect=OSError("no display")),
        ):
            assert self._main(about, [], {"DISPLAY": ":0"}) == 1
