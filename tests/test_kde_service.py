"""Tests for wayvox_kde_service config, HTTP, and ydotool helpers.

The service module is kde-plasmoid/wayvox_kde_service.py. tests.kde_support
loads that file by path because kde-plasmoid is not a package.
"""

import json
import subprocess
import urllib.error
from unittest.mock import MagicMock, patch

from tests.kde_support import kde_svc


class TestLoadConfig:
    """Checks that settings.json loads, and bad files become empty dicts."""

    def test_valid_json(self, tmp_path):
        """A valid settings file is returned as a dictionary."""
        cfg = {"server-host": "10.0.0.1", "server-port": 9999}
        cfg_file = tmp_path / "settings.json"
        cfg_file.write_text(json.dumps(cfg))
        with patch.object(kde_svc, "CONFIG_PATH", str(cfg_file)):
            result = kde_svc.load_config()
        assert result == cfg

    def test_file_missing(self, tmp_path):
        """A missing settings file loads as an empty dictionary."""
        with patch.object(kde_svc, "CONFIG_PATH", str(tmp_path / "nope.json")):
            result = kde_svc.load_config()
        assert result == {}

    def test_invalid_json(self, tmp_path):
        """Invalid JSON in the settings file loads as an empty dictionary."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json!!!")
        with patch.object(kde_svc, "CONFIG_PATH", str(bad)):
            result = kde_svc.load_config()
        assert result == {}

    def test_empty_file(self, tmp_path):
        """An empty settings file loads as an empty dictionary."""
        empty = tmp_path / "empty.json"
        empty.write_text("")
        with patch.object(kde_svc, "CONFIG_PATH", str(empty)):
            result = kde_svc.load_config()
        assert result == {}

    def test_valid_nested_config(self, tmp_path):
        """Nested Language Buddy keys are preserved when the file is valid."""
        cfg = {
            "server-host": "192.168.1.100",
            "server-port": 8080,
            "language-buddy-enabled": True,
            "language-buddy-bypass": True,
            "language-buddy-timeout": 60,
        }
        cfg_file = tmp_path / "settings.json"
        cfg_file.write_text(json.dumps(cfg))
        with patch.object(kde_svc, "CONFIG_PATH", str(cfg_file)):
            result = kde_svc.load_config()
        assert result["language-buddy-enabled"] is True
        assert result["language-buddy-timeout"] == 60


class TestServerUrl:
    """Checks server URL construction from explicit and saved settings."""

    def test_default_config(self):
        """An empty config uses the loopback host and port 5000."""
        url = kde_svc.server_url("/rewrite", config={})
        assert url == "http://127.0.0.1:5000/rewrite"

    def test_custom_config(self):
        """Host and port from the supplied config are written into the URL."""
        cfg = {"server-host": "10.0.0.5", "server-port": 8888}
        url = kde_svc.server_url("/api/v1", config=cfg)
        assert url == "http://10.0.0.5:8888/api/v1"

    def test_none_config_calls_load_config(self):
        """A missing config argument loads the saved host and port."""
        saved = {"server-host": "host1", "server-port": 1234}
        with patch.object(kde_svc, "load_config", return_value=saved):
            url = kde_svc.server_url("/test")
        assert url == "http://host1:1234/test"

    def test_empty_path(self):
        """An empty path leaves the URL at the host and port only."""
        url = kde_svc.server_url("", config={})
        assert url == "http://127.0.0.1:5000"

    def test_path_with_query_string(self):
        """Query strings are appended to the configured host and port."""
        url = kde_svc.server_url(
            "/search?q=hello",
            config={"server-host": "localhost", "server-port": 3000},
        )
        assert url == "http://localhost:3000/search?q=hello"

    def test_partial_config_uses_defaults(self):
        """A config with only a host or only a port fills in the other."""
        url = kde_svc.server_url("/x", config={"server-host": "myhost"})
        assert url == "http://myhost:5000/x"

        url = kde_svc.server_url("/x", config={"server-port": 7777})
        assert url == "http://127.0.0.1:7777/x"


class TestHttpPost:
    """Checks that HTTP posts return JSON or None when the call fails."""

    def test_success(self):
        """A JSON response body is decoded and returned."""
        response_data = {"result": "ok"}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = lambda item: item
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mocked:
            result = kde_svc.http_post("http://localhost:5000/test", {"key": "val"})
        assert result == response_data
        mocked.assert_called_once()

    def test_http_error(self):
        """An HTTP error status returns None."""
        error = urllib.error.HTTPError("http://x", 500, "Server Error", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            result = kde_svc.http_post("http://x", {})
        assert result is None

    def test_timeout(self):
        """A URL error such as a timeout returns None."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            result = kde_svc.http_post("http://x", {})
        assert result is None

    def test_json_decode_error(self):
        """A response that is not JSON returns None."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda item: item
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = kde_svc.http_post("http://x", {})
        assert result is None

    def test_connection_refused(self):
        """A refused connection returns None."""
        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionRefusedError("refused"),
        ):
            result = kde_svc.http_post("http://x", {})
        assert result is None


class TestTypeText:
    """Checks that type_text shells out to ydotool and swallows failures."""

    def test_success(self):
        """Successful typing invokes ydotool with the text and delay."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0)
            mock_sub.DEVNULL = -1
            kde_svc.type_text("hello world", delay_ms=4)
        mock_sub.run.assert_called_once_with(
            ["ydotool", "type", "-d", "4", "--", "hello world"],
            check=True,
            stdout=-1,
            stderr=-1,
        )

    def test_custom_delay(self):
        """The delay argument is forwarded to ydotool as text."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.type_text("test", delay_ms=10)
        args = mock_sub.run.call_args[0][0]
        assert args[3] == "10"

    def test_ydotool_missing(self):
        """A missing ydotool binary is logged and does not escape the helper."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError("ydotool not found")
            mock_sub.DEVNULL = -1
            kde_svc.type_text("hello")

    def test_subprocess_error(self):
        """A failing ydotool process is logged and does not escape the helper."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.run.side_effect = subprocess.CalledProcessError(1, "ydotool")
            mock_sub.DEVNULL = -1
            kde_svc.type_text("test")


class TestBackspaceN:
    """Checks that backspace_n presses the backspace key the requested times."""

    def test_zero(self):
        """Zero backspaces do not launch ydotool."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.backspace_n(0)
        mock_sub.run.assert_not_called()

    def test_three(self):
        """Three backspaces launch ydotool three times with the backspace key."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.backspace_n(3)
        assert mock_sub.run.call_count == 3
        for call in mock_sub.run.call_args_list:
            assert call[0][0] == ["ydotool", "key", "14:1", "14:0"]

    def test_subprocess_failure_continues(self):
        """A key-press failure is ignored and the remaining presses still run."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            mock_sub.run.side_effect = OSError("fail")
            kde_svc.backspace_n(3)
        assert mock_sub.run.call_count == 3

    def test_one(self):
        """A single backspace launches ydotool once."""
        with patch.object(kde_svc, "subprocess") as mock_sub:
            mock_sub.DEVNULL = -1
            kde_svc.backspace_n(1)
        assert mock_sub.run.call_count == 1


class TestModuleConstants:
    """Checks the module-level tone list and settings path."""

    def test_default_tones(self):
        """The default rewrite tones are diplomatic and professional."""
        assert kde_svc.DEFAULT_TONES == ["diplomatic", "professional"]

    def test_config_path_is_string(self):
        """The settings path is a string that names settings.json."""
        assert isinstance(kde_svc.CONFIG_PATH, str)
        assert "settings.json" in kde_svc.CONFIG_PATH


def test_load_config_and_server_url_together(tmp_path):
    """Load a settings file and build the rewrite URL from that config."""
    cfg = {"server-host": "10.0.0.1", "server-port": 8080}
    cfg_file = tmp_path / "settings.json"
    cfg_file.write_text(json.dumps(cfg))

    with patch.object(kde_svc, "CONFIG_PATH", str(cfg_file)):
        url = kde_svc.server_url("/rewrite")
    assert url == "http://10.0.0.1:8080/rewrite"
