"""Stream mute and settings tests."""

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import push_to_talk as ptt


class TestMuteOtherStreams:
    """Tests covering muteotherstreams."""

    def _pactl_json(self, source_outputs):
        return json.dumps(source_outputs)

    def _output(self, index, pid, app="App", mute=False, volume="100%"):
        return {
            "index": index,
            "mute": mute,
            "volume": {
                "mono": {
                    "value": 65536,
                    "value_percent": volume,
                    "db": "0.00 dB",
                },
            },
            "properties": {"application.process.id": str(pid), "application.name": app},
        }

    @patch("subprocess.run")
    def test_mutes_other_streams(self, mock_run):
        """Mutes other streams."""
        outputs = [self._output(10, 100, "Chrome"), self._output(20, 200, "parec")]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._pactl_json(outputs),
        )
        result = ptt.mute_other_streams(200)
        assert len(result) == 1
        assert result[0][0] == 10  # index
        assert result[0][1] is False  # was_muted
        assert result[0][2] == "100%"  # original volume
        # list + mute + volume
        assert mock_run.call_count == 3

    @patch("subprocess.run")
    def test_skips_own_pid(self, mock_run):
        """Skips own pid."""
        outputs = [self._output(5, 999)]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._pactl_json(outputs),
        )
        result = ptt.mute_other_streams(999)
        assert not result
        assert mock_run.call_count == 1

    @patch("subprocess.run")
    def test_skips_peak_detect(self, mock_run):
        """Skips peak detect."""
        outputs = [
            {
                "index": 7,
                "mute": False,
                "volume": {
                    "mono": {
                        "value": 65536,
                        "value_percent": "100%",
                        "db": "0.00 dB",
                    },
                },
                "properties": {
                    "application.process.id": "111",
                    "application.name": "pavucontrol",
                    "media.name": "Peak detect",
                },
            }
        ]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._pactl_json(outputs),
        )
        result = ptt.mute_other_streams(222)
        assert not result

    @patch("subprocess.run")
    def test_records_already_muted(self, mock_run):
        """Records already muted."""
        outputs = [self._output(7, 111, "App", mute=True)]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._pactl_json(outputs),
        )
        result = ptt.mute_other_streams(222)
        assert result == [(7, True, "100%")]

    @patch("subprocess.run")
    def test_pactl_not_found(self, mock_run):
        """Pactl not found."""
        mock_run.side_effect = FileNotFoundError("pactl not found")
        result = ptt.mute_other_streams(100)
        assert not result

    @patch("subprocess.run")
    def test_pactl_failure(self, mock_run):
        """Pactl failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = ptt.mute_other_streams(100)
        assert not result

    @patch("subprocess.run")
    def test_no_source_outputs(self, mock_run):
        """No source outputs."""
        mock_run.return_value = MagicMock(returncode=0, stdout="[]")
        result = ptt.mute_other_streams(100)
        assert not result


class TestUnmuteStreams:
    """Tests covering unmute streams."""

    @patch("subprocess.run")
    def test_unmutes_previously_not_muted(self, mock_run):
        """Unmutes previously not muted."""
        ptt.unmute_streams([(10, False, "100%"), (20, False, "80%")])
        assert mock_run.call_count == 4  # 2x volume restore + 2x unmute
        cmds = [c[0][0] for c in mock_run.call_args_list]
        assert ["pactl", "set-source-output-volume", "10", "100%"] in cmds
        assert ["pactl", "set-source-output-volume", "20", "80%"] in cmds
        assert ["pactl", "set-source-output-mute", "10", "0"] in cmds
        assert ["pactl", "set-source-output-mute", "20", "0"] in cmds

    @patch("subprocess.run")
    def test_skips_unmute_for_already_muted(self, mock_run):
        """Skips unmute for already muted."""
        ptt.unmute_streams([(10, True, "100%"), (20, False, "100%")])
        assert mock_run.call_count == 3  # restore vol for both, unmute only idx 20
        cmds = [c[0][0] for c in mock_run.call_args_list]
        assert ["pactl", "set-source-output-mute", "20", "0"] in cmds
        assert ["pactl", "set-source-output-mute", "10", "0"] not in cmds

    @patch("subprocess.run")
    def test_empty_list(self, mock_run):
        """Empty list."""
        ptt.unmute_streams([])
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_all_already_muted(self, mock_run):
        """All already muted."""
        ptt.unmute_streams([(10, True, "100%"), (20, True, "100%")])
        # Still restores volume, just doesn't unmute
        assert mock_run.call_count == 2
        cmds = [c[0][0] for c in mock_run.call_args_list]
        assert ["pactl", "set-source-output-volume", "10", "100%"] in cmds

    @patch("subprocess.run")
    def test_unmute_failure_logged(self, mock_run):
        """Unmute failure logged."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "pactl")
        ptt.unmute_streams([(10, False, "100%")])
        mock_run.assert_called_once()


class TestLoadConfig:
    """Tests covering loadconfig."""

    def test_returns_dict_from_valid_json(self, tmp_path):
        """Returns dict from valid json."""
        config_file = tmp_path / "settings.json"
        config_file.write_text('{"hotkey": "KEY_RIGHTALT", "vad-threshold": -30}')
        result = ptt.load_config(str(config_file))
        assert result == {"hotkey": "KEY_RIGHTALT", "vad-threshold": -30}

    def test_returns_empty_dict_for_missing_file(self):
        """Returns empty dict for missing file."""
        result = ptt.load_config("/nonexistent/path/settings.json")
        assert result == {}

    def test_returns_empty_dict_for_invalid_json(self, tmp_path):
        """Returns empty dict for invalid json."""
        config_file = tmp_path / "settings.json"
        config_file.write_text("NOT VALID JSON{{{")
        result = ptt.load_config(str(config_file))
        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        """Returns empty dict for empty file."""
        config_file = tmp_path / "settings.json"
        config_file.write_text("")
        result = ptt.load_config(str(config_file))
        assert result == {}


class TestBuildSettings:
    """Tests covering buildsettings."""

    def _make_args(self, **overrides):
        defaults = {
            "key": "KEY_RIGHTCTRL",
            "backend": "openvino",
            "recall_key": "KEY_PAUSE",
            "language": None,
            "vad_threshold": -40,
            "stream_interval": 3.0,
            "type_delay": 4,
            "auto_punctuate": False,
            "translate_to": None,
            "no_sound": False,
            "no_formatting": False,
            "mute_other_streams": False,
            "no_commands": False,
            "no_notify": False,
            "port": 5000,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_defaults_from_argparse(self):
        """Defaults from argparse."""
        settings = ptt.build_settings(self._make_args())
        assert settings["key"] == "KEY_RIGHTCTRL"
        assert settings["sound"] is True
        assert settings["formatting"] is True
        assert settings["voice_commands"] is True
        assert settings["notifications"] is True
        assert settings["mute_streams"] is False

    def test_negative_flags_invert(self):
        """Negative flags invert."""
        settings = ptt.build_settings(
            self._make_args(
                no_sound=True,
                no_formatting=True,
                no_commands=True,
                no_notify=True,
            ),
        )
        assert settings["sound"] is False
        assert settings["formatting"] is False
        assert settings["voice_commands"] is False
        assert settings["notifications"] is False

    def test_env_var_overrides(self):
        """Env var overrides."""
        with patch.dict(
            os.environ,
            {"WHISPER_LANGUAGE": "de", "WHISPER_NO_SOUND": "1"},
        ):
            settings = ptt.build_settings(self._make_args())
        assert settings["language"] == "de"
        assert settings["sound"] is False


class TestApplyConfig:
    """Tests covering applyconfig."""

    def test_applies_matching_keys(self):
        """Applies matching keys."""
        settings = {"key": "KEY_RIGHTCTRL", "vad_threshold": -40, "sound": True}
        config = {"hotkey": "KEY_RIGHTALT", "vad-threshold": -30}
        changed = ptt.apply_config(settings, config)
        assert changed is True
        assert settings["key"] == "KEY_RIGHTALT"
        assert settings["vad_threshold"] == -30

    def test_no_change_returns_false(self):
        """No change returns false."""
        settings = {"key": "KEY_RIGHTCTRL", "sound": True}
        config = {"hotkey": "KEY_RIGHTCTRL", "audio-feedback-enabled": True}
        changed = ptt.apply_config(settings, config)
        assert changed is False

    def test_ignores_unknown_keys(self):
        """Ignores unknown keys."""
        settings = {"key": "KEY_RIGHTCTRL"}
        config = {"unknown-key": "value"}
        changed = ptt.apply_config(settings, config)
        assert changed is False

    def test_type_coercion(self):
        """Type coercion."""
        settings = {"vad_threshold": -40}
        config = {"vad-threshold": -30}
        changed = ptt.apply_config(settings, config)
        assert changed is True
        assert settings["vad_threshold"] == -30

    def test_invalid_type_skipped(self):
        """Invalid type skipped."""
        settings = {"vad_threshold": -40}
        config = {"vad-threshold": "not-a-number"}
        changed = ptt.apply_config(settings, config)
        assert changed is False
        assert settings["vad_threshold"] == -40

    def test_boolean_values(self):
        """Boolean values."""
        settings = {"sound": True, "formatting": False}
        config = {"audio-feedback-enabled": False, "dictation-formatting-enabled": True}
        changed = ptt.apply_config(settings, config)
        assert changed is True
        assert settings["sound"] is False
        assert settings["formatting"] is True

    def test_empty_config(self):
        """Empty config."""
        settings = {"key": "KEY_RIGHTCTRL"}
        changed = ptt.apply_config(settings, {})
        assert changed is False


class TestWatchConfig:
    """Tests covering watch config."""

    @pytest.mark.asyncio
    async def test_detects_file_change(self, tmp_path):
        """Detects file change."""
        config_file = tmp_path / "settings.json"
        config_file.write_text('{"hotkey": "KEY_RIGHTCTRL"}')

        settings = {"key": "KEY_RIGHTCTRL"}
        changes = []

        task = asyncio.create_task(
            ptt.watch_config(
                settings,
                lambda s: changes.append(dict(s)),
                str(config_file),
                interval=0.1,
            ),
        )
        await asyncio.sleep(0.2)

        time.sleep(0.05)
        config_file.write_text('{"hotkey": "KEY_RIGHTALT"}')

        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert len(changes) >= 1
        assert settings["key"] == "KEY_RIGHTALT"

    @pytest.mark.asyncio
    async def test_handles_missing_file(self, tmp_path):
        """Handles missing file."""
        settings = {"key": "KEY_RIGHTCTRL"}
        changes = []

        task = asyncio.create_task(
            ptt.watch_config(
                settings,
                lambda s: changes.append(1),
                str(tmp_path / "nonexistent.json"),
                interval=0.1,
            ),
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert not changes

    @pytest.mark.asyncio
    async def test_ignores_malformed_json(self, tmp_path):
        """Ignores malformed json."""
        config_file = tmp_path / "settings.json"
        config_file.write_text('{"hotkey": "KEY_RIGHTCTRL"}')

        settings = {"key": "KEY_RIGHTCTRL"}
        changes = []

        task = asyncio.create_task(
            ptt.watch_config(
                settings,
                lambda s: changes.append(1),
                str(config_file),
                interval=0.1,
            ),
        )
        await asyncio.sleep(0.2)

        time.sleep(0.05)
        config_file.write_text("INVALID JSON{{{")

        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert settings["key"] == "KEY_RIGHTCTRL"
        assert not changes
