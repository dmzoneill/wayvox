"""Typing, clipboard, and D-Bus handoff tests."""

import os
from unittest.mock import MagicMock, patch

from ptt_support import call_dbus

import dictation_output
import push_to_talk as ptt


class TestTypeText:
    """Tests covering typetext."""

    def test_empty_text_returns_immediately(self):
        """Empty text returns immediately."""
        with patch.object(dictation_output, "launch_typing_proc") as mock_launch:
            ptt.type_text("", delay_ms=2)
        mock_launch.assert_not_called()

    def test_none_text_treated_as_falsy(self):
        """None text treated as falsy."""
        with patch.object(dictation_output, "launch_typing_proc") as mock_launch:
            ptt.type_text(None, delay_ms=2)
        mock_launch.assert_not_called()

    def test_wayland_ydotool_success(self):
        """Wayland ydotool success."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ) as mock_launch,
        ):
            ptt.type_text("hello", delay_ms=4)
        assert mock_launch.call_count == 1
        cmd = mock_launch.call_args[0][0]
        assert cmd[0] == "ydotool"
        assert "type" in cmd
        assert "hello" in cmd

    def test_wayland_falls_back_to_wtype(self):
        """Wayland falls back to wtype."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                side_effect=[None, 0],
            ) as mock_launch,
        ):
            ptt.type_text("hello", delay_ms=2)
        assert mock_launch.call_count == 2
        cmd = mock_launch.call_args_list[1][0][0]
        assert cmd[0] == "wtype"

    def test_wayland_falls_back_to_wl_copy(self):
        """Wayland falls back to wl copy."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                side_effect=[None, None],
            ),
            patch("subprocess.run") as mock_run,
        ):
            ptt.type_text("hello", delay_ms=2)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "wl-copy"
        assert "hello" in cmd

    def test_wayland_all_fail(self):
        """Wayland all fail."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                side_effect=[None, None],
            ),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            ptt.type_text("hello", delay_ms=2)  # should not raise

    def test_wayland_abort_stops_fallback(self):
        """rc < 0 (SIGTERM) must not fall through to wtype."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=-15,
            ) as mock_launch,
        ):
            ptt.type_text("hello", delay_ms=2)
        assert mock_launch.call_count == 1

    def test_x11_xdotool_success(self):
        """X11 xdotool success."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ) as mock_launch,
        ):
            ptt.type_text("hello", delay_ms=4)
        cmd = mock_launch.call_args[0][0]
        assert cmd[0] == "xdotool"
        assert "--clearmodifiers" in cmd
        assert "hello" in cmd

    def test_x11_xdotool_not_found(self):
        """X11 xdotool not found."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=None,
            ),
        ):
            ptt.type_text("hello", delay_ms=2)  # should not raise

    def test_x11_xdotool_fails(self):
        """X11 xdotool fails."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=1,
            ),
        ):
            ptt.type_text("hello", delay_ms=2)  # should not raise


class TestClipboardMode:
    """Tests covering clipboardmode."""

    def setup_method(self):
        """Setup method."""
        ptt.set_clipboard_mode(False)

    def teardown_method(self):
        """Teardown method."""
        ptt.set_clipboard_mode(False)

    def test_set_clipboard_mode_true(self):
        """Set clipboard mode true."""
        ptt.set_clipboard_mode(True)
        assert ptt.RUNTIME.clipboard_mode is True

    def test_set_clipboard_mode_false(self):
        """Set clipboard mode false."""
        ptt.set_clipboard_mode(True)
        ptt.set_clipboard_mode(False)
        assert ptt.RUNTIME.clipboard_mode is False

    def test_type_text_uses_clipboard_when_mode_enabled(self):
        """Type text uses clipboard when mode enabled."""
        ptt.set_clipboard_mode(True)
        with (
            patch.object(dictation_output, "type_text_clipboard") as mock_clip,
            patch.object(
                dictation_output,
                "launch_typing_proc",
            ) as mock_launch,
        ):
            ptt.type_text("hello", delay_ms=2)
        mock_clip.assert_called_once_with("hello")
        mock_launch.assert_not_called()

    def test_type_text_skips_clipboard_when_mode_disabled(self):
        """Type text skips clipboard when mode disabled."""
        ptt.set_clipboard_mode(False)
        with (
            patch.object(dictation_output, "type_text_clipboard") as mock_clip,
            patch.dict(
                os.environ,
                {"XDG_SESSION_TYPE": "wayland"},
            ),
            patch.object(dictation_output, "launch_typing_proc", return_value=0),
        ):
            ptt.type_text("hello", delay_ms=2)
        mock_clip.assert_not_called()

    def test_type_text_clipboard_empty_text_skipped(self):
        """Type text clipboard empty text skipped."""
        ptt.set_clipboard_mode(True)
        with patch.object(dictation_output, "type_text_clipboard") as mock_clip:
            ptt.type_text("", delay_ms=2)
        mock_clip.assert_not_called()

    def _make_wl_popen_mock(self):
        """Return a mock Popen instance suitable for wl-copy --foreground."""
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.wait.return_value = 0
        proc.__enter__.return_value = proc
        proc.__exit__.return_value = False
        return proc

    def test_type_text_clipboard_wayland_paste_sequence(self):
        """Type text clipboard wayland paste sequence."""
        wl_proc = self._make_wl_popen_mock()
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "clipboard_save",
                return_value=b"old",
            ),
            patch("subprocess.Popen", return_value=wl_proc) as mock_popen,
            patch.object(
                dictation_output,
                "clipboard_write",
                return_value=True,
            ) as mock_write,
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ) as mock_launch,
            patch.object(
                dictation_output,
                "is_terminal_focused",
                return_value=False,
            ),
            patch(
                "time.sleep",
            ),
        ):
            ptt.type_text_clipboard("hello")
        # wl-copy Popen started with --foreground (no --paste-once: clipboard mgr would consume it)
        popen_cmd = mock_popen.call_args[0][0]
        assert popen_cmd[0] == "wl-copy"
        assert "--foreground" in popen_cmd
        assert "--paste-once" not in popen_cmd
        # sent Ctrl+V via ydotool (non-terminal)
        cmd = mock_launch.call_args[0][0]
        assert cmd[0] == "ydotool"
        assert "29:1" in cmd
        assert "42:1" not in cmd  # no Shift
        # wl_proc killed manually after paste
        wl_proc.terminate.assert_called_once()
        # restored old clipboard
        mock_write.assert_called_once_with(b"old")

    def test_type_text_clipboard_wayland_terminal_uses_ctrl_shift_v(self):
        """Type text clipboard wayland terminal uses ctrl shift v."""
        wl_proc = self._make_wl_popen_mock()
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "clipboard_save",
                return_value=None,
            ),
            patch("subprocess.Popen", return_value=wl_proc),
            patch.object(
                dictation_output,
                "clipboard_clear",
            ),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ) as mock_launch,
            patch.object(
                dictation_output,
                "is_terminal_focused",
                return_value=True,
            ),
            patch(
                "time.sleep",
            ),
        ):
            ptt.type_text_clipboard("hello")
        cmd = mock_launch.call_args[0][0]
        # Ctrl+Shift+V = 29, 42, 47
        assert "42:1" in cmd  # LShift pressed

    def test_type_text_clipboard_wayland_clears_on_no_prior(self):
        """Type text clipboard wayland clears on no prior."""
        wl_proc = self._make_wl_popen_mock()
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "clipboard_save",
                return_value=None,
            ),
            patch("subprocess.Popen", return_value=wl_proc),
            patch.object(
                dictation_output,
                "clipboard_clear",
            ) as mock_clear,
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ),
            patch.object(
                dictation_output,
                "is_terminal_focused",
                return_value=False,
            ),
            patch(
                "time.sleep",
            ),
        ):
            ptt.type_text_clipboard("hello")
        mock_clear.assert_called_once()

    def test_type_text_clipboard_wayland_wlcopy_not_found_aborts(self):
        """Type text clipboard wayland wlcopy not found aborts."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch.object(
                dictation_output,
                "clipboard_save",
                return_value=None,
            ),
            patch("subprocess.Popen", side_effect=FileNotFoundError),
            patch.object(
                dictation_output,
                "is_terminal_focused",
                return_value=False,
            ),
            patch.object(
                dictation_output,
                "launch_typing_proc",
            ) as mock_launch,
        ):
            ptt.type_text_clipboard("hello")
        mock_launch.assert_not_called()

    def test_type_text_clipboard_x11_paste_sequence(self):
        """Type text clipboard x11 paste sequence."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            patch.object(
                dictation_output,
                "clipboard_save",
                return_value=b"old",
            ),
            patch.object(
                dictation_output,
                "clipboard_write",
                return_value=True,
            ) as mock_write,
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ) as mock_launch,
            patch.object(
                dictation_output,
                "is_terminal_focused",
                return_value=False,
            ),
            patch(
                "time.sleep",
            ),
        ):
            ptt.type_text_clipboard("hello")
        cmd = mock_launch.call_args[0][0]
        assert cmd[0] == "xdotool"
        assert "ctrl+v" in cmd
        assert "shift" not in cmd
        assert mock_write.call_count == 2  # write text + restore

    def test_type_text_clipboard_x11_terminal_uses_ctrl_shift_v(self):
        """Type text clipboard x11 terminal uses ctrl shift v."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}),
            patch.object(
                dictation_output,
                "clipboard_save",
                return_value=None,
            ),
            patch.object(dictation_output, "clipboard_write", return_value=True),
            patch.object(
                dictation_output,
                "clipboard_clear",
            ),
            patch.object(
                dictation_output,
                "launch_typing_proc",
                return_value=0,
            ) as mock_launch,
            patch.object(
                dictation_output,
                "is_terminal_focused",
                return_value=True,
            ),
            patch(
                "time.sleep",
            ),
        ):
            ptt.type_text_clipboard("hello")
        cmd = mock_launch.call_args[0][0]
        assert "ctrl+shift+v" in cmd

    def test_is_terminal_focused_known_terminals(self):
        """Is terminal focused known terminals."""
        for app_id in ["org.gnome.Terminal", "org.gnome.Console", "foot", "kitty"]:
            focused = patch.object(
                dictation_output,
                "get_focused_app",
                return_value=(app_id, ""),
            )
            with focused:
                assert ptt.is_terminal_focused()
            with patch.object(
                dictation_output,
                "get_focused_app",
                return_value=(app_id + ".desktop", ""),
            ):
                assert ptt.is_terminal_focused()

    def test_is_terminal_focused_non_terminal(self):
        """Is terminal focused non terminal."""
        with patch.object(
            dictation_output,
            "get_focused_app",
            return_value=("org.mozilla.Firefox", ""),
        ):
            assert not ptt.is_terminal_focused()

    def test_clipboard_save_wayland_success(self):
        """Clipboard save wayland success."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch(
                "subprocess.run",
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=b"old text")
            result = ptt.clipboard_save()
        assert result == b"old text"

    def test_clipboard_save_wayland_empty(self):
        """Clipboard save wayland empty."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch(
                "subprocess.run",
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout=b"")
            result = ptt.clipboard_save()
        assert result is None

    def test_clipboard_write_wayland_success(self):
        """Clipboard write wayland success."""
        with (
            patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}),
            patch(
                "subprocess.run",
            ) as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = ptt.clipboard_write("hello")
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "wl-copy"


class TestAbortTyping:
    """Tests covering aborttyping."""

    def test_terminates_running_proc(self):
        """Terminates running proc."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        ptt.RUNTIME.typing_proc = mock_proc
        ptt.abort_typing()
        mock_proc.terminate.assert_called_once()
        ptt.RUNTIME.typing_proc = None

    def test_no_proc_does_nothing(self):
        """No proc does nothing."""
        ptt.RUNTIME.typing_proc = None
        ptt.abort_typing()  # should not raise

    def test_already_finished_proc_not_terminated(self):
        """Already finished proc not terminated."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        ptt.RUNTIME.typing_proc = mock_proc
        ptt.abort_typing()
        mock_proc.terminate.assert_not_called()
        ptt.RUNTIME.typing_proc = None


class TestTryDbusCall:
    """Tests covering trydbuscall."""

    def test_success_returns_unpacked(self):
        """Success returns unpacked."""
        mock_result = MagicMock()
        mock_result.unpack.return_value = (True,)
        mock_bus = MagicMock()
        mock_bus.call_sync.return_value = mock_result

        mock_gio = MagicMock()
        mock_gio.bus_get_sync.return_value = mock_bus
        mock_glib = MagicMock()

        with (
            patch.object(dictation_output, "Gio", mock_gio),
            patch.object(dictation_output, "GLib", mock_glib),
        ):
            result = ptt.try_dbus_call(
                ("org.test", "/obj", "iface"),
                "Method",
                MagicMock(),
                "(b)",
            )
        assert result == (True,)

    def test_exception_returns_none(self):
        """Exception returns none."""
        mock_gio = MagicMock()
        mock_gio.bus_get_sync.side_effect = RuntimeError("D-Bus error")
        mock_glib = MagicMock()

        with (
            patch.object(dictation_output, "Gio", mock_gio),
            patch.object(dictation_output, "GLib", mock_glib),
        ):
            result = ptt.try_dbus_call(
                ("org.test", "/obj", "iface"),
                "Method",
                MagicMock(),
                "(b)",
            )
        assert result is None

    def test_none_reply_type(self):
        """None reply type."""
        mock_result = MagicMock()
        mock_result.unpack.return_value = ()
        mock_bus = MagicMock()
        mock_bus.call_sync.return_value = mock_result

        mock_gio = MagicMock()
        mock_gio.bus_get_sync.return_value = mock_bus
        mock_glib = MagicMock()

        with (
            patch.object(dictation_output, "Gio", mock_gio),
            patch.object(dictation_output, "GLib", mock_glib),
        ):
            ptt.try_dbus_call(
                ("org.test", "/obj", "iface"),
                "Method",
                MagicMock(),
                None,
            )
        # Should pass None for reply_type
        call_args = mock_bus.call_sync.call_args[0]
        assert call_args[5] is None


class TestTryDbusHandoff:
    """Tests covering trydbushandoff."""

    def test_first_bus_succeeds(self):
        """The first bus accepts a plain transcription handoff."""
        result, _glib = call_dbus(
            lambda: ptt.try_dbus_handoff("hello"),
            return_value=(True,),
        )
        assert result is True

    def test_first_bus_fails_second_succeeds(self):
        """A failed first bus is followed by a successful second bus."""
        result, _glib = call_dbus(
            lambda: ptt.try_dbus_handoff("hello"),
            side_effect=[None, (True,)],
        )
        assert result is True

    def test_all_buses_fail(self):
        """Handoff returns false when every bus call fails."""
        result, _glib = call_dbus(lambda: ptt.try_dbus_handoff("hello"))
        assert result is False

    def test_result_false_treated_as_not_handled(self):
        """A false D-Bus result does not count as a successful handoff."""
        result, _glib = call_dbus(
            lambda: ptt.try_dbus_handoff("hello"),
            return_value=(False,),
        )
        assert result is False


class TestNotify:
    """Tests covering notify."""

    def test_basic_notification(self, mock_subprocess):
        """Basic notification."""
        ptt.notify("Test", "Body text", timeout_ms=5000)
        args = mock_subprocess.call_args[0][0]
        assert "notify-send" in args
        assert "--expire-time" in args
        assert "5000" in args
        assert "Test" in args
        assert "Body text" in args

    def test_no_body(self, mock_subprocess):
        """No body."""
        ptt.notify("Title only")
        args = mock_subprocess.call_args[0][0]
        assert "Body" not in " ".join(args)
        # Body not appended when empty
        assert len(args) == 4  # notify-send --expire-time 3000 Title only

    def test_failure_silenced(self, mock_subprocess):
        """Failure silenced."""
        mock_subprocess.side_effect = FileNotFoundError("no notify-send")
        ptt.notify("Test")  # Should not raise


class TestSendKeys:
    """Tests covering sendkeys."""

    def test_wayland_ydotool(self, mock_subprocess):
        """Wayland ydotool."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.send_keys("29:1 30:1 30:0 29:0")
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "ydotool"
        assert "key" in args

    def test_wayland_ydotool_failure_silenced(self, mock_subprocess):
        """Wayland ydotool failure silenced."""
        mock_subprocess.side_effect = FileNotFoundError
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            ptt.send_keys("29:1 30:1 30:0 29:0")  # Should not raise

    def test_x11_xdotool_select_all(self, mock_subprocess):
        """X11 xdotool select all."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("29:1 30:1 30:0 29:0")
        # ctrl+ a -> xdotool key ctrl+a
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "xdotool"
        assert "ctrl+a" in args

    def test_x11_return_key(self, mock_subprocess):
        """X11 return key."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("28:1 28:0")
        args = mock_subprocess.call_args[0][0]
        assert "Return" in args

    def test_x11_backspace(self, mock_subprocess):
        """X11 backspace."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("14:1 14:0")
        args = mock_subprocess.call_args[0][0]
        assert "BackSpace" in args

    def test_x11_tab(self, mock_subprocess):
        """X11 tab."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("15:1 15:0")
        args = mock_subprocess.call_args[0][0]
        assert "Tab" in args

    def test_x11_combo_ctrl_z(self, mock_subprocess):
        """X11 combo ctrl z."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("29:1 44:1 44:0 29:0")
        args = mock_subprocess.call_args[0][0]
        assert "ctrl+z" in args

    def test_x11_xdotool_failure_silenced(self, mock_subprocess):
        """X11 xdotool failure silenced."""
        mock_subprocess.side_effect = FileNotFoundError
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("28:1 28:0")  # Should not raise

    def test_unknown_scancode_skipped(self, mock_subprocess):
        """Unknown scancode skipped."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}):
            ptt.send_keys("99:1 99:0")
        # unknown code maps to "" — no xdotool calls
        mock_subprocess.assert_not_called()


class TestExecuteVoiceCommand:
    """Tests covering executevoicecommand."""

    def test_exact_match(self):
        """Exact match."""
        with patch.object(dictation_output, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("select all")
        assert result is True
        mock_sk.assert_called_once()

    def test_case_insensitive(self):
        """Case insensitive."""
        with patch.object(dictation_output, "send_keys"):
            result = ptt.execute_voice_command("Select All")
        assert result is True

    def test_strips_whitespace(self):
        """Strips whitespace."""
        with patch.object(dictation_output, "send_keys"):
            result = ptt.execute_voice_command("  select all  ")
        assert result is True

    def test_strips_punctuation(self):
        """Strips punctuation."""
        with patch.object(dictation_output, "send_keys"):
            result = ptt.execute_voice_command("select all.")
        assert result is True

    def test_strips_please_suffix(self):
        """Strips please suffix."""
        with patch.object(dictation_output, "send_keys"):
            result = ptt.execute_voice_command("undo that please")
        assert result is True

    def test_strips_please_and_punctuation(self):
        # "Copy that please!" -> lower -> "copy that please!" -> rstrip -> "copy that please"
        # -> strip " please" -> "copy that" -> match
        """Strips please and punctuation."""
        with patch.object(dictation_output, "send_keys"):
            result = ptt.execute_voice_command("Copy that please!")
        assert result is True

    def test_please_with_comma_before_not_stripped(self):
        # "Copy that, please!" -> "copy that, please" -> "copy that," -- no match
        """Please with comma before not stripped."""
        result = ptt.execute_voice_command("Copy that, please!")
        assert result is False

    def test_no_match_returns_false(self):
        """No match returns false."""
        result = ptt.execute_voice_command("random text")
        assert result is False

    def test_type_action(self):
        """Type action."""
        with patch.object(dictation_output, "type_text") as mock_tt:
            result = ptt.execute_voice_command("period")
        assert result is True
        mock_tt.assert_called_once_with(".", delay_ms=2)

    def test_new_paragraph_double_enter(self):
        """New paragraph double enter."""
        with patch.object(dictation_output, "send_keys") as mock_sk:
            result = ptt.execute_voice_command("new paragraph")
        assert result is True
        assert mock_sk.call_count == 2

    def test_multiple_punctuation_marks(self):
        """Test all punctuation voice commands."""
        punct_map = {
            "period": ".",
            "comma": ",",
            "question mark": "?",
            "exclamation mark": "!",
            "exclamation point": "!",
            "colon": ":",
            "semicolon": ";",
        }
        for cmd, expected in punct_map.items():
            with patch.object(dictation_output, "type_text") as mock_tt:
                result = ptt.execute_voice_command(cmd)
            assert result is True, f"Failed for command: {cmd}"
            mock_tt.assert_called_once_with(expected, delay_ms=2)
