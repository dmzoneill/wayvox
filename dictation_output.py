"""Typing, clipboard paste, D-Bus handoff, and dictation formatting."""

import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time

try:
    from gi.repository import Gio, GLib
except ImportError:  # desktop integration is optional
    Gio = None
    GLib = None

log = logging.getLogger("ptt")

VOICE_COMMANDS = {
    "select all": [("key", "29:1 30:1 30:0 29:0")],
    "undo that": [("key", "29:1 44:1 44:0 29:0")],
    "redo that": [("key", "29:1 21:1 21:0 29:0")],
    "copy that": [("key", "29:1 46:1 46:0 29:0")],
    "paste that": [("key", "29:1 47:1 47:0 29:0")],
    "cut that": [("key", "29:1 45:1 45:0 29:0")],
    "new line": [("key", "28:1 28:0")],
    "new paragraph": [("key", "28:1 28:0"), ("key", "28:1 28:0")],
    "tab key": [("key", "15:1 15:0")],
    "delete that": [("key", "14:1 14:0")],
    "delete word": [("key", "29:1 14:1 14:0 29:0")],
    "period": [("type", ".")],
    "comma": [("type", ",")],
    "question mark": [("type", "?")],
    "exclamation mark": [("type", "!")],
    "exclamation point": [("type", "!")],
    "colon": [("type", ":")],
    "semicolon": [("type", ";")],
    "open quote": [("type", '"')],
    "close quote": [("type", '"')],
}

DEFAULT_APP_CONTEXTS = {
    "org.mozilla.Thunderbird.desktop": {"tones": ["professional"]},
    "org.mozilla.thunderbird.desktop": {"tones": ["professional"]},
    "com.slack.Slack.desktop": {"tones": ["diplomatic"]},
    "org.gnome.Evolution.desktop": {"tones": ["professional"]},
    "org.gnome.Terminal.desktop": {"tones": []},
    "org.gnome.Console.desktop": {"tones": []},
    "com.raggesilver.BlackBox.desktop": {"tones": []},
    "code.desktop": {"tones": []},
}


class TypingRuntime:
    """Mutable typing-process state shared by the dictation client."""

    def __init__(self):
        """Start with no typing process and clipboard paste disabled."""
        self.typing_proc = None
        self.lock = threading.Lock()
        self.clipboard_mode = False

    def bind(self, proc):
        """Remember *proc* so Escape can cancel the keystrokes it is sending."""
        self.typing_proc = proc

    def release(self, proc):
        """Forget *proc* when it is still the active typing process."""
        if self.typing_proc is proc:
            self.typing_proc = None


RUNTIME = TypingRuntime()


def set_clipboard_mode(enabled):
    """Enable or disable clipboard paste instead of keystroke typing."""
    RUNTIME.clipboard_mode = bool(enabled)


def launch_typing_proc(cmd):
    """Run *cmd*, remember it so Escape can cancel it, and return its exit code.

    Returns None when the executable is missing.
    """
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as proc:
            with RUNTIME.lock:
                RUNTIME.bind(proc)
            return_code = proc.wait()
            with RUNTIME.lock:
                RUNTIME.release(proc)
            return return_code
    except FileNotFoundError:
        return None


def abort_typing():
    """Terminate any in-flight typing subprocess (called on Escape)."""
    with RUNTIME.lock:
        proc = RUNTIME.typing_proc
    if proc and proc.poll() is None:
        proc.terminate()


def clipboard_save():
    """Return current clipboard bytes (or None if empty/unavailable)."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        try:
            r = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True,
                check=False,
            )
            return r.stdout if r.returncode == 0 else None
        except FileNotFoundError:
            return None
    else:
        for cmd in [
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
        ]:
            try:
                r = subprocess.run(cmd, capture_output=True, check=False)
                if r.returncode == 0:
                    return r.stdout
            except FileNotFoundError:
                continue
        return None


def clipboard_write(data):
    """Write bytes or str to clipboard. Returns True on success."""
    if isinstance(data, str):
        data = data.encode()
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        try:
            subprocess.run(["wl-copy", "--"], input=data, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    else:
        for cmd in [
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]:
            try:
                subprocess.run(cmd, input=data, check=True)
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        return False


def clipboard_clear():
    """Clear the clipboard, or write an empty payload on X11."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        try:
            subprocess.run(["wl-copy", "--clear"], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    else:
        clipboard_write(b"")


TERMINAL_APP_IDS = {
    "org.gnome.Terminal",
    "org.gnome.Console",
    "com.raggesilver.BlackBox",
    "com.mitchellh.ghostty",
    "io.elementary.terminal",
    "org.kde.konsole",
    "org.kde.yakuake",
    "org.xfce.terminal",
    "com.alacritty.Alacritty",
    "foot",
    "kitty",
    "wezterm-gui",
    "net.sourceforge.xterm",
}


def is_terminal_focused():
    """True when the focused window is a known terminal emulator."""
    app_id, _ = get_focused_app()
    return app_id.removesuffix(".desktop") in TERMINAL_APP_IDS


def _warn_paste_result(return_code, tool):
    """Log when the paste keystroke tool is missing or fails."""
    if return_code is None:
        log.warning("%s not found; text is in clipboard — paste manually", tool)
    elif return_code != 0:
        log.warning("%s key paste failed (rc=%d)", tool, return_code)


def _paste_keys(is_terminal):
    """ydotool scancodes for Ctrl-V, or Ctrl-Shift-V in a terminal."""
    if is_terminal:
        return ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]
    return ["29:1", "47:1", "47:0", "29:0"]


def _paste_with_wl_copy(text, is_terminal):
    """Offer *text* with wl-copy and press the paste shortcut.

    wl-copy stays in the foreground until after the paste. ``--paste-once``
    exits when a clipboard manager reads the offer, so the real Ctrl-V
    would see an empty clipboard.
    """
    try:
        with subprocess.Popen(
            ["wl-copy", "--foreground", "--"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as wl_proc:
            wl_proc.stdin.write(text.encode())
            wl_proc.stdin.close()
            time.sleep(0.1)
            return_code = launch_typing_proc(
                ["ydotool", "key", *_paste_keys(is_terminal)],
            )
            _warn_paste_result(return_code, "ydotool")
            time.sleep(0.2)
            wl_proc.terminate()
            try:
                wl_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                wl_proc.kill()
    except FileNotFoundError:
        log.error(
            "wl-copy not found — install wl-clipboard for clipboard paste mode",
        )
        return False
    return True


def _paste_with_xdotool(text, is_terminal):
    """Write *text* to the X11 clipboard and press the paste shortcut."""
    if not clipboard_write(text):
        log.error("Clipboard write failed — install xclip or xsel")
        return False
    key_combo = "ctrl+shift+v" if is_terminal else "ctrl+v"
    return_code = launch_typing_proc(["xdotool", "key", key_combo])
    _warn_paste_result(return_code, "xdotool")
    return True


def type_text_clipboard(text):
    """Paste *text* with the clipboard, then restore the previous contents."""
    previous = clipboard_save()
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    terminal = is_terminal_focused()
    if session_type == "wayland":
        pasted = _paste_with_wl_copy(text, terminal)
    else:
        pasted = _paste_with_xdotool(text, terminal)
    if not pasted:
        return
    time.sleep(0.1)
    if previous is not None:
        clipboard_write(previous)
    else:
        clipboard_clear()


def type_text(text, delay_ms=2):
    """Type text into the focused window using wtype (Wayland) or xdotool (X11).

    When clipboard mode is active, pastes via clipboard + Ctrl-V instead, which
    is atomic (no window-focus issue) and instant regardless of text length.
    """
    if not text:
        return

    if RUNTIME.clipboard_mode:
        type_text_clipboard(text)
        return

    session_type = os.environ.get("XDG_SESSION_TYPE", "")

    if session_type == "wayland":
        for tool, cmd in [
            ("ydotool", ["ydotool", "type", "-d", str(delay_ms), "--", text]),
            ("wtype", ["wtype", "-d", str(delay_ms), "--", text]),
        ]:
            rc = launch_typing_proc(cmd)
            if rc is None:
                log.warning("%s not found", tool)
                continue
            if rc <= 0:  # 0 = success; negative = killed by signal (Escape abort)
                return
            log.warning("%s failed (rc=%d), trying next", tool, rc)
        try:
            subprocess.run(["wl-copy", text], check=True)
            log.info(
                "Text copied to clipboard (install ydotool or wtype for direct typing)",
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log.error("All typing methods failed: %s", e)
    else:
        rc = launch_typing_proc(
            [
                "xdotool",
                "type",
                "--delay",
                str(delay_ms),
                "--clearmodifiers",
                "--",
                text,
            ],
        )
        if rc is None:
            log.error("xdotool not found")
        elif rc > 0:
            log.error("xdotool failed (rc=%d)", rc)


async def type_text_async(text, delay_ms=2):
    """Run type_text in a thread executor so the asyncio event loop stays responsive.
    This lets Escape key events be processed and abort_typing() called mid-output."""
    await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: type_text(text, delay_ms=delay_ms),
    )


def _dbus_errors():
    """Exception types a failed desktop D-Bus call may raise."""
    found = [OSError, RuntimeError, TypeError, ValueError, AttributeError]
    error_type = getattr(GLib, "Error", None) if GLib is not None else None
    if isinstance(error_type, type) and issubclass(error_type, BaseException):
        found.append(error_type)
    return tuple(found)


def try_dbus_call(destination, method, args_variant, reply_type):
    """Call one D-Bus method and return the unpacked result, or None.

    *destination* is ``(bus_name, object_path, interface)``.
    """
    if Gio is None or GLib is None:
        return None
    bus_name, object_path, interface = destination
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION)
        reply_variant = GLib.VariantType(reply_type) if reply_type else None
        result = bus.call_sync(
            bus_name,
            object_path,
            interface,
            method,
            args_variant,
            reply_variant,
            Gio.DBusCallFlags.NONE,
            500,
            None,
        )
        return result.unpack()
    except _dbus_errors():
        return None


DBUS_BUS_NAMES = ["org.gnome.Shell", "com.whisper.LanguageBuddy"]
DBUS_OBJECT_PATH = "/com/whisper/LanguageBuddy"
DBUS_INTERFACE = "com.whisper.LanguageBuddy"


def try_dbus_handoff(text):
    """Try to hand off transcription to desktop extension via D-Bus.
    Tries GNOME Shell first, then standalone KDE service."""
    if GLib is None:
        return False
    for bus_name in DBUS_BUS_NAMES:
        result = try_dbus_call(
            (bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE),
            "HandleTranscription",
            GLib.Variant("(s)", (text,)),
            "(b)",
        )
        if result and result[0]:
            return True
    return False


def notify(summary, body="", timeout_ms=3000):
    """Show a desktop notification."""
    cmd = ["notify-send", "--expire-time", str(timeout_ms), summary]
    if body:
        cmd.append(body)
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def send_keys(scancode_str):
    """Send key events via ydotool (Wayland) or xdotool (X11)."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        try:
            subprocess.run(
                ["ydotool", "key"] + scancode_str.split(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    else:
        key_map = {
            "29:1": "ctrl+",
            "29:0": "",
            "30:1": "a",
            "30:0": "",
            "44:1": "z",
            "44:0": "",
            "21:1": "y",
            "21:0": "",
            "46:1": "c",
            "46:0": "",
            "47:1": "v",
            "47:0": "",
            "45:1": "x",
            "45:0": "",
            "28:1": "Return",
            "28:0": "",
            "15:1": "Tab",
            "15:0": "",
            "14:1": "BackSpace",
            "14:0": "",
        }
        keys = []
        combo = ""
        for code in scancode_str.split():
            mapped = key_map.get(code, "")
            if mapped.endswith("+"):
                combo += mapped
            elif mapped:
                keys.append(combo + mapped)
                combo = ""
        for key in keys:
            try:
                subprocess.run(
                    ["xdotool", "key", key],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass


def execute_voice_command(text):
    """Check if text matches a voice command and execute it. Returns True if handled."""
    normalized = text.strip().lower().rstrip(".!?,;")
    if normalized.endswith(" please"):
        normalized = normalized[:-7]
    for phrase, actions in VOICE_COMMANDS.items():
        if normalized == phrase:
            for action_type, action_val in actions:
                if action_type == "key":
                    send_keys(action_val)
                elif action_type == "type":
                    type_text(action_val, delay_ms=2)
            return True
    return False


class TranscriptionHistory:
    """Rolling log of recent transcriptions in SQLite."""

    def __init__(self, max_items=50):
        """Create the history database under the Wayvox config directory."""
        self.max_items = max_items
        db_dir = os.path.expanduser("~/.config/wayvox")
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, "history.db")
        create_sql = (
            "CREATE TABLE IF NOT EXISTS history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, timestamp REAL NOT NULL)"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(create_sql)

    def add(self, text):
        """Insert a transcription and drop rows beyond *max_items*."""
        kept_rows = "SELECT id FROM history ORDER BY id DESC LIMIT ?"
        delete_sql = f"DELETE FROM history WHERE id NOT IN ({kept_rows})"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO history (text, timestamp) VALUES (?, ?)",
                (text, time.time()),
            )
            conn.execute(delete_sql, (self.max_items,))

    def recent(self, n=10):
        """Return the newest ``(text, timestamp)`` rows."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT text, timestamp FROM history ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()


def get_focused_app():
    """Query desktop extension for the focused application via D-Bus.
    Tries GNOME Shell first, then standalone KDE service."""
    for bus_name in DBUS_BUS_NAMES:
        result = try_dbus_call(
            (bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE),
            "GetFocusedApp",
            None,
            "(ss)",
        )
        if result:
            return result
    return ("", "")


def load_app_contexts():
    """Load per-app context overrides from config file, merged with defaults."""
    contexts = dict(DEFAULT_APP_CONTEXTS)
    config_path = os.path.expanduser("~/.config/wayvox/app-contexts.json")
    if not os.path.exists(config_path):
        legacy_path = os.path.expanduser("~/.config/whisper-npu/app-contexts.json")
        if os.path.exists(legacy_path):
            config_path = legacy_path
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                contexts.update(json.load(handle))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return contexts


def try_dbus_handoff_with_context(text, context_json):
    """Hand off transcription with per-app context to desktop extension."""
    if GLib is None:
        return False
    for bus_name in DBUS_BUS_NAMES:
        result = try_dbus_call(
            (bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE),
            "HandleTranscriptionWithContext",
            GLib.Variant("(ss)", (text, context_json)),
            "(b)",
        )
        if result and result[0]:
            return True
    return False


def try_dbus_history_picker(history):
    """Ask desktop extension to show history picker overlay."""
    if GLib is None:
        return False
    items = history.recent(10)
    if not items:
        return False
    items_json = json.dumps([{"text": t, "ts": ts} for t, ts in items])
    for bus_name in DBUS_BUS_NAMES:
        result = try_dbus_call(
            (bus_name, DBUS_OBJECT_PATH, DBUS_INTERFACE),
            "ShowHistoryPicker",
            GLib.Variant("(s)", (items_json,)),
            "(b)",
        )
        if result and result[0]:
            return True
    return False


def _spawn_player(argv):
    """Start a sound player and return it so playback can outlive this call."""
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def play_sound(sound_name):
    """Play a freedesktop sound, falling back to the bundled file."""
    system_path = f"/usr/share/sounds/freedesktop/stereo/{sound_name}.oga"
    bundled_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sounds",
        f"{sound_name}.oga",
    )
    path = system_path if os.path.exists(system_path) else bundled_path
    if not os.path.exists(path):
        return
    for player in ("pw-play", "paplay"):
        try:
            _spawn_player([player, path])
        except FileNotFoundError:
            continue
        else:
            return


WORD_NUMS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

ABBREVIATIONS = {
    r"\bmister\b": "Mr.",
    r"\bmissus\b": "Mrs.",
    r"\bdoctor\b": "Dr.",
    r"\bversus\b": "vs.",
    r"\bet cetera\b": "etc.",
    r"\bfor example\b": "e.g.",
    r"\bthat is\b": "i.e.",
}


def format_dictation(text):
    """Apply dictation formatting rules."""
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(
        r"\b(\w+)\s+at\s+(\w+)\s+dot\s+(\w+)\b",
        lambda m: f"{m.group(1)}@{m.group(2)}.{m.group(3)}",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bw w w dot\s+(\w+)\s+dot\s+(\w+)\b",
        lambda m: f"www.{m.group(1)}.{m.group(2)}",
        text,
        flags=re.IGNORECASE,
    )

    for word, num in sorted(WORD_NUMS.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{word}\b", str(num), text, flags=re.IGNORECASE)

    text = re.sub(
        r"\b(\d+)\s+hundred\s+(\d+)\b",
        lambda m: str(int(m.group(1)) * 100 + int(m.group(2))),
        text,
    )
    text = re.sub(r"\b(\d+)\s+hundred\b", lambda m: str(int(m.group(1)) * 100), text)

    if text:
        text = text[0].upper() + text[1:]

    return text
