#!/usr/bin/env python3
"""D-Bus service + Qt6 overlay for Wayvox on KDE Plasma.

Registers the legacy com.whisper.LanguageBuddy compatibility interface on the
session bus with the same interface used by the GNOME extension, so
push_to_talk.py works unmodified on KDE.

Requires: PyQt6, dbus-python
"""

import importlib
import json
import logging
import subprocess
import sys
import time
import urllib.request
from subprocess import CalledProcessError, TimeoutExpired
from typing import Any

import dbus
import dbus.mainloop.glib
import dbus.service

from wayvox_paths import config_file

# PyQt6 ships as a C extension whose names pylint cannot see. Load them
# dynamically so missing-name checks stay quiet and the names stay patchable.
_QtCore = importlib.import_module("PyQt6.QtCore")
_QtGui = importlib.import_module("PyQt6.QtGui")
_QtWidgets = importlib.import_module("PyQt6.QtWidgets")
QObject: Any = _QtCore.QObject
Qt = _QtCore.Qt
QThread = _QtCore.QThread
QTimer = _QtCore.QTimer
pyqtSignal = _QtCore.pyqtSignal
QColor = _QtGui.QColor
QFont = _QtGui.QFont
QGuiApplication = _QtGui.QGuiApplication
QApplication = _QtWidgets.QApplication
QGraphicsDropShadowEffect = _QtWidgets.QGraphicsDropShadowEffect
QHBoxLayout = _QtWidgets.QHBoxLayout
QLabel = _QtWidgets.QLabel
QPushButton: Any = _QtWidgets.QPushButton
QSizePolicy = _QtWidgets.QSizePolicy
QVBoxLayout = _QtWidgets.QVBoxLayout
QWidget: Any = _QtWidgets.QWidget

try:
    from gi.repository import GLib
except ImportError:
    GLib = None

log = logging.getLogger("wayvox-kde")
logging.basicConfig(level=logging.DEBUG, format="[wayvox-kde] %(message)s")

# D-Bus API names are public compatibility methods and must remain camel case.
# pylint: disable=invalid-name

CONFIG_PATH = str(config_file("settings.json"))
DEFAULT_TONES = ["diplomatic", "professional"]
HEADER_COLOR = "color: rgba(255, 255, 255, 153);"
HEADER_TRACKING = "text-transform: uppercase; letter-spacing: 1px;"
HEADER_STYLE = f"{HEADER_COLOR} {HEADER_TRACKING}"
CARD_STYLE = """
    QPushButton {
        background-color: rgba(255, 255, 255, 15);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 20);
        text-align: left;
        padding: 0;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 30);
        border-color: rgba(120, 174, 237, 128);
    }
"""
PANEL_STYLE = """
    QWidget {
        background-color: rgba(30, 30, 30, 242);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 25);
    }
"""
DISMISS_STYLE = """
    QPushButton {
        background-color: transparent;
        border-radius: 14px;
        color: rgba(255, 255, 255, 128);
        font-size: 14px;
        border: none;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 38);
        color: rgba(255, 255, 255, 230);
    }
"""


def _real_exceptions(*candidates):
    """Return real exception types, skipping mocks and subclasses."""
    chosen = []
    for candidate in candidates:
        if not isinstance(candidate, type):
            continue
        if not issubclass(candidate, BaseException):
            continue
        if any(issubclass(candidate, existing) for existing in chosen):
            continue
        kept = []
        for existing in chosen:
            if not issubclass(existing, candidate):
                kept.append(existing)
        chosen = kept
        chosen.append(candidate)
    return tuple(chosen)


DBUS_CALL_ERRORS = _real_exceptions(
    getattr(dbus, "DBusException", None),
    AttributeError,
    TypeError,
    KeyError,
    ValueError,
)


def load_config():
    """Read settings.json, or return an empty dict if it cannot be parsed."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as settings:
            return json.load(settings)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def server_url(path, config=None):
    """Build an HTTP URL for path from config or the saved settings."""
    if config is None:
        config = load_config()
    host = config.get("server-host", "127.0.0.1")
    port = config.get("server-port", 5000)
    return f"http://{host}:{port}{path}"


def http_post(url, body):
    """POST JSON to the server and return the decoded body, or None."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read())
    except (OSError, json.JSONDecodeError) as exc:
        log.error("HTTP POST %s failed: %s", url, exc)
        return None


def type_text(text, delay_ms=4):
    """Type text into the focused window with ydotool."""
    try:
        subprocess.run(
            ["ydotool", "type", "-d", str(delay_ms), "--", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, CalledProcessError) as exc:
        log.error("ydotool type failed: %s", exc)


def backspace_n(count):
    """Press backspace count times, ignoring individual key failures."""
    for _ in range(count):
        try:
            press = ["14:1", "14:0"]
            command = ["ydotool", "key", *press]
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, CalledProcessError):
            continue


def _discard_layout(layout):
    """Delete widgets stored on a layout, including one nested level."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
            continue
        nested = item.layout()
        if nested is None:
            continue
        while nested.count():
            child = nested.takeAt(0)
            child_widget = child.widget()
            if child_widget is not None:
                child_widget.deleteLater()


def _load_kwin_script():
    """Ask KWin to load the query script, ignoring bus failures."""
    try:
        bus = dbus.SessionBus()
        kwin = bus.get_object("org.kde.KWin", "/KWin")
        iface = dbus.Interface(kwin, "org.kde.KWin")
        iface.loadScript("/dev/null", "wayvox-query")
    except DBUS_CALL_ERRORS:
        return None
    return True


def _window_from_kwin():
    """Return the focused KWin class and title, or None when KWin fails."""
    try:
        bus = dbus.SessionBus()
        scripting = bus.get_object("org.kde.KWin", "/Scripting")
        result = scripting.Get(
            "org.kde.KWin.Scripting",
            "activeClient",
            dbus_interface="org.freedesktop.DBus.Properties",
        )
        class_name = str(result.get("resourceClass", ""))
        caption = str(result.get("caption", ""))
        return (class_name, caption)
    except DBUS_CALL_ERRORS:
        return None


def _window_from_xdotool():
    """Return the focused window class and title reported by xdotool."""
    try:
        class_bytes = subprocess.check_output(
            ["xdotool", "getactivewindow", "getwindowclassname"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        title_bytes = subprocess.check_output(
            ["xdotool", "getactivewindow", "getwindowname"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        return (class_bytes.decode().strip(), title_bytes.decode().strip())
    except (OSError, CalledProcessError, TimeoutExpired, UnicodeError):
        return ("", "")


class OverlayCard(QPushButton):
    """A single tone card in a floating overlay."""

    def __init__(self, tone, text, ready=False, parent=None):
        """Create a card labeled with tone and showing text."""
        super().__init__(parent)
        self.tone = tone
        self.card_text = text
        self.ready = ready
        if ready:
            cursor = Qt.CursorShape.PointingHandCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.setFlat(True)
        self.update_style()
        self._build_layout(tone, text)

    def _build_layout(self, tone, text):
        """Add the tone heading and the body label to the card."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.tone_label = QLabel(tone[0].upper() + tone[1:])
        self.tone_label.setFont(QFont("Sans", 9, QFont.Weight.Bold))
        self.tone_label.setStyleSheet("color: rgba(120, 174, 237, 0.9);")
        layout.addWidget(self.tone_label)

        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setFont(QFont("Sans", 11))
        self.text_label.setStyleSheet("color: rgba(255, 255, 255, 0.9);")
        layout.addWidget(self.text_label)

    def update_text(self, text):
        """Show text and mark the card ready to be selected."""
        self.card_text = text
        self.text_label.setText(text)
        self.ready = True
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_style()

    def update_style(self):
        """Apply the translucent card stylesheet."""
        self.setStyleSheet(CARD_STYLE)


class FloatingPanel(QWidget):
    """Frameless panel anchored to the bottom-right of the screen."""

    def __init__(self):
        """Create the translucent shell, shadow, and dismiss timer."""
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.X11BypassWindowManagerHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.dismiss)

        self._container = QWidget(self)
        self._container.setStyleSheet(PANEL_STYLE)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 128))
        self._container.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._container)

        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(8)

    def _bind_call(self, method, value):
        """Return a click slot that calls method with value."""

        def _choose(_checked=False):
            """Forward the click to method with the captured value."""
            method(value)

        return _choose

    def _dismiss_button(self):
        """Return the small close button shared by the overlays."""
        button = QPushButton("✕")
        button.setFixedSize(28, 28)
        button.setStyleSheet(DISMISS_STYLE)
        button.clicked.connect(self.dismiss)
        return button

    def _add_header(self, title):
        """Add a title row with a dismiss button to the panel."""
        header_layout = QHBoxLayout()
        header = QLabel(title)
        header.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        header.setStyleSheet(HEADER_STYLE)
        header_layout.addWidget(header, stretch=1)
        header_layout.addWidget(self._dismiss_button())
        self._layout.addLayout(header_layout)

    def _present(self, timeout_sec):
        """Size the panel, anchor it, show it, and arm the dismiss timer."""
        self.setMinimumWidth(380)
        self.setMaximumWidth(480)
        self.adjustSize()
        self.position()
        self.show()
        if timeout_sec > 0:
            self.timer.start(timeout_sec * 1000)

    def position(self):
        """Move the panel to the bottom-right of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geom = screen.availableGeometry()
        padding = 100
        x_pos = geom.x() + geom.width() - self.width() - padding
        y_pos = geom.y() + geom.height() - self.height() - padding
        self.move(x_pos, y_pos)

    def clear(self):
        """Stop the dismiss timer and delete the panel's child widgets."""
        self.timer.stop()
        _discard_layout(self._layout)

    def dismiss(self):
        """Hide the panel and drop its contents."""
        self.timer.stop()
        self.hide()
        self.clear()


class LanguageBuddyOverlay(FloatingPanel):
    """Floating overlay at the bottom-right showing rewrite variants."""

    card_selected = pyqtSignal(str)

    def __init__(self):
        """Create the card map for the rewrite overlay."""
        super().__init__()
        self.cards = {}

    def show_overlay(self, tones, original_text, timeout_sec=30):
        """Show the original text and one pending card per tone."""
        self.clear()
        self._add_header("Language Buddy")

        original = OverlayCard("original", original_text, ready=True)
        original.clicked.connect(self._bind_call(self.on_select, original_text))
        self._layout.addWidget(original)
        self.cards["original"] = original

        for tone in tones:
            card = OverlayCard(tone, "Processing...", ready=False)
            card.clicked.connect(self._bind_call(self.on_card_click, tone))
            self._layout.addWidget(card)
            self.cards[tone] = card

        self._present(timeout_sec)

    def update_card(self, tone, text):
        """Replace one tone card and keep the overlay anchored."""
        card = self.cards.get(tone)
        if card:
            card.update_text(text)
            self.adjustSize()
            self.position()

    def on_card_click(self, tone):
        """Select a tone card when that variant is ready."""
        card = self.cards.get(tone)
        if card and card.ready:
            self.on_select(card.card_text)

    def on_select(self, text):
        """Emit the chosen text and close the overlay."""
        self.card_selected.emit(text)
        self.dismiss()

    def clear(self):
        """Forget the cards and delete the widgets in the overlay."""
        self.cards.clear()
        super().clear()


class HistoryOverlay(FloatingPanel):
    """Floating overlay that lists recent transcriptions."""

    def show_history(self, items, timeout_sec=10):
        """Show one card per history item and arm the dismiss timer."""
        self.clear()
        self._add_header("Transcription History")
        for item in items:
            text = item.get("text", "")
            timestamp = item.get("ts", 0)
            if len(text) > 120:
                display = text[:120] + "..."
            else:
                display = text
            card = OverlayCard(self.time_ago(timestamp), display, ready=True)
            card.clicked.connect(self._bind_call(self.on_select, text))
            self._layout.addWidget(card)
        self._present(timeout_sec)

    def time_ago(self, timestamp):
        """Describe how long ago a unix timestamp is, in short form."""
        diff = time.time() - timestamp
        if diff < 60:
            return "just now"
        if diff < 3600:
            minutes = int(diff // 60)
            return f"{minutes}m ago"
        if diff < 86400:
            hours = int(diff // 3600)
            return f"{hours}h ago"
        days = int(diff // 86400)
        return f"{days}d ago"

    def on_select(self, text):
        """Hide the history list and type the chosen transcription."""
        self.dismiss()
        type_text(text)


class RewriteWorker(QObject):
    """Fetch one rewrite tone without blocking the Qt thread."""

    finished = pyqtSignal(str, str)

    def __init__(self, text, tone, config):
        """Store the text, tone, and server settings for the request."""
        super().__init__()
        self.text = text
        self.tone = tone
        self.config = config

    def variant_text(self):
        """Return the rewritten text, or the original when rewrite fails."""
        url = server_url("/rewrite", self.config)
        payload = {"text": self.text, "tones": [self.tone]}
        result = http_post(url, payload)
        if not result or "variants" not in result:
            return self.text
        for variant in result["variants"]:
            matched = variant.get("tone") == self.tone
            if matched and not variant.get("error"):
                return variant["text"]
        return self.text

    def run(self):
        """Rewrite the text and emit the tone with the resulting string."""
        self.finished.emit(self.tone, self.variant_text())


class LanguageBuddyService(dbus.service.Object):
    """Session-bus service that shows rewrite and history overlays."""

    def __init__(self, bus_name, overlay, history_overlay):
        """Publish the service and keep both overlays alive."""
        super().__init__(bus_name, "/com/whisper/LanguageBuddy")
        self.overlay = overlay
        self.history_overlay = history_overlay
        self._threads = []

    @dbus.service.method(
        "com.whisper.LanguageBuddy",
        in_signature="s",
        out_signature="b",
    )
    def HandleTranscription(self, text):
        """Show rewrite cards when Language Buddy is enabled."""
        config = load_config()
        if not config.get("language-buddy-enabled", False):
            return False
        log.info("HandleTranscription: %s", text[:80])
        self.process(str(text), config)
        return True

    @dbus.service.method(
        "com.whisper.LanguageBuddy",
        in_signature="ss",
        out_signature="b",
    )
    def HandleTranscriptionWithContext(self, text, context_json):
        """Show rewrite cards using tones from a JSON context object."""
        config = load_config()
        if not config.get("language-buddy-enabled", False):
            return False
        log.info("HandleTranscriptionWithContext: %s", text[:80])
        try:
            context = json.loads(str(context_json))
            tones = context.get("tones", []) or DEFAULT_TONES
        except (json.JSONDecodeError, TypeError):
            tones = DEFAULT_TONES
        self.process(str(text), config, tones)
        return True

    @dbus.service.method(
        "com.whisper.LanguageBuddy",
        in_signature="",
        out_signature="ss",
    )
    def GetFocusedApp(self):
        """Return the focused application class name and window title."""
        _load_kwin_script()
        focused = _window_from_kwin()
        if focused is not None:
            return focused
        return _window_from_xdotool()

    @dbus.service.method(
        "com.whisper.LanguageBuddy",
        in_signature="s",
        out_signature="b",
    )
    def ShowHistoryPicker(self, items_json):
        """Show history cards for a JSON list of transcriptions."""
        try:
            items = json.loads(str(items_json))
            if not items:
                return False
            self.history_overlay.show_history(items, timeout_sec=10)
            return True
        except (TypeError, ValueError, AttributeError) as exc:
            log.error("ShowHistoryPicker: %s", exc)
            return False

    def process(self, text, config, tones=None):
        """Show rewrite cards and type whichever variant the user picks."""
        bypass = config.get("language-buddy-bypass", False)
        timeout_sec = config.get("language-buddy-timeout", 30)
        if bypass:
            type_text(text)
        if tones is None:
            tones = list(DEFAULT_TONES)
            if len(text) > 50:
                tones.append("summarize")
        original_text = text

        def on_select(selected):
            """Type the choice, replacing a bypass insertion when it changed."""
            if bypass and selected != original_text:
                backspace_n(len(original_text))
            if not bypass or selected != original_text:
                type_text(selected)

        self.overlay.card_selected.disconnect()
        self.overlay.card_selected.connect(on_select)
        self.overlay.show_overlay(tones, text, timeout_sec)
        for tone in tones:
            worker = RewriteWorker(text, tone, config)
            thread = QThread()
            worker.moveToThread(thread)
            worker.finished.connect(self.overlay.update_card)
            worker.finished.connect(thread.quit)
            thread.started.connect(worker.run)

            def forget_thread(worker_thread=thread):
                """Drop the worker thread after it stops."""
                self._threads.remove(worker_thread)

            thread.finished.connect(forget_thread)
            self._threads.append(thread)
            thread.start()


def _start_glib_pump(app):
    """Return a timer that drains GLib, or None when GLib is missing."""
    if GLib is None:
        log.error("GLib is unavailable; D-Bus callbacks will not be dispatched")
        return None

    glib_loop = GLib.MainLoop()

    def glib_tick():
        """Dispatch pending GLib sources without blocking Qt."""
        context = glib_loop.get_context()
        while context.pending():
            context.iteration(False)

    timer = QTimer(app)
    timer.timeout.connect(glib_tick)
    timer.start(50)
    return timer


def main():
    """Register the Language Buddy service and run the Qt event loop."""
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    app = QApplication(sys.argv)
    app.setApplicationName("wayvox-kde")
    app.setQuitOnLastWindowClosed(False)

    overlay = LanguageBuddyOverlay()
    history_overlay = HistoryOverlay()
    bus = dbus.SessionBus()
    bus_name = dbus.service.BusName("com.whisper.LanguageBuddy", bus)
    service = LanguageBuddyService(bus_name, overlay, history_overlay)
    pump = _start_glib_pump(app)
    log.info("D-Bus service registered: %s", service)
    log.info("Listening for transcription handoffs (glib=%s)", pump is not None)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
