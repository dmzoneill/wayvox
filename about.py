#!/usr/bin/env python3
"""Show detected NPU, GPUs, the inference library, and versions.

Opened from the GNOME or KDE tray. Reads GET /system from the running
whisper server so the versions match that process. If the server is down,
this dialog shows the local Python view and says the service was not reached.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

import diagnostics

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    from gi.repository import Gdk, GLib, Gtk
except (ImportError, ValueError) as exc:
    gi = None
    Gdk = None
    GLib = None
    Gtk = None
    _GTK_ERROR: BaseException | None = exc
else:
    _GTK_ERROR = None

_AboutBase: Any = Gtk.Application if Gtk is not None else object


def _dialog_errors():
    errors = (AttributeError, OSError, RuntimeError, TypeError, ValueError)
    if GLib is None:
        return errors
    glib_error = getattr(GLib, "Error", None)
    if isinstance(glib_error, type):
        return errors + (glib_error,)
    return errors


def fetch_report(url):
    """GET a /system report and fill in text when the server omitted it."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=4) as response:
        payload = json.loads(response.read().decode())
    if not isinstance(payload, dict):
        raise ValueError("server returned a non-object")
    payload.setdefault("source", url)
    if not payload.get("text"):
        payload["text"] = diagnostics.format_report(payload)
    return payload


def report_for(url):
    """Return the server report, or a local report when the server is down."""
    try:
        return fetch_report(url)
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        report = diagnostics.collect()
        report["source"] = "local"
        report["warning"] = (
            f"Server did not answer at {url} ({exc.__class__.__name__}: {exc}). "
            "Library versions below are this Python, not the Wayvox service. "
            "Action: start or restart the Wayvox service, "
            "confirm the tray host and port, then reopen About."
        )
        report["text"] = diagnostics.format_report(report)
        return report


def _padded_column():
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    for apply_margin in (
        box.set_margin_top,
        box.set_margin_bottom,
        box.set_margin_start,
        box.set_margin_end,
    ):
        apply_margin(12)
    return box


class About(_AboutBase):  # type: ignore[misc, valid-type]
    """GTK application that shows Wayvox hardware and library versions."""

    def __init__(self, report_text):
        """Create the About application for one system report."""
        super().__init__(application_id="one.whisper.npu.About")
        self.report_text = report_text
        self.status = None

    def report_body(self):
        """Return the report text shown in the window."""
        return self.report_text

    def copy_report(self, _button):
        """Copy the system report to the clipboard and update the status line."""
        display = Gdk.Display.get_default()
        if display is None:
            self.status.set_label("No display clipboard.")
            return
        display.get_clipboard().set(self.report_body())
        self.status.set_label("Copied.")

    def _report_view(self):
        buf = Gtk.TextBuffer()
        buf.set_text(self.report_body())
        view = Gtk.TextView(
            buffer=buf,
            monospace=True,
            editable=False,
            cursor_visible=False,
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
        )
        view.set_left_margin(8)
        view.set_top_margin(8)
        view.set_right_margin(8)
        view.set_bottom_margin(8)
        scrolled = Gtk.ScrolledWindow(vexpand=True)
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_child(view)
        return scrolled

    def _button_row(self, window):
        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        copy_btn = Gtk.Button(label="Copy")
        copy_btn.add_css_class("suggested-action")
        close_btn = Gtk.Button(label="Close")
        self.status = Gtk.Label(label="", xalign=0, hexpand=True)
        copy_btn.connect("clicked", self.copy_report)
        close_btn.connect("clicked", lambda *_args: window.close())
        buttons.append(copy_btn)
        buttons.append(close_btn)
        buttons.append(self.status)
        return buttons

    def do_activate(self):
        """Build the About window and present it."""
        window = Gtk.ApplicationWindow(application=self, title="About Wayvox")
        window.set_default_size(740, 580)
        root = _padded_column()
        window.set_child(root)
        root.append(
            Gtk.Label(
                label=(
                    "Detected NPU and GPUs, the library talking to the inference device, "
                    "and the versions to check when dictation fails."
                ),
                wrap=True,
                xalign=0,
            ),
        )
        root.append(self._report_view())
        root.append(self._button_row(window))
        window.present()


def run_dialog(text):
    """Show the system report in a GTK window and return the app status."""
    if Gtk is None or Gdk is None:
        detail = _GTK_ERROR or "GTK 4 is not available"
        raise RuntimeError(str(detail))
    app = About(text)
    return app.run([sys.argv[0]])


def main(argv=None):
    """Print the system report, or open it in a GTK dialog."""
    parser = argparse.ArgumentParser(description="Wayvox system information dialog")
    parser.add_argument("--url", default="", help="GET this /system URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--print", dest="print_only", action="store_true")
    args = parser.parse_args(argv)
    url = args.url or f"http://{args.host}:{args.port}/system"
    text = report_for(url).get("text") or ""
    graphical = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")
    if args.print_only or not graphical:
        print(text)
        return 0
    try:
        return run_dialog(text) or 0
    except _dialog_errors() as exc:
        print(text)
        print(f"GTK dialog failed ({exc}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
