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

import diagnostics


def fetch_report(url):
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
    try:
        return fetch_report(url)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
        report = diagnostics.collect()
        report["source"] = "local"
        report["warning"] = (
            f"Server did not answer at {url} ({exc.__class__.__name__}: {exc}). "
            "Library versions below are this Python, not the Voxforge service. "
            "Action: start or restart the Voxforge service, confirm the tray host and port, then reopen About."
        )
        report["text"] = diagnostics.format_report(report)
        return report


def run_dialog(text):
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    from gi.repository import Gdk, Gtk

    class About(Gtk.Application):
        def __init__(self):
            super().__init__(application_id="one.whisper.npu.About")

        def do_activate(self):
            window = Gtk.ApplicationWindow(application=self, title="About Voxforge")
            window.set_default_size(740, 580)

            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            root.set_margin_top(12)
            root.set_margin_bottom(12)
            root.set_margin_start(12)
            root.set_margin_end(12)
            window.set_child(root)

            root.append(Gtk.Label(
                label=(
                    "Detected NPU and GPUs, the library talking to the inference device, "
                    "and the versions to check when dictation fails."
                ),
                wrap=True,
                xalign=0,
            ))

            buf = Gtk.TextBuffer()
            buf.set_text(text)
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
            root.append(scrolled)

            buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            copy_btn = Gtk.Button(label="Copy")
            copy_btn.add_css_class("suggested-action")
            close_btn = Gtk.Button(label="Close")
            status = Gtk.Label(label="", xalign=0, hexpand=True)

            def copy(_button):
                display = Gdk.Display.get_default()
                if display is None:
                    status.set_label("No display clipboard.")
                    return
                display.get_clipboard().set(text)
                status.set_label("Copied.")

            copy_btn.connect("clicked", copy)
            close_btn.connect("clicked", lambda *_args: window.close())
            buttons.append(copy_btn)
            buttons.append(close_btn)
            buttons.append(status)
            root.append(buttons)
            window.present()

    app = About()
    return app.run([sys.argv[0]])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voxforge system information dialog")
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
    except Exception as exc:
        print(text)
        print(f"GTK dialog failed ({exc}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
