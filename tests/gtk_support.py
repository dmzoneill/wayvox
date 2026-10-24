"""Fake GTK 4 modules so the tray dialogs can be tested without a display."""

import importlib
import sys
from unittest.mock import MagicMock, patch


class FakeApplication:
    """Stand-in for Gtk.Application that records run() calls."""

    def __init__(self, application_id=None):
        self.application_id = application_id
        self.run_args = None

    def run(self, argv):
        """Record *argv* and report success."""
        self.run_args = argv
        return 0

    def get_application_id(self):
        """Return the id passed to the constructor."""
        return self.application_id


class FakeTextBuffer:
    """Stand-in for Gtk.TextBuffer that keeps its text."""

    def __init__(self):
        self.text = ""

    def set_text(self, text):
        """Replace the buffer text."""
        self.text = text

    def get_start_iter(self):
        """Return a start marker."""
        return 0

    def get_end_iter(self):
        """Return an end marker."""
        return len(self.text)

    def get_text(self, start, end, _hidden):
        """Return the text between two markers."""
        return self.text[start:end]


def fake_gtk():
    """Return a MagicMock Gtk with real classes where code depends on them."""
    gtk = MagicMock(name="Gtk")
    gtk.Application = FakeApplication
    gtk.TextBuffer = FakeTextBuffer
    return gtk


def import_with_fake_gtk(module_name):
    """Import *module_name* against fake gi modules; return (module, Gtk, Gdk)."""
    gtk = fake_gtk()
    gdk = MagicMock(name="Gdk")
    glib = MagicMock(name="GLib")
    glib.Error = type("GLibError", (Exception,), {})
    repository = MagicMock(Gtk=gtk, Gdk=gdk, GLib=glib)
    modules = {"gi": MagicMock(), "gi.repository": repository}
    with patch.dict(sys.modules, modules):
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    return module, gtk, gdk


def import_without_gtk(module_name):
    """Import *module_name* while gi is unavailable."""
    with patch.dict(sys.modules, {"gi": None}):
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
