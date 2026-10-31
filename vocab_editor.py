#!/usr/bin/env python3
"""Edit Whisper technical terms and mishear replacements.

Opened from the GNOME or KDE tray. Writes ~/.config/wayvox/vocabulary.json.
The transcription server reloads that file on the next utterance.
"""

import sys
from typing import Any

import vocab

try:
    import gi

    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk
except (ImportError, ValueError) as exc:
    gi = None
    Gtk = None
    _GTK_ERROR: BaseException | None = exc
else:
    _GTK_ERROR = None

_EditorBase: Any = Gtk.Application if Gtk is not None else object


def _editor_root():
    root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    margins = {
        "set_margin_top": 12,
        "set_margin_bottom": 12,
        "set_margin_start": 12,
        "set_margin_end": 12,
    }
    for name, value in margins.items():
        getattr(root, name)(value)
    return root


def _scrolled_view(buf):
    view = Gtk.TextView(buffer=buf, monospace=True, wrap_mode=Gtk.WrapMode.NONE)
    view.set_left_margin(6)
    view.set_top_margin(6)
    scrolled = Gtk.ScrolledWindow(vexpand=True)
    scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
    scrolled.set_child(view)
    scrolled.set_size_request(-1, 160)
    return scrolled


class Editor(_EditorBase):
    """GTK editor for the Wayvox technical vocabulary file."""

    def __init__(self):
        """Create the vocabulary editor application."""
        super().__init__(application_id="one.whisper.npu.VocabEditor")
        self.enabled = None
        self.preamble = None
        self.terms_buf = None
        self.rules_buf = None
        self.status = None

    def editor_hint(self):
        """Return the instruction shown above the term and replacement editors."""
        return (
            "Terms are one per line. The top of the list is sent as the Whisper "
            "initial prompt. The full list is sent as hotwords. Put the words "
            "Whisper gets wrong first.\n"
            "Replacements are Python regex, one per line:  "
            "\\bpolar quests?\\b => pull request\n"
            "Add [cs] at the end of a line for a case-sensitive pattern. "
            "Save takes effect on the next transcription."
        )

    def _header(self):
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header.append(
            Gtk.Label(
                label="Bias Whisper and fix mishears",
                xalign=0,
                hexpand=True,
            ),
        )
        self.enabled = Gtk.Switch(valign=Gtk.Align.CENTER)
        header.append(self.enabled)
        return header

    def _form(self, root):
        root.append(Gtk.Label(label="Prompt preamble", xalign=0))
        self.preamble = Gtk.Entry()
        root.append(self.preamble)
        self.terms_buf = Gtk.TextBuffer()
        self.rules_buf = Gtk.TextBuffer()
        root.append(Gtk.Label(label="Terms", xalign=0))
        root.append(_scrolled_view(self.terms_buf))
        root.append(Gtk.Label(label="Replacements", xalign=0))
        root.append(_scrolled_view(self.rules_buf))

    def _buttons(self, window):
        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        save_btn = Gtk.Button(label="Save")
        save_btn.add_css_class("suggested-action")
        reset_btn = Gtk.Button(label="Reset to defaults")
        close_btn = Gtk.Button(label="Close")
        save_btn.connect("clicked", self.save_vocabulary)
        reset_btn.connect("clicked", self._reset_to_defaults)
        close_btn.connect("clicked", lambda *_args: window.close())
        buttons.append(save_btn)
        buttons.append(reset_btn)
        buttons.append(close_btn)
        return buttons

    def _reset_to_defaults(self, *_args):
        self.load_form(
            vocab.default_config(),
            "Defaults loaded. Save to keep them.",
        )

    def load_form(self, config, status):
        """Fill the form from a vocabulary config and set the status line."""
        self.enabled.set_active(bool(config.get("enabled", True)))
        self.preamble.set_text(config.get("preamble") or vocab.DEFAULT_PREAMBLE)
        self.terms_buf.set_text("\n".join(config.get("terms") or []))
        self.rules_buf.set_text(vocab.rules_to_text(config.get("replacements") or []))
        self.status.set_label(status)

    def save_vocabulary(self, *_args):
        """Write the form to the vocabulary file and update the status line."""
        rules, parse_errors = vocab.parse_rules(_buffer_text(self.rules_buf))
        config = {
            "enabled": self.enabled.get_active(),
            "preamble": self.preamble.get_text().strip(),
            "terms": vocab.parse_terms(_buffer_text(self.terms_buf)),
            "replacements": rules,
        }
        errors = parse_errors + vocab.validate_config(config)
        if errors:
            self.status.set_label("Not saved. " + errors[0])
            return
        try:
            vocab.save(config)
        except (OSError, ValueError) as exc:
            self.status.set_label(f"Not saved. {exc}")
            return
        term_count = len(config["terms"])
        rule_count = len(config["replacements"])
        path = vocab.default_path()
        saved = f"Saved {term_count} terms and {rule_count} rules to {path}"
        self.status.set_label(saved)

    def do_activate(self):
        """Build the vocabulary window, load the current file, and present it."""
        window = Gtk.ApplicationWindow(application=self, title="Whisper Vocabulary")
        window.set_default_size(760, 700)
        root = _editor_root()
        window.set_child(root)
        root.append(self._header())
        hint = Gtk.Label(
            label=self.editor_hint(),
            wrap=True,
            xalign=0,
            justify=Gtk.Justification.LEFT,
        )
        root.append(hint)
        self._form(root)
        root.append(self._buttons(window))
        self.status = Gtk.Label(label="", xalign=0, wrap=True)
        root.append(self.status)
        self.load_form(vocab.load(), f"Editing {vocab.default_path()}")
        window.present()


def _buffer_text(buf):
    return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)


def main():
    """Open the vocabulary editor, or explain how to edit the file by hand."""
    if Gtk is None:
        print(
            f"GTK 4 is required to edit the vocabulary ({_GTK_ERROR}).",
            file=sys.stderr,
        )
        print(f"Edit {vocab.default_path()} directly.", file=sys.stderr)
        return 1
    app = Editor()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
