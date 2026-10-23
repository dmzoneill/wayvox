#!/usr/bin/env python3
"""Edit Whisper technical terms and mishear replacements.

Opened from the GNOME or KDE tray. Writes ~/.config/wayvox/vocabulary.json.
The transcription server reloads that file on the next utterance.
"""

import sys

import vocab


def _buffer_text(buf):
    return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)


def main():
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        from gi.repository import Gtk
    except Exception as exc:
        print(f"GTK 4 is required to edit the vocabulary ({exc}).", file=sys.stderr)
        print(f"Edit {vocab.default_path()} directly.", file=sys.stderr)
        return 1

    class Editor(Gtk.Application):
        def __init__(self):
            super().__init__(application_id="one.whisper.npu.VocabEditor")

        def do_activate(self):
            window = Gtk.ApplicationWindow(application=self, title="Whisper Vocabulary")
            window.set_default_size(760, 700)

            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            root.set_margin_top(12)
            root.set_margin_bottom(12)
            root.set_margin_start(12)
            root.set_margin_end(12)
            window.set_child(root)

            header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            header.append(Gtk.Label(label="Bias Whisper and fix mishears", xalign=0, hexpand=True))
            self.enabled = Gtk.Switch(valign=Gtk.Align.CENTER)
            header.append(self.enabled)
            root.append(header)

            hint = Gtk.Label(
                label=(
                    "Terms are one per line. The top of the list is sent as the Whisper initial prompt. "
                    "The full list is sent as hotwords. Put the words Whisper gets wrong first.\n"
                    "Replacements are Python regex, one per line:  \\bpolar quests?\\b => pull request\n"
                    "Add [cs] at the end of a line for a case-sensitive pattern. "
                    "Save takes effect on the next transcription."
                ),
                wrap=True,
                xalign=0,
                justify=Gtk.Justification.LEFT,
            )
            root.append(hint)

            root.append(Gtk.Label(label="Prompt preamble", xalign=0))
            self.preamble = Gtk.Entry()
            root.append(self.preamble)

            self.terms_buf = Gtk.TextBuffer()
            self.rules_buf = Gtk.TextBuffer()
            root.append(Gtk.Label(label="Terms", xalign=0))
            root.append(self._scrolled(self.terms_buf))
            root.append(Gtk.Label(label="Replacements", xalign=0))
            root.append(self._scrolled(self.rules_buf))

            buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            save_btn = Gtk.Button(label="Save")
            save_btn.add_css_class("suggested-action")
            reset_btn = Gtk.Button(label="Reset to defaults")
            close_btn = Gtk.Button(label="Close")
            save_btn.connect("clicked", self._save)
            reset_btn.connect("clicked", lambda *_: self._load(vocab.default_config(), "Defaults loaded. Save to keep them."))
            close_btn.connect("clicked", lambda *_: window.close())
            buttons.append(save_btn)
            buttons.append(reset_btn)
            buttons.append(close_btn)
            root.append(buttons)

            self.status = Gtk.Label(label="", xalign=0, wrap=True)
            root.append(self.status)

            self._load(vocab.load(), f"Editing {vocab.default_path()}")
            window.present()

        def _scrolled(self, buf):
            view = Gtk.TextView(buffer=buf, monospace=True, wrap_mode=Gtk.WrapMode.NONE)
            view.set_left_margin(6)
            view.set_top_margin(6)
            scrolled = Gtk.ScrolledWindow(vexpand=True)
            scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            scrolled.set_child(view)
            scrolled.set_size_request(-1, 160)
            return scrolled

        def _load(self, config, status):
            self.enabled.set_active(bool(config.get("enabled", True)))
            self.preamble.set_text(config.get("preamble") or vocab.DEFAULT_PREAMBLE)
            self.terms_buf.set_text("\n".join(config.get("terms") or []))
            self.rules_buf.set_text(vocab.rules_to_text(config.get("replacements") or []))
            self.status.set_label(status)

        def _save(self, *_args):
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
            self.status.set_label(
                f"Saved {len(config['terms'])} terms and {len(config['replacements'])} rules to {vocab.default_path()}"
            )

    app = Editor()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
