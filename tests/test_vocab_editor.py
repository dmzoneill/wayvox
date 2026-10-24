"""Tests for the vocabulary editor in vocab_editor.py."""

from unittest.mock import MagicMock, patch

import pytest
from gtk_support import import_with_fake_gtk, import_without_gtk


@pytest.fixture(name="editor_gtk")
def fixture_editor_gtk():
    """Import vocab_editor.py against fake GTK modules."""
    return import_with_fake_gtk("vocab_editor")


@pytest.fixture(name="editor")
def fixture_editor(editor_gtk):
    """Return an activated Editor with its vocabulary file mocked."""
    module = editor_gtk[0]
    config = {"enabled": False, "terms": ["kubectl"], "replacements": []}
    with (
        patch.object(module.vocab, "load", return_value=config),
        patch.object(module.vocab, "default_path", return_value="/tmp/v.json"),
    ):
        app = module.Editor()
        app.do_activate()
    app.status = MagicMock()
    return module, app


class TestEditor:
    """Loading, saving, and resetting the form."""

    def test_activate_loads_file(self, editor_gtk):
        """Activate loads file."""
        module, gtk, _gdk = editor_gtk
        with (
            patch.object(module.vocab, "load", return_value={}),
            patch.object(module.vocab, "default_path", return_value="/v"),
        ):
            app = module.Editor()
            app.do_activate()
        assert app.terms_buf.text == ""
        app.status.set_label.assert_called_with("Editing /v")
        gtk.ApplicationWindow.return_value.present.assert_called_once()
        assert "Python regex" in app.editor_hint()

    def test_load_form(self, editor):
        """Load form."""
        _module, app = editor
        assert app.terms_buf.text == "kubectl"
        app.enabled.set_active.assert_called_with(False)

    def test_save(self, editor):
        """Save."""
        module, app = editor
        app.terms_buf.set_text("alpha\nbeta")
        app.rules_buf.set_text("")
        app.enabled.get_active.return_value = True
        app.preamble.get_text.return_value = " intro "
        with (
            patch.object(module.vocab, "save") as save,
            patch.object(module.vocab, "default_path", return_value="/v"),
        ):
            app.save_vocabulary()
        saved = save.call_args[0][0]
        assert saved["terms"] == ["alpha", "beta"]
        assert saved["preamble"] == "intro"
        app.status.set_label.assert_called_with("Saved 2 terms and 0 rules to /v")

    def test_save_rejects_invalid(self, editor):
        """Save rejects invalid."""
        module, app = editor
        app.enabled.get_active.return_value = True
        app.preamble.get_text.return_value = ""
        with (
            patch.object(module.vocab, "parse_rules", return_value=([], ["bad rule"])),
            patch.object(module.vocab, "save") as save,
        ):
            app.save_vocabulary()
        save.assert_not_called()
        app.status.set_label.assert_called_with("Not saved. bad rule")

    def test_save_reports_os_error(self, editor):
        """Save reports os error."""
        module, app = editor
        app.enabled.get_active.return_value = True
        app.preamble.get_text.return_value = ""
        with patch.object(module.vocab, "save", side_effect=OSError("read-only")):
            app.save_vocabulary()
        app.status.set_label.assert_called_with("Not saved. read-only")

    def test_reset_button_loads_defaults(self, editor_gtk, editor):
        """Reset button loads defaults."""
        _module, app = editor
        gtk = editor_gtk[1]
        connects = gtk.Button.return_value.connect.call_args_list
        handlers = {
            getattr(call.args[1], "__name__", ""): call.args[1] for call in connects
        }
        reset = handlers["_reset_to_defaults"]
        reset(None)
        app.status.set_label.assert_called_with("Defaults loaded. Save to keep them.")


class TestMain:
    """Entry point with and without GTK."""

    def test_runs_app(self, editor_gtk):
        """Runs app."""
        assert editor_gtk[0].main() == 0

    def test_without_gtk(self, capsys):
        """Without gtk."""
        module = import_without_gtk("vocab_editor")
        assert module.main() == 1
        assert "GTK 4 is required" in capsys.readouterr().err
