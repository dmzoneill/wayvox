import Adw from "gi://Adw";
import Gtk from "gi://Gtk";
import { ExtensionPreferences } from "resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js";

import { openAbout, openVocabEditor } from "./utils.js";

export default class WhisperNpuPreferences extends ExtensionPreferences {
  fillPreferencesWindow(window) {
    const settings = this.getSettings();

    const page = new Adw.PreferencesPage();

    page.add(this._buildServerGroup(settings));
    page.add(this._buildOptionsGroup(settings));
    page.add(this._buildVocabularyGroup());
    page.add(this._buildAboutGroup(settings));

    window.add(page);
  }

  _buildServerGroup(settings) {
    const group = new Adw.PreferencesGroup({
      title: "Server Connection",
    });

    const hostEntry = new Gtk.Entry({ hexpand: true });
    hostEntry.set_text(settings.get_string("server-host"));
    hostEntry.connect("changed", () => {
      settings.set_string("server-host", hostEntry.get_text());
    });

    const hostRow = new Adw.ActionRow({ title: "Host" });
    hostRow.add_suffix(hostEntry);
    hostRow.activatable = false;
    group.add(hostRow);

    const portAdj = new Gtk.Adjustment({
      lower: 1024,
      upper: 65535,
      step_increment: 1,
      value: settings.get_int("server-port"),
    });
    const portSpin = new Gtk.SpinButton({
      adjustment: portAdj,
      numeric: true,
      valign: Gtk.Align.CENTER,
    });
    portSpin.connect("value-changed", () => {
      settings.set_int("server-port", portSpin.get_value_as_int());
    });

    const portRow = new Adw.ActionRow({ title: "Port" });
    portRow.add_suffix(portSpin);
    portRow.activatable = false;
    group.add(portRow);

    return group;
  }

  _buildOptionsGroup(settings) {
    const group = new Adw.PreferencesGroup({
      title: "Server Options",
    });

    const DEVICE_OPTIONS = ["auto", "NPU", "GPU", "CUDA", "CPU"];
    const deviceCombo = new Gtk.ComboBoxText({ valign: Gtk.Align.CENTER });
    for (const d of DEVICE_OPTIONS) {
      deviceCombo.append_text(d);
    }
    deviceCombo.set_active(
      Math.max(0, DEVICE_OPTIONS.indexOf(settings.get_string("device"))),
    );
    deviceCombo.connect("changed", () => {
      settings.set_string("device", deviceCombo.get_active_text());
    });

    const deviceRow = new Adw.ActionRow({ title: "Device" });
    deviceRow.add_suffix(deviceCombo);
    deviceRow.activatable = false;
    group.add(deviceRow);

    const backendCombo = new Gtk.ComboBoxText({ valign: Gtk.Align.CENTER });
    for (const b of ["openvino", "whisper-cpp"]) {
      backendCombo.append_text(b);
    }
    backendCombo.set_active(
      ["openvino", "whisper-cpp"].indexOf(settings.get_string("backend")),
    );
    backendCombo.connect("changed", () => {
      settings.set_string("backend", backendCombo.get_active_text());
    });

    const backendRow = new Adw.ActionRow({ title: "Backend" });
    backendRow.add_suffix(backendCombo);
    backendRow.activatable = false;
    group.add(backendRow);

    const HOTKEYS = [
      "KEY_RIGHTCTRL",
      "KEY_LEFTCTRL",
      "KEY_RIGHTALT",
      "KEY_LEFTALT",
      "KEY_RIGHTSHIFT",
      "KEY_LEFTSHIFT",
      "KEY_SCROLLLOCK",
      "KEY_PAUSE",
      "KEY_F23",
      "KEY_VOICECOMMAND",
      "KEY_ASSISTANT",
    ];
    const hotkeyCombo = new Gtk.ComboBoxText({ valign: Gtk.Align.CENTER });
    for (const k of HOTKEYS) {
      hotkeyCombo.append_text(k);
    }
    const currentHotkey = settings.get_string("hotkey");
    hotkeyCombo.set_active(Math.max(0, HOTKEYS.indexOf(currentHotkey)));
    hotkeyCombo.connect("changed", () => {
      settings.set_string("hotkey", hotkeyCombo.get_active_text());
    });

    const hotkeyRow = new Adw.ActionRow({ title: "Hotkey" });
    hotkeyRow.add_suffix(hotkeyCombo);
    hotkeyRow.activatable = false;
    group.add(hotkeyRow);

    const intervalAdj = new Gtk.Adjustment({
      lower: 0.5,
      upper: 30.0,
      step_increment: 0.5,
      value: settings.get_double("stream-interval"),
    });
    const intervalSpin = new Gtk.SpinButton({
      adjustment: intervalAdj,
      digits: 1,
      numeric: true,
      valign: Gtk.Align.CENTER,
    });
    intervalSpin.connect("value-changed", () => {
      settings.set_double("stream-interval", intervalSpin.get_value());
    });

    const intervalRow = new Adw.ActionRow({
      title: "Stream Interval (seconds)",
    });
    intervalRow.add_suffix(intervalSpin);
    intervalRow.activatable = false;
    group.add(intervalRow);

    const hfEntry = new Gtk.Entry({ hexpand: true });
    hfEntry.set_text(settings.get_string("hf-org"));
    hfEntry.connect("changed", () => {
      settings.set_string("hf-org", hfEntry.get_text());
    });

    const hfRow = new Adw.ActionRow({ title: "HuggingFace Organization" });
    hfRow.add_suffix(hfEntry);
    hfRow.activatable = false;
    group.add(hfRow);

    return group;
  }

  _buildVocabularyGroup() {
    const group = new Adw.PreferencesGroup({
      title: "Technical Vocabulary",
      description:
        "Bias Whisper toward computer terms and fix leftover mishears such as polar quest. Edits apply on the next transcription.",
    });

    const button = new Gtk.Button({
      label: "Edit Vocabulary...",
      valign: Gtk.Align.CENTER,
    });
    const row = new Adw.ActionRow({ title: "Terms and replacements" });
    row.add_suffix(button);
    row.activatable = false;
    button.connect("clicked", () => {
      if (!openVocabEditor(this.path))
        row.set_subtitle(
          "Editor not found. Expected vocab-editor.py in the Wayvox checkout.",
        );
    });
    group.add(row);
    return group;
  }

  _buildAboutGroup(settings) {
    const group = new Adw.PreferencesGroup({
      title: "About",
      description:
        "NPU, GPUs, the inference library, and versions from the running server.",
    });
    const button = new Gtk.Button({
      label: "About...",
      valign: Gtk.Align.CENTER,
    });
    const row = new Adw.ActionRow({ title: "System information" });
    row.add_suffix(button);
    row.activatable = false;
    button.connect("clicked", () => {
      if (
        !openAbout(
          this.path,
          settings.get_string("server-host"),
          settings.get_int("server-port"),
        )
      )
        row.set_subtitle(
          "About dialog not found. Expected about.py in the Wayvox checkout.",
        );
    });
    group.add(row);
    return group;
  }
}
