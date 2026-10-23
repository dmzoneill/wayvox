#!/usr/bin/env python3
"""Push-to-talk voice dictation for GNOME and Wayland.

Hold the hotkey to record, release to transcribe, and type the result.
evdev reads the keyboard. wl-copy with ydotool or wtype inserts text.

Usage:
    python3 push_to_talk.py [--key KEY_RIGHTCTRL] [--port 5000]
"""

import argparse
import asyncio
import logging

import dictation_audio
from dictation_audio import (
    AudioBuffer,
    auto_punctuate,
    find_keyboards,
    mute_other_streams,
    record_audio,
    stop_recording,
    transcribe_batch,
    transcribe_chunk,
    transcribe_stream,
    translate_text,
    trim_silence,
    unmute_streams,
)
from dictation_config import (
    CHANNELS,
    CONFIG_KEY_MAP,
    CONFIG_PATH,
    CORRECTION_WINDOW,
    HOLD_THRESHOLD,
    MODEL_SIZE_ORDER,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    apply_config,
    build_settings,
    is_correction_tap,
    load_config,
    watch_config,
)
from dictation_output import (
    ABBREVIATIONS,
    DBUS_BUS_NAMES,
    DBUS_INTERFACE,
    DBUS_OBJECT_PATH,
    DEFAULT_APP_CONTEXTS,
    RUNTIME,
    VOICE_COMMANDS,
    WORD_NUMS,
    TranscriptionHistory,
    abort_typing,
    clipboard_save,
    clipboard_write,
    execute_voice_command,
    format_dictation,
    get_focused_app,
    is_terminal_focused,
    load_app_contexts,
    notify,
    play_sound,
    send_keys,
    set_clipboard_mode,
    try_dbus_call,
    try_dbus_handoff,
    try_dbus_handoff_with_context,
    try_dbus_history_picker,
    type_text,
    type_text_async,
    type_text_clipboard,
)
from dictation_session import run_dictation

__all__ = (
    "ABBREVIATIONS",
    "AudioBuffer",
    "CHANNELS",
    "CONFIG_KEY_MAP",
    "CONFIG_PATH",
    "CORRECTION_WINDOW",
    "DBUS_BUS_NAMES",
    "DBUS_INTERFACE",
    "DBUS_OBJECT_PATH",
    "DEFAULT_APP_CONTEXTS",
    "HOLD_THRESHOLD",
    "MODEL_SIZE_ORDER",
    "RUNTIME",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "TranscriptionHistory",
    "VOICE_COMMANDS",
    "WORD_NUMS",
    "abort_typing",
    "apply_config",
    "auto_punctuate",
    "build_settings",
    "clipboard_save",
    "clipboard_write",
    "execute_voice_command",
    "find_keyboards",
    "format_dictation",
    "get_focused_app",
    "is_correction_tap",
    "is_terminal_focused",
    "load_app_contexts",
    "load_config",
    "mute_other_streams",
    "notify",
    "play_sound",
    "record_audio",
    "run_dictation",
    "send_keys",
    "set_clipboard_mode",
    "stop_recording",
    "transcribe_batch",
    "transcribe_chunk",
    "transcribe_stream",
    "translate_text",
    "trim_silence",
    "try_dbus_call",
    "try_dbus_handoff",
    "try_dbus_handoff_with_context",
    "try_dbus_history_picker",
    "type_text",
    "type_text_async",
    "type_text_clipboard",
    "unmute_streams",
    "watch_config",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ptt")


def _parse_args(argv=None):
    """Parse dictation client command-line arguments."""
    parser = argparse.ArgumentParser(description="Push-to-talk voice dictation")
    parser.add_argument(
        "--key",
        default="KEY_RIGHTCTRL",
        help="Hold key name (default: KEY_RIGHTCTRL)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Whisper server port (default: 5000)",
    )
    parser.add_argument(
        "--type-delay",
        type=int,
        default=4,
        help="ydotool inter-key delay in ms (default: 4)",
    )
    parser.add_argument(
        "--backend",
        choices=["openvino", "whisper-cpp"],
        default="openvino",
        help="Transcription backend (default: openvino)",
    )
    parser.add_argument(
        "--stream-interval",
        type=float,
        default=3.0,
        help="Seconds between streaming transcription updates (default: 3.0)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=int,
        default=-40,
        help="VAD silence threshold in dB (default: -40)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        default=False,
        help="Disable desktop notifications after transcription",
    )
    parser.add_argument(
        "--no-commands",
        action="store_true",
        default=False,
        help="Disable voice command detection",
    )
    parser.add_argument(
        "--recall-key",
        default="KEY_PAUSE",
        help="Key to show transcription history picker (default: KEY_PAUSE)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for multilingual models (e.g., 'de', 'fr')",
    )
    parser.add_argument(
        "--auto-punctuate",
        action="store_true",
        default=False,
        help="Post-process transcriptions with LLM for punctuation",
    )
    parser.add_argument(
        "--translate-to",
        default=None,
        help="Translate transcriptions to target language (e.g., 'French')",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        default=False,
        help="Disable audio feedback sounds",
    )
    parser.add_argument(
        "--no-formatting",
        action="store_true",
        default=False,
        help="Disable dictation formatting (numbers, abbreviations)",
    )
    parser.add_argument(
        "--mute-other-streams",
        action="store_true",
        default=False,
        help="Mute other recording streams while recording (e.g. video call mic)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Load settings and listen for the push-to-talk hotkey."""
    args = _parse_args(argv)
    if dictation_audio.aiohttp is None:
        log.error("aiohttp required: pip install aiohttp")
        return 1
    settings = build_settings(args)
    settings["port"] = args.port
    loaded = load_config()
    if loaded:
        apply_config(settings, loaded)
    asyncio.run(run_dictation(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
