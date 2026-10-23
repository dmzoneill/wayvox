"""Load Wayvox dictation settings from the environment and JSON config."""

import asyncio
import json
import logging
import os

from wayvox_paths import CONFIG_DIR, LEGACY_CONFIG_DIR

log = logging.getLogger("ptt")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
HOLD_THRESHOLD = 1.5
CORRECTION_WINDOW = 3.0


def is_correction_tap(hold_duration, last_wav_bytes, now, last_transcription_end):
    """True for a short tap that should re-transcribe the previous utterance."""
    quick_tap = hold_duration < 0.5 and last_wav_bytes
    recent = (now - last_transcription_end) < CORRECTION_WINDOW
    return quick_tap and recent


MODEL_SIZE_ORDER = ["tiny", "base", "small", "medium", "large"]

CONFIG_PATH = str(CONFIG_DIR / "settings.json")
LEGACY_CONFIG_PATH = str(LEGACY_CONFIG_DIR / "settings.json")

CONFIG_KEY_MAP = {
    "hotkey": ("key", str, "KEY_RIGHTCTRL"),
    "backend": ("backend", str, "openvino"),
    "recall-key": ("recall_key", str, "KEY_PAUSE"),
    "language": ("language", str, ""),
    "vad-threshold": ("vad_threshold", int, -40),
    "stream-interval": ("stream_interval", float, 3.0),
    "auto-punctuate": ("auto_punctuate", bool, False),
    "translate-to": ("translate_to", str, ""),
    "audio-feedback-enabled": ("sound", bool, True),
    "dictation-formatting-enabled": ("formatting", bool, True),
    "mute-other-streams": ("mute_streams", bool, False),
    "voice-commands-enabled": ("voice_commands", bool, True),
    "notifications-enabled": ("notifications", bool, True),
    "clipboard-mode": ("clipboard_mode", bool, False),
}


def load_config(path=CONFIG_PATH):
    """Load settings from JSON config file. Returns empty dict on failure."""
    missing_current = path == CONFIG_PATH and not os.path.exists(path)
    has_legacy = os.path.exists(LEGACY_CONFIG_PATH)
    if missing_current and has_legacy:
        path = LEGACY_CONFIG_PATH
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _env_text(name):
    """Return an environment variable, or an empty string when unset."""
    return os.environ.get(name, "")


def _env_enabled(name):
    """True when an environment variable is the string ``1``."""
    return _env_text(name) == "1"


def build_settings(args):
    """Build initial settings dict from CLI args and env vars."""
    language = args.language or _env_text("WHISPER_LANGUAGE") or ""
    translate_to = args.translate_to or _env_text("WHISPER_TRANSLATE_TO") or ""
    auto_punctuate = args.auto_punctuate or _env_enabled("WHISPER_AUTO_PUNCTUATE")
    sound_disabled = args.no_sound or _env_enabled("WHISPER_NO_SOUND")
    formatting_disabled = args.no_formatting or _env_enabled("WHISPER_NO_FORMATTING")
    mute_streams = args.mute_other_streams or _env_enabled("WHISPER_MUTE_STREAMS")
    return {
        "key": args.key,
        "backend": args.backend,
        "recall_key": args.recall_key,
        "language": language,
        "vad_threshold": args.vad_threshold,
        "stream_interval": args.stream_interval,
        "type_delay": args.type_delay,
        "auto_punctuate": auto_punctuate,
        "translate_to": translate_to,
        "sound": not sound_disabled,
        "formatting": not formatting_disabled,
        "mute_streams": mute_streams,
        "voice_commands": not args.no_commands,
        "notifications": not args.no_notify,
        "clipboard_mode": False,
    }


def apply_config(settings, config):
    """Apply config file values to settings dict. Returns True if anything changed."""
    changed = False
    for json_key, (internal_key, coerce, _default) in CONFIG_KEY_MAP.items():
        if json_key in config:
            try:
                new_val = coerce(config[json_key])
            except (ValueError, TypeError):
                continue
            if settings.get(internal_key) != new_val:
                log.info(
                    "Setting %s: %s -> %s",
                    internal_key,
                    settings.get(internal_key),
                    new_val,
                )
                settings[internal_key] = new_val
                changed = True
    return changed


async def watch_config(settings, on_change, path=CONFIG_PATH, interval=2.0):
    """Poll config file for changes and apply them to settings dict."""
    last_mtime = 0.0
    try:
        last_mtime = os.path.getmtime(path)
    except OSError:
        pass

    while True:
        await asyncio.sleep(interval)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime <= last_mtime:
            continue
        last_mtime = mtime
        config = load_config(path)
        if not config:
            continue
        if apply_config(settings, config):
            on_change(settings)
