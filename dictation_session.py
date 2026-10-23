"""Hold, toggle, and correction modes for the dictation client."""

import asyncio
import io
import json
import logging
import os
import subprocess
import wave
from dataclasses import dataclass, field

from dictation_audio import (
    auto_punctuate,
    find_keyboards,
    mute_other_streams,
    record_audio,
    stop_recording,
    transcribe_batch,
    transcribe_chunk,
    translate_text,
    trim_silence,
    unmute_streams,
)
from dictation_config import (
    CHANNELS,
    HOLD_THRESHOLD,
    MODEL_SIZE_ORDER,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    is_correction_tap,
    watch_config,
)
from dictation_deps import aiohttp, ecodes, evdev
from dictation_output import (
    TranscriptionHistory,
    abort_typing,
    execute_voice_command,
    format_dictation,
    get_focused_app,
    load_app_contexts,
    notify,
    play_sound,
    set_clipboard_mode,
    try_dbus_handoff,
    try_dbus_handoff_with_context,
    try_dbus_history_picker,
    type_text_async,
)

log = logging.getLogger("ptt")

_MIN_WAV_BYTES = 44


@dataclass
class TakeState:
    """Audio and typing state for the recording that is in progress."""

    recording: object = None
    streaming_task: object = None
    typed_char_count: int = 0
    streamed_text: str = ""
    key_down_time: float = 0.0
    mode: str | None = None
    muted_streams: list = field(default_factory=list)


@dataclass
class PreviousTake:
    """The last delivered utterance, kept so a quick tap can correct it."""

    wav_bytes: object = None
    typed_text: str = ""
    transcription_end: float = 0.0
    used_dbus: bool = False


@dataclass
class DictationState:
    """Mutable state for one running dictation process."""

    settings: dict
    derived: dict = field(default_factory=dict)
    history: TranscriptionHistory | None = None
    app_contexts: dict = field(default_factory=dict)
    take: TakeState = field(default_factory=TakeState)
    previous: PreviousTake = field(default_factory=PreviousTake)


def make_state(settings):
    """Build a session with a history database and the per-app contexts."""
    return DictationState(
        settings=settings,
        history=TranscriptionHistory(),
        app_contexts=load_app_contexts(),
    )


def refresh_derived(state):
    """Recompute key codes and the backend port from the current settings."""
    settings = state.settings
    new_key = getattr(ecodes, settings["key"], None)
    if new_key:
        state.derived["target_key"] = new_key
    if settings.get("backend") == "openvino":
        state.derived["backend_port"] = int(settings.get("port", 5000))
    else:
        state.derived["backend_port"] = 5001
    state.derived["recall_key"] = getattr(ecodes, settings["recall_key"], None)
    set_clipboard_mode(settings.get("clipboard_mode", False))


def _wav_bytes(raw_audio):
    """Wrap raw PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.Wave_write(buffer) as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(raw_audio)
    return buffer.getvalue()


def _common_prefix(left, right):
    """Return how many leading characters *left* and *right* share."""
    shared = 0
    for old, new in zip(left, right):
        if old != new:
            break
        shared += 1
    return shared


def backspace_count(count):
    """Send *count* backspace keys on Wayland or X11."""
    if count <= 0:
        return
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        _backspace_wayland(count)
        return
    subprocess.run(
        ["xdotool", "key", "--repeat", str(count), "BackSpace"],
        check=True,
    )


def _backspace_wayland(count):
    """Press BackSpace *count* times with ydotool."""
    try:
        for _ in range(count):
            subprocess.run(
                ["ydotool", "key", "14:1", "14:0"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.debug("ydotool backspace failed")


async def _type_difference(previous, text, delay_ms):
    """Backspace the changed suffix and type the new one."""
    shared = _common_prefix(previous, text)
    erase = len(previous) - shared
    suffix = text[shared:]
    if erase > 0:
        backspace_count(erase)
    if suffix:
        await type_text_async(suffix, delay_ms=delay_ms)
    return erase, suffix


def _release_muted(state):
    """Restore recording streams muted for this utterance."""
    if not state.take.muted_streams:
        return
    unmute_streams(state.take.muted_streams)
    state.take.muted_streams = []


async def streaming_loop(state):
    """Type partial whisper.cpp text while a toggle recording is still open."""
    settings = state.settings
    interval = settings["stream_interval"]
    delay_ms = settings["type_delay"]
    port = state.derived["backend_port"]
    await asyncio.sleep(interval)
    while state.take.recording.proc.poll() is None:
        await _stream_once(state, port, delay_ms)
        await asyncio.sleep(interval)


async def _stream_once(state, port, delay_ms):
    """Transcribe the current buffer and type only the changed suffix."""
    raw_audio = state.take.recording.snapshot()
    raw_audio = trim_silence(raw_audio, threshold_db=state.settings["vad_threshold"])
    if len(raw_audio) < SAMPLE_RATE * SAMPLE_WIDTH:
        await asyncio.sleep(0.5)
        return
    wav_bytes = _wav_bytes(raw_audio)
    duration = len(raw_audio) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    if duration < 0.5:
        await asyncio.sleep(0.5)
        return
    log.info("Streaming chunk: %.1fs of audio", duration)
    text = await transcribe_chunk(
        wav_bytes,
        port,
        language=state.settings["language"] or None,
    )
    if not text or text == state.take.streamed_text:
        return
    previous = state.take.streamed_text
    erased, suffix = await _type_difference(previous, text, delay_ms)
    state.take.typed_char_count += len(suffix) - erased
    log.info("Live (-%d +%d): %s", erased, len(suffix), text[:80])
    state.take.streamed_text = text


async def _cancel_streaming(state):
    """Stop the live streaming task if one is running."""
    task = state.take.streaming_task
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    state.take.streaming_task = None


def _capture_wav(state):
    """Stop parec and return WAV bytes, or None when the take is empty."""
    wav_bytes = stop_recording(
        state.take.recording,
        vad_threshold=state.settings["vad_threshold"],
    )
    state.take.recording = None
    _release_muted(state)
    return wav_bytes


def _wav_duration(wav_bytes):
    """Return the audio duration in seconds for a WAV payload."""
    payload = max(0, len(wav_bytes) - _MIN_WAV_BYTES)
    return payload / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)


async def _finalize_toggle(state, wav_bytes, port, delay_ms):
    """Apply the final whisper.cpp diff for a toggle recording."""
    text = await transcribe_chunk(
        wav_bytes,
        port,
        language=state.settings["language"] or None,
    )
    if text:
        previous = state.take.streamed_text
        erased, suffix = await _type_difference(previous, text, delay_ms)
        log.info("Final (-%d +%d): %s", erased, len(suffix), text[:80])
    state.take.typed_char_count = 0
    state.take.streamed_text = ""
    state.take.mode = None


async def _transcribe_hold(state, wav_bytes, port):
    """Transcribe a finished hold-mode recording."""
    language = state.settings["language"] or None
    if state.settings["backend"] == "whisper-cpp":
        return await transcribe_chunk(wav_bytes, port, language=language)
    return await transcribe_batch(wav_bytes, port, language=language)


async def _postprocess(state, text, port):
    """Format, punctuate, and translate *text* when those options are on."""
    settings = state.settings
    if settings["formatting"]:
        text = format_dictation(text)
    if settings["auto_punctuate"]:
        text = await auto_punctuate(text, port)
    if settings["translate_to"]:
        original = text
        text = await translate_text(text, settings["translate_to"], port)
        log.info("Translated: %s -> %s", original[:40], text[:40])
    return text


async def _deliver(state, text, delay_ms):
    """Hand text to the desktop shell, or type it when no shell accepts it."""
    app_id, _window_title = get_focused_app()
    context = state.app_contexts.get(app_id, {})
    if context.get("tones"):
        payload = json.dumps({"tones": context["tones"], "app": app_id})
        if try_dbus_handoff_with_context(text, payload):
            log.info("Context-aware handoff (%s): %s", app_id, text[:80])
            return True
    if try_dbus_handoff(text):
        log.info("Handed off to Language Buddy: %s", text[:80])
        return True
    log.info("Typing: %s", text[:80])
    await type_text_async(text, delay_ms=delay_ms)
    return False


async def finalize_recording(state, port, delay_ms):
    """Stop the current take, transcribe it, and type or hand off the text."""
    await _cancel_streaming(state)
    wav_bytes = _capture_wav(state)
    if wav_bytes is None:
        log.info("No speech detected (silence only)")
        state.take.typed_char_count = 0
        state.take.mode = None
        return
    duration = _wav_duration(wav_bytes)
    if duration < 0.3:
        log.info("Too short (%.1fs), skipping", duration)
        state.take.typed_char_count = 0
        state.take.mode = None
        return
    log.info("Recorded %.1fs of audio", duration)
    backend = state.settings["backend"]
    cpp_toggle = backend == "whisper-cpp" and state.take.mode == "toggle"
    if cpp_toggle:
        await _finalize_toggle(state, wav_bytes, port, delay_ms)
        return
    await _finish_hold(state, wav_bytes, port, delay_ms)


async def _finish_hold(state, wav_bytes, port, delay_ms):
    """Transcribe a hold-mode take and deliver the text."""
    log.info("Transcribing...")
    text = await _transcribe_hold(state, wav_bytes, port)
    if not text:
        log.info("No transcription returned")
        state.take.mode = None
        return
    text = text.strip()
    if state.settings["sound"]:
        play_sound("message")
    text = await _postprocess(state, text, port)
    state.history.add(text)
    if _handle_voice_command(state, text):
        return
    state.previous.used_dbus = await _deliver(state, text, delay_ms)
    _remember_take(state, wav_bytes, text)


def _handle_voice_command(state, text):
    """Run a voice command and return True when the text was consumed."""
    if not state.settings["voice_commands"]:
        return False
    if not execute_voice_command(text):
        return False
    log.info("Voice command: %s", text[:80])
    if state.settings["notifications"]:
        notify("Voice Command", text[:80])
    state.take.mode = None
    return True


def _remember_take(state, wav_bytes, text):
    """Keep the last take so a quick tap can correct it."""
    if state.settings["notifications"]:
        notify("Whisper", text[:200])
    if state.settings["sound"]:
        play_sound("complete")
    state.previous.wav_bytes = wav_bytes
    state.previous.typed_text = text
    state.previous.transcription_end = asyncio.get_event_loop().time()
    state.take.mode = None


def _http_errors():
    """Errors a correction HTTP call is allowed to swallow."""
    found = [
        asyncio.TimeoutError,
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ]
    if aiohttp is not None:
        found.insert(0, aiohttp.ClientError)
    return tuple(found)


async def _fetch_json(url):
    """GET *url* and return the JSON object, or None on failure."""
    if aiohttp is None:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
    except _http_errors():
        return None


def _model_rank(name):
    """Return the size rank of a model name, or -1 when it is unknown."""
    lowered = (name or "").lower()
    for index, size in enumerate(MODEL_SIZE_ORDER):
        if size in lowered:
            return index
    return -1


def _larger_model(models, current_name):
    """Pick the largest listed model that is bigger than *current_name*."""
    best_rank = _model_rank(current_name)
    larger = None
    for name in models:
        rank = _model_rank(name)
        if rank > best_rank:
            larger = name
            best_rank = rank
    return larger


async def _retranscribe(state, model_name):
    """POST the last WAV to a named model and return the new text."""
    port = state.derived["backend_port"]
    url = f"http://127.0.0.1:{port}/transcribe/{model_name}"
    if state.settings["language"]:
        url += f"?language={state.settings['language']}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=state.previous.wav_bytes) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get("text", "").strip()
    except _http_errors():
        return None


async def correct_last_transcription(state):
    """Re-transcribe the previous take with a larger model."""
    if not state.previous.wav_bytes or not state.previous.typed_text:
        state.take.mode = None
        return
    if state.previous.used_dbus:
        log.info("Correction not available after Language Buddy handoff")
        notify("Whisper", "Correction not available after Language Buddy")
        state.take.mode = None
        return
    port = state.derived["backend_port"]
    listing = await _fetch_json(f"http://127.0.0.1:{port}/models")
    if listing is None:
        state.take.mode = None
        return
    current = await _fetch_json(f"http://127.0.0.1:{port}/model/default")
    current_name = "" if current is None else current.get("model", "")
    larger = _larger_model(listing.get("models", []), current_name)
    if not larger:
        log.info("No larger model available for correction")
        notify("Whisper", "No larger model available")
        state.take.mode = None
        return
    await _apply_correction(state, larger)


async def _apply_correction(state, model_name):
    """Replace the previously typed text when a larger model disagrees."""
    log.info("Correction: re-transcribing with %s", model_name)
    new_text = await _retranscribe(state, model_name)
    if new_text is None:
        state.take.mode = None
        return
    if new_text and new_text != state.previous.typed_text:
        backspace_count(len(state.previous.typed_text))
        await type_text_async(new_text, delay_ms=state.settings["type_delay"])
        log.info("Corrected: %s -> %s", state.previous.typed_text[:40], new_text[:40])
        if state.settings["notifications"]:
            notify("Correction", new_text[:200])
        state.previous.typed_text = new_text
    else:
        log.info("Same result with larger model")
        if state.settings["notifications"]:
            notify("Whisper", "Same result with larger model")
    state.take.mode = None


async def pump_events(device, event_queue):
    """Forward one keyboard's events until the device disconnects."""
    try:
        async for event in device.async_read_loop():
            escape = event.type == ecodes.EV_KEY and event.code == ecodes.KEY_ESC
            if escape and event.value == 1:
                abort_typing()
            await event_queue.put(event)
    except OSError:
        log.warning("Keyboard %s disconnected", device.name)
    finally:
        try:
            device.close()
        except OSError:
            log.debug("Keyboard %s already closed", device.name)


async def watch_keyboards(state, initial_devices, event_queue):
    """Keep a reader per keyboard and rescan so KVM switches reconnect."""
    pumps = {}
    for device in initial_devices:
        pumps[device.path] = asyncio.create_task(pump_events(device, event_queue))
    while True:
        await asyncio.sleep(2.0)
        _drop_finished(pumps)
        _attach_new_keyboards(state, pumps, event_queue)


def _drop_finished(pumps):
    """Forget keyboard tasks that already exited."""
    for path, task in list(pumps.items()):
        if task.done():
            del pumps[path]


def _attach_new_keyboards(state, pumps, event_queue):
    """Open keyboards that appeared since the last scan."""
    found = find_keyboards(exclude=pumps.keys(), target_key_name=state.settings["key"])
    for device in found:
        log.info("Keyboard %s connected (%s)", device.name, device.path)
        pumps[device.path] = asyncio.create_task(pump_events(device, event_queue))


def _copy_last_transcript(state):
    """Copy the newest history row when the shell has no history picker."""
    items = state.history.recent(1)
    if not items:
        return
    try:
        subprocess.run(
            ["wl-copy", items[0][0]],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        notify("Whisper", "Last transcription copied to clipboard")
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.debug("wl-copy failed while copying the last transcript")


async def _on_recall(state, key_event):
    """Show the history picker, or copy the last transcript."""
    recall = state.derived["recall_key"]
    pressed = recall and key_event.scancode == recall
    if not pressed or key_event.keystate != key_event.key_down:
        return False
    if not try_dbus_history_picker(state.history):
        _copy_last_transcript(state)
    return True


def _start_recording(state):
    """Begin a new take and mute other capture streams when asked."""
    state.take.key_down_time = asyncio.get_event_loop().time()
    state.take.mode = None
    state.take.typed_char_count = 0
    state.take.streamed_text = ""
    log.info("Recording...")
    if state.settings["sound"]:
        play_sound("message-new-instant")
    state.take.recording = record_audio()
    if state.settings["mute_streams"]:
        state.take.muted_streams = mute_other_streams(state.take.recording.proc.pid)


async def _on_key_down(state):
    """Start a recording, or finish a toggle recording on the second tap."""
    if state.take.recording is None:
        _start_recording(state)
        return
    if state.take.mode == "toggle":
        log.info("Toggle stop")
        await finalize_recording(
            state,
            state.derived["backend_port"],
            state.settings["type_delay"],
        )


def _stop_ignored_take(state):
    """Discard the short tap that only selected correction mode."""
    if state.take.recording:
        threshold = state.settings["vad_threshold"]
        stop_recording(state.take.recording, vad_threshold=threshold)
        state.take.recording = None
    _release_muted(state)


async def _enter_toggle(state, hold_duration):
    """Switch a short tap into toggle mode and start live streaming."""
    state.take.mode = "toggle"
    log.info("Toggle mode — tap again to stop (held %.1fs)", hold_duration)
    if state.settings["backend"] != "whisper-cpp":
        return
    state.take.streaming_task = asyncio.create_task(streaming_loop(state))


async def _on_key_up(state):
    """Choose hold, toggle, or correction from how long the key was held."""
    if state.take.recording is None:
        return
    now = asyncio.get_event_loop().time()
    hold_duration = now - state.take.key_down_time
    if state.take.mode == "toggle":
        return
    if _wants_correction(state, hold_duration, now):
        await _run_correction(state)
        return
    if hold_duration < HOLD_THRESHOLD:
        await _enter_toggle(state, hold_duration)
        return
    state.take.mode = "hold"
    log.info("Hold mode — transcribing (held %.1fs)", hold_duration)
    await finalize_recording(
        state,
        state.derived["backend_port"],
        state.settings["type_delay"],
    )


def _wants_correction(state, hold_duration, now):
    """True when this tap should re-transcribe the previous utterance."""
    return is_correction_tap(
        hold_duration,
        state.previous.wav_bytes,
        now,
        state.previous.transcription_end,
    )


async def _run_correction(state):
    """Drop the tap's audio and correct the previous transcription."""
    state.take.mode = "correction"
    log.info("Correction mode — re-transcribing with larger model")
    _stop_ignored_take(state)
    await correct_last_transcription(state)


async def handle_key_event(state, event):
    """Dispatch one evdev event for the hotkey or the recall key."""
    if event.type != ecodes.EV_KEY:
        return
    key_event = evdev.categorize(event)
    if await _on_recall(state, key_event):
        return
    if key_event.scancode != state.derived["target_key"]:
        return
    if key_event.keystate == key_event.key_down:
        await _on_key_down(state)
        return
    if key_event.keystate == key_event.key_up:
        await _on_key_up(state)


def _require_runtime(state, keyboards):
    """Return an error string when the client cannot start, else None."""
    if evdev is None or ecodes is None:
        return "evdev required: pip install evdev"
    if state.derived.get("target_key") is None:
        return f"Unknown key: {state.settings['key']}"
    if not keyboards:
        return "No keyboard found in /dev/input/ — add the user to the input group"
    return None


async def run_dictation(settings):
    """Listen for the hotkey until the process is stopped."""
    state = make_state(settings)
    refresh_derived(state)
    keyboards = find_keyboards(target_key_name=settings["key"])
    problem = _require_runtime(state, keyboards)
    if problem:
        log.error("%s", problem)
        raise SystemExit(1)
    _log_ready(state, keyboards)

    def _on_config_change(_updated):
        """Apply a reloaded settings file to the running session."""
        refresh_derived(state)

    asyncio.create_task(watch_config(settings, _on_config_change))
    event_queue = asyncio.Queue()
    asyncio.create_task(watch_keyboards(state, keyboards, event_queue))
    while True:
        event = await event_queue.get()
        await handle_key_event(state, event)


def _log_ready(state, keyboards):
    """Log the devices and hotkey the client is listening on."""
    names = ", ".join(device.name for device in keyboards)
    log.info(
        "Listening on %s (key: %s, backend: %s, port: %d)",
        names,
        state.settings["key"],
        state.settings["backend"],
        state.derived["backend_port"],
    )
    log.info(
        "Hold %s to record, release to transcribe and type",
        state.settings["key"],
    )
