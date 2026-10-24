"""Tests for the hold, toggle, and correction logic in dictation_session."""

import asyncio
import io
import json
import subprocess
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import dictation_session as ds

FAKE_ECODES = SimpleNamespace(
    EV_KEY=1,
    KEY_ESC=27,
    KEY_RIGHTCTRL=97,
    KEY_PAUSE=119,
)
TARGET = FAKE_ECODES.KEY_RIGHTCTRL
RECALL = FAKE_ECODES.KEY_PAUSE
SMALL = ds.MODEL_SIZE_ORDER[0]
LARGE = ds.MODEL_SIZE_ORDER[-1]


def _settings(**overrides):
    """Return a complete settings dict with optional overrides."""
    settings = {
        "key": "KEY_RIGHTCTRL",
        "recall_key": "KEY_PAUSE",
        "backend": "openvino",
        "port": 5000,
        "stream_interval": 0,
        "type_delay": 4,
        "vad_threshold": -40,
        "language": None,
        "formatting": False,
        "auto_punctuate": False,
        "translate_to": None,
        "sound": False,
        "notifications": False,
        "voice_commands": False,
        "mute_streams": False,
    }
    settings.update(overrides)
    return settings


def _state(**overrides):
    """Build a DictationState with a mock history and derived key codes."""
    state = ds.DictationState(settings=_settings(**overrides), history=MagicMock())
    state.derived = {"target_key": TARGET, "recall_key": RECALL, "backend_port": 5000}
    return state


def _run(coro):
    """Run *coro* on a fresh event loop."""
    return asyncio.run(coro)


def _wav(seconds):
    """Return WAV bytes holding *seconds* of silence."""
    buffer = io.BytesIO()
    with wave.Wave_write(buffer) as handle:
        handle.setnchannels(ds.CHANNELS)
        handle.setsampwidth(ds.SAMPLE_WIDTH)
        handle.setframerate(ds.SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * int(ds.SAMPLE_RATE * seconds))
    return buffer.getvalue()


def _finalize(state, wav):
    """Finish the current take with stop_recording returning *wav*."""
    state.take.recording = MagicMock()
    with patch.object(ds, "stop_recording", return_value=wav):
        _run(ds.finalize_recording(state, 5000, 4))


@pytest.fixture(name="codes", autouse=True)
def fixture_codes():
    """Expose fake evdev key codes to the session module."""
    with patch.object(ds, "ecodes", FAKE_ECODES):
        yield FAKE_ECODES


class TestSetup:
    """State creation and derived settings."""

    def test_make_state(self):
        with (
            patch.object(ds, "TranscriptionHistory") as history,
            patch.object(ds, "load_app_contexts", return_value={"a": {}}),
        ):
            state = ds.make_state({"k": 1})
        assert state.history is history.return_value
        assert state.app_contexts == {"a": {}}

    def test_refresh_derived_openvino(self):
        state = _state(port="5050", clipboard_mode=True)
        with patch.object(ds, "set_clipboard_mode") as clip:
            ds.refresh_derived(state)
        assert state.derived["target_key"] == TARGET
        assert state.derived["backend_port"] == 5050
        assert state.derived["recall_key"] == RECALL
        clip.assert_called_once_with(True)

    def test_refresh_derived_cpp_unknown_key(self):
        state = _state(backend="whisper-cpp", key="KEY_NOPE")
        state.derived = {}
        with patch.object(ds, "set_clipboard_mode"):
            ds.refresh_derived(state)
        assert "target_key" not in state.derived
        assert state.derived["backend_port"] == 5001


class TestBackspace:
    """Backspace helpers for Wayland and X11."""

    def test_zero_is_noop(self):
        with patch.object(ds.subprocess, "run") as run:
            ds.backspace_count(0)
        run.assert_not_called()

    def test_x11(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with patch.object(ds.subprocess, "run") as run:
            ds.backspace_count(3)
        assert run.call_args[0][0][-2:] == ["3", "BackSpace"]

    def test_wayland(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        with patch.object(ds.subprocess, "run") as run:
            ds.backspace_count(2)
        assert run.call_count == 2

    def test_wayland_missing_ydotool(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        with patch.object(ds.subprocess, "run", side_effect=FileNotFoundError) as run:
            ds.backspace_count(2)
        assert run.call_count == 1


class TestStreaming:
    """Live partial transcription while a toggle take is open."""

    def _loop(self, state, audio_chunks, texts):
        """Run streaming_loop over *audio_chunks* snapshots."""
        recording = MagicMock()
        recording.snapshot.side_effect = audio_chunks
        recording.proc.poll.side_effect = [None] * len(audio_chunks) + [0]
        state.take.recording = recording
        with (
            patch.object(ds, "trim_silence", side_effect=lambda audio, **_: audio),
            patch.object(ds, "transcribe_chunk", AsyncMock(side_effect=texts)),
            patch.object(ds, "type_text_async", AsyncMock()) as typer,
            patch.object(ds, "backspace_count") as back,
            patch.object(ds.asyncio, "sleep", AsyncMock()) as sleep,
        ):
            _run(ds.streaming_loop(state))
        return typer, back, sleep

    def test_types_only_the_changed_suffix(self):
        state = _state(backend="whisper-cpp")
        second = b"\x01\x00" * ds.SAMPLE_RATE
        texts = ["hello wx", "hello wx", "hello world"]
        typer, back, _sleep = self._loop(state, [second] * 3, texts)
        assert [c.args[0] for c in typer.await_args_list] == ["hello wx", "orld"]
        back.assert_called_once_with(1)
        assert state.take.streamed_text == "hello world"
        assert state.take.typed_char_count == len("hello world")

    def test_skips_short_audio(self):
        state = _state(backend="whisper-cpp")
        typer, _back, sleep = self._loop(state, [b""], [])
        typer.assert_not_awaited()
        sleep.assert_any_await(0.5)

    def test_skips_under_half_second(self):
        state = _state(backend="whisper-cpp")
        audio = b"\x01\x00" * ds.SAMPLE_RATE
        with patch.object(ds, "CHANNELS", 4):
            typer, _back, sleep = self._loop(state, [audio], [])
        typer.assert_not_awaited()
        sleep.assert_any_await(0.5)


class TestFinalize:
    """Finishing a take in hold and toggle modes."""

    def test_silence_restores_muted_streams(self):
        state = _state()
        state.take.mode = "hold"
        state.take.muted_streams = [1]
        with patch.object(ds, "unmute_streams") as unmute:
            _finalize(state, None)
        unmute.assert_called_once_with([1])
        assert not state.take.muted_streams
        assert state.take.mode is None

    def test_too_short(self):
        state = _state()
        state.take.mode = "hold"
        _finalize(state, _wav(0.1))
        assert state.take.mode is None

    def test_cancels_live_stream(self):
        state = _state()

        async def scenario():
            state.take.streaming_task = asyncio.create_task(asyncio.sleep(10))
            state.take.recording = MagicMock()
            with patch.object(ds, "stop_recording", return_value=None):
                await ds.finalize_recording(state, 5000, 4)

        _run(scenario())
        assert state.take.streaming_task is None

    def test_toggle_cpp(self):
        state = _state(backend="whisper-cpp")
        state.take.mode = "toggle"
        state.take.streamed_text = "hi"
        with (
            patch.object(ds, "transcribe_chunk", AsyncMock(return_value="hi there")),
            patch.object(ds, "type_text_async", AsyncMock()) as typer,
        ):
            _finalize(state, _wav(1.0))
        typer.assert_awaited_once_with(" there", delay_ms=4)
        assert state.take.streamed_text == ""

    def test_toggle_cpp_empty(self):
        state = _state(backend="whisper-cpp")
        state.take.mode = "toggle"
        with (
            patch.object(ds, "transcribe_chunk", AsyncMock(return_value="")),
            patch.object(ds, "type_text_async", AsyncMock()) as typer,
        ):
            _finalize(state, _wav(1.0))
        typer.assert_not_awaited()
        assert state.take.mode is None

    def test_hold_no_text(self):
        state = _state()
        state.take.mode = "hold"
        with patch.object(ds, "transcribe_batch", AsyncMock(return_value="")):
            _finalize(state, _wav(1.0))
        assert state.take.mode is None

    def test_hold_postprocesses_and_remembers(self):
        state = _state(
            backend="whisper-cpp",
            sound=True,
            notifications=True,
            formatting=True,
            auto_punctuate=True,
            translate_to="French",
            language="en",
        )
        state.take.mode = "hold"
        with (
            patch.object(ds, "transcribe_chunk", AsyncMock(return_value=" hi ")),
            patch.object(ds, "format_dictation", side_effect=str.upper),
            patch.object(ds, "auto_punctuate", AsyncMock(return_value="HI.")),
            patch.object(ds, "translate_text", AsyncMock(return_value="SALUT.")),
            patch.object(ds, "get_focused_app", return_value=("x", "t")),
            patch.object(ds, "try_dbus_handoff", return_value=True),
            patch.object(ds, "play_sound") as sound,
            patch.object(ds, "notify") as notify,
        ):
            _finalize(state, _wav(1.0))
        state.history.add.assert_called_once_with("SALUT.")
        assert state.previous.typed_text == "SALUT."
        assert state.previous.used_dbus is True
        assert sound.call_count == 2
        notify.assert_called_once_with("Whisper", "SALUT.")

    def test_hold_voice_command(self):
        state = _state(voice_commands=True, notifications=True)
        state.take.mode = "hold"
        with (
            patch.object(ds, "transcribe_batch", AsyncMock(return_value="undo")),
            patch.object(ds, "execute_voice_command", return_value=True),
            patch.object(ds, "notify") as notify,
            patch.object(ds, "type_text_async", AsyncMock()) as typer,
        ):
            _finalize(state, _wav(1.0))
        typer.assert_not_awaited()
        notify.assert_called_once_with("Voice Command", "undo")

    def _deliver(self, contexts, context_ok, plain_ok, **settings):
        """Run a hold take through delivery and return the typing mock."""
        state = _state(**settings)
        state.app_contexts = contexts
        with (
            patch.object(ds, "transcribe_batch", AsyncMock(return_value="hi")),
            patch.object(ds, "execute_voice_command", return_value=False),
            patch.object(ds, "get_focused_app", return_value=("code", "t")),
            patch.object(
                ds,
                "try_dbus_handoff_with_context",
                return_value=context_ok,
            ) as ctx,
            patch.object(ds, "try_dbus_handoff", return_value=plain_ok),
            patch.object(ds, "type_text_async", AsyncMock()) as typer,
        ):
            _finalize(state, _wav(1.0))
        return state, ctx, typer

    def test_context_handoff(self):
        contexts = {"code": {"tones": ["formal"]}}
        state, ctx, typer = self._deliver(contexts, True, False, voice_commands=True)
        assert json.loads(ctx.call_args[0][1]) == {"tones": ["formal"], "app": "code"}
        typer.assert_not_awaited()
        assert state.previous.used_dbus is True

    def test_plain_handoff_after_context_refused(self):
        contexts = {"code": {"tones": ["x"]}}
        state, ctx, typer = self._deliver(contexts, False, True)
        ctx.assert_called_once()
        typer.assert_not_awaited()
        assert state.previous.used_dbus is True

    def test_types_when_no_shell(self):
        state, _ctx, typer = self._deliver({}, False, False)
        typer.assert_awaited_once_with("hi", delay_ms=4)
        assert state.previous.used_dbus is False


class FakeResponse:
    """Minimal aiohttp response context manager."""

    def __init__(self, status, payload):
        self.status = status
        self.payload = payload

    async def json(self):
        """Return the canned payload."""
        return self.payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class FakeSession:
    """aiohttp ClientSession stand-in that answers by URL path."""

    def __init__(self, routes):
        self.routes = routes

    def get(self, url, **_kwargs):
        """Return the canned response for *url*, or raise its error."""
        for suffix, reply in self.routes.items():
            if suffix in url:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise ConnectionError(url)

    post = get

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _aiohttp(routes):
    """Return an aiohttp stand-in whose sessions use *routes*."""
    return SimpleNamespace(
        ClientSession=lambda: FakeSession(routes),
        ClientError=ConnectionError,
    )


class TestCorrection:
    """Re-transcribing the previous take with a larger model."""

    def _prev_state(self, **overrides):
        state = _state(**overrides)
        state.previous.wav_bytes = b"wav"
        state.previous.typed_text = "old"
        state.take.mode = "correction"
        return state

    def _correct(self, state, routes):
        with (
            patch.object(ds, "aiohttp", _aiohttp(routes)),
            patch.object(ds, "backspace_count") as back,
            patch.object(ds, "type_text_async", AsyncMock()) as typer,
            patch.object(ds, "notify") as notify,
        ):
            _run(ds.correct_last_transcription(state))
        assert state.take.mode is None
        return back, typer, notify

    def test_nothing_to_correct(self):
        state = _state()
        state.take.mode = "correction"
        _run(ds.correct_last_transcription(state))
        assert state.take.mode is None

    def test_after_dbus(self):
        state = self._prev_state()
        state.previous.used_dbus = True
        _back, _typer, notify = self._correct(state, {})
        notify.assert_called_once()

    def test_without_aiohttp(self):
        state = self._prev_state()
        with patch.object(ds, "aiohttp", None):
            _run(ds.correct_last_transcription(state))
        assert state.take.mode is None

    def test_models_unavailable(self):
        state = self._prev_state()
        _back, typer, _notify = self._correct(state, {"/models": FakeResponse(500, {})})
        typer.assert_not_awaited()

    def test_no_larger_model(self):
        state = self._prev_state()
        routes = {
            "/models": FakeResponse(200, {"models": [SMALL, "other"]}),
            "/model/default": FakeResponse(200, {"model": LARGE}),
        }
        _back, _typer, notify = self._correct(state, routes)
        notify.assert_called_once_with("Whisper", "No larger model available")

    def test_corrects_with_larger_model(self):
        state = self._prev_state(notifications=True, language="de")
        routes = {
            "/models": FakeResponse(200, {"models": [SMALL, LARGE]}),
            "/model/default": OSError("down"),
            f"/transcribe/{LARGE}?language=de": FakeResponse(200, {"text": " new "}),
        }
        back, typer, notify = self._correct(state, routes)
        back.assert_called_once_with(3)
        typer.assert_awaited_once_with("new", delay_ms=4)
        notify.assert_called_once_with("Correction", "new")
        assert state.previous.typed_text == "new"

    def test_same_result(self):
        state = self._prev_state(notifications=True)
        routes = {
            "/models": FakeResponse(200, {"models": [LARGE]}),
            "/model/default": FakeResponse(200, {"model": SMALL}),
            "/transcribe/": FakeResponse(200, {"text": "old"}),
        }
        _back, typer, notify = self._correct(state, routes)
        typer.assert_not_awaited()
        notify.assert_called_once_with("Whisper", "Same result with larger model")

    def test_retranscribe_failed(self):
        state = self._prev_state()
        routes = {
            "/models": FakeResponse(200, {"models": [LARGE]}),
            "/model/default": FakeResponse(200, {"model": SMALL}),
            "/transcribe/": FakeResponse(503, {}),
        }
        _back, typer, _notify = self._correct(state, routes)
        typer.assert_not_awaited()

    def test_retranscribe_error(self):
        state = self._prev_state()
        routes = {
            "/models": FakeResponse(200, {"models": [LARGE]}),
            "/model/default": FakeResponse(200, {"model": SMALL}),
            "/transcribe/": TypeError("bad"),
        }
        _back, typer, _notify = self._correct(state, routes)
        typer.assert_not_awaited()


class FakeDevice:
    """Keyboard device that yields canned events then disconnects."""

    def __init__(self, events, path="/dev/input/event0", close_error=False):
        self.events = events
        self.path = path
        self.name = "kbd"
        self.close_error = close_error
        self.closed = False

    async def async_read_loop(self):
        """Yield the canned events, then simulate a disconnect."""
        for event in self.events:
            yield event
        raise OSError("gone")

    def close(self):
        """Record the close, optionally failing like a vanished device."""
        self.closed = True
        if self.close_error:
            raise OSError("closed")


class TestKeyboards:
    """Keyboard event pumps and hot-plug scanning."""

    def test_pump_events_aborts_on_escape(self):
        escape = SimpleNamespace(type=FAKE_ECODES.EV_KEY, code=27, value=1)
        other = SimpleNamespace(type=0, code=0, value=0)
        device = FakeDevice([escape, other], close_error=True)

        async def scenario():
            queue = asyncio.Queue()
            await ds.pump_events(device, queue)
            return queue.qsize()

        with patch.object(ds, "abort_typing") as abort:
            assert _run(scenario()) == 2
        abort.assert_called_once()
        assert device.closed

    def test_watch_keyboards_rescans(self):
        state = _state()
        first = FakeDevice([], "/a")
        second = FakeDevice([], "/b")
        real_sleep = asyncio.sleep
        calls = []

        async def fake_sleep(_seconds):
            calls.append(_seconds)
            await real_sleep(0)
            if len(calls) == 2:
                raise asyncio.CancelledError

        async def scenario():
            with (
                patch.object(ds, "find_keyboards", return_value=[second]) as find,
                patch.object(ds.asyncio, "sleep", fake_sleep),
                pytest.raises(asyncio.CancelledError),
            ):
                await ds.watch_keyboards(state, [first], asyncio.Queue())
            return find

        find = _run(scenario())
        assert find.call_count == 1
        assert first.closed
        assert second.closed


def _key(scancode, keystate):
    """Return a categorized key event stand-in."""
    return SimpleNamespace(scancode=scancode, keystate=keystate, key_down=1, key_up=0)


def _press(state, *keys):
    """Feed categorized *keys* through handle_key_event."""
    event = SimpleNamespace(type=FAKE_ECODES.EV_KEY)
    fake_evdev = SimpleNamespace(categorize=MagicMock(side_effect=list(keys)))

    async def scenario():
        with patch.object(ds, "evdev", fake_evdev):
            for _ in keys:
                await ds.handle_key_event(state, event)

    _run(scenario())


class TestRecall:
    """History picker and clipboard fallback."""

    def _recall(self, history, run_error=None):
        state = _state()
        state.history.recent.return_value = history
        with (
            patch.object(ds, "try_dbus_history_picker", return_value=False),
            patch.object(ds.subprocess, "run", side_effect=run_error) as run,
            patch.object(ds, "notify") as notify,
        ):
            _press(state, _key(RECALL, 1))
        return run, notify

    def test_copies_last_transcript(self):
        run, notify = self._recall([("text",)])
        assert run.call_args[0][0] == ["wl-copy", "text"]
        notify.assert_called_once()

    def test_empty_history(self):
        run, _notify = self._recall([])
        run.assert_not_called()

    def test_wl_copy_failure(self):
        error = subprocess.CalledProcessError(1, "wl-copy")
        _run_mock, notify = self._recall([("text",)], run_error=error)
        notify.assert_not_called()

    def test_picker_shown(self):
        state = _state()
        with patch.object(ds, "try_dbus_history_picker", return_value=True):
            _press(state, _key(RECALL, 1), _key(RECALL, 0))
        state.history.recent.assert_not_called()


class TestKeyHandling:
    """Hotkey down/up dispatch into hold, toggle, and correction."""

    def test_ignores_other_events(self):
        state = _state()
        _run(ds.handle_key_event(state, SimpleNamespace(type=0)))
        _press(state, _key(5, 1), _key(TARGET, 2))
        assert state.take.recording is None

    def test_key_down_starts_recording(self):
        state = _state(sound=True, mute_streams=True)
        recording = MagicMock()
        recording.proc.pid = 42
        with (
            patch.object(ds, "record_audio", return_value=recording),
            patch.object(ds, "mute_other_streams", return_value=[7]) as mute,
            patch.object(ds, "play_sound"),
        ):
            _press(state, _key(TARGET, 1))
        mute.assert_called_once_with(42)
        assert state.take.muted_streams == [7]
        assert state.take.recording is recording

    def test_second_tap_stops_toggle(self):
        state = _state()
        state.take.recording = MagicMock()
        state.take.mode = "toggle"
        with patch.object(ds, "finalize_recording", AsyncMock()) as final:
            _press(state, _key(TARGET, 1), _key(TARGET, 0))
        final.assert_awaited_once_with(state, 5000, 4)
        state.take.mode = "hold"
        with patch.object(ds, "finalize_recording", AsyncMock()) as final:
            _press(state, _key(TARGET, 1))
        final.assert_not_awaited()

    def test_key_up_without_recording(self):
        state = _state()
        _press(state, _key(TARGET, 0))
        assert state.take.mode is None

    def _release(self, state, held, correction=False):
        """Release the hotkey after holding it for *held* seconds."""
        state.take.recording = MagicMock()
        event = SimpleNamespace(type=FAKE_ECODES.EV_KEY)
        fake_evdev = SimpleNamespace(categorize=MagicMock(return_value=_key(TARGET, 0)))

        async def scenario():
            state.take.key_down_time = asyncio.get_running_loop().time() - held
            with (
                patch.object(ds, "evdev", fake_evdev),
                patch.object(ds, "is_correction_tap", return_value=correction),
                patch.object(ds, "finalize_recording", AsyncMock()) as final,
                patch.object(ds, "correct_last_transcription", AsyncMock()) as fix,
                patch.object(ds, "stop_recording"),
                patch.object(ds, "unmute_streams"),
                patch.object(ds, "streaming_loop", AsyncMock()),
            ):
                await ds.handle_key_event(state, event)
                if state.take.streaming_task:
                    await state.take.streaming_task
            return final, fix

        return _run(scenario())

    def test_long_hold_transcribes(self):
        state = _state()
        final, _fix = self._release(state, ds.HOLD_THRESHOLD + 1)
        final.assert_awaited_once()
        assert state.take.mode == "hold"

    def test_short_tap_enters_toggle(self):
        state = _state()
        self._release(state, 0)
        assert state.take.mode == "toggle"
        assert state.take.streaming_task is None
        state = _state(backend="whisper-cpp")
        self._release(state, 0)
        assert state.take.streaming_task is not None

    def test_quick_tap_corrects(self):
        state = _state()
        state.take.muted_streams = [3]
        _final, fix = self._release(state, 0, correction=True)
        fix.assert_awaited_once_with(state)
        assert state.take.recording is None
        assert state.take.mode == "correction"


class TestRunDictation:
    """Startup checks and the main event loop."""

    def _start(self, keyboards, evdev_module, target=TARGET, captured=None):
        """Run run_dictation until the first event, recording into *captured*."""
        captured = {} if captured is None else captured

        def fake_make_state(_settings):
            state = _state()
            captured["state"] = state
            return state

        def fake_refresh(state):
            state.derived["target_key"] = target
            captured["refreshes"] = captured.get("refreshes", 0) + 1

        async def fake_watch_config(_settings, callback):
            captured["callback"] = callback

        async def fake_watch_keyboards(_state, _devices, queue):
            await queue.put("event")

        async def stop_after_reload(*_args):
            captured["callback"]({})
            raise asyncio.CancelledError

        with (
            patch.object(ds, "evdev", evdev_module),
            patch.object(ds, "make_state", fake_make_state),
            patch.object(ds, "refresh_derived", fake_refresh),
            patch.object(ds, "find_keyboards", return_value=keyboards),
            patch.object(ds, "watch_config", fake_watch_config),
            patch.object(ds, "watch_keyboards", fake_watch_keyboards),
            patch.object(
                ds, "handle_key_event", AsyncMock(side_effect=stop_after_reload)
            ),
        ):
            _run(ds.run_dictation(_settings()))
        return captured

    def test_requires_evdev(self):
        with pytest.raises(SystemExit):
            self._start([FakeDevice([])], None)

    def test_requires_known_key(self):
        with pytest.raises(SystemExit):
            self._start([FakeDevice([])], MagicMock(), target=None)

    def test_requires_keyboard(self):
        with pytest.raises(SystemExit):
            self._start([], MagicMock())

    def test_handles_events_and_reloads(self):
        captured = {}
        with pytest.raises(asyncio.CancelledError):
            self._start([FakeDevice([])], MagicMock(), captured=captured)
        assert captured["refreshes"] == 2
