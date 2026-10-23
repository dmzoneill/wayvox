"""Audio capture, keyboard, and transcription tests."""

import asyncio
import io
import signal
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ptt_support import (
    _AsyncCtx,
    _AsyncIterLines,
    _loud_raw,
    _make_aiohttp_mocks,
    _quiet_raw,
    _sine_samples,
    make_silent_raw,
)

import dictation_audio
import dictation_output
import push_to_talk as ptt


class TestConstants:
    """Verify module-level constants."""

    def test_sample_rate(self):
        """Sample rate."""
        assert ptt.SAMPLE_RATE == 16000

    def test_channels(self):
        """Channels."""
        assert ptt.CHANNELS == 1

    def test_sample_width(self):
        """Sample width."""
        assert ptt.SAMPLE_WIDTH == 2

    def test_hold_threshold(self):
        """Hold threshold."""
        assert ptt.HOLD_THRESHOLD == 1.5

    def test_correction_window(self):
        """Correction window."""
        assert ptt.CORRECTION_WINDOW == 3.0

    def test_model_size_order(self):
        """Model size order."""
        assert ptt.MODEL_SIZE_ORDER == ["tiny", "base", "small", "medium", "large"]

    def test_voice_commands_has_expected_keys(self):
        """Voice commands has expected keys."""
        expected = {
            "select all",
            "undo that",
            "redo that",
            "copy that",
            "paste that",
            "cut that",
            "new line",
            "new paragraph",
            "tab key",
            "delete that",
            "delete word",
            "period",
            "comma",
            "question mark",
            "exclamation mark",
            "exclamation point",
            "colon",
            "semicolon",
            "open quote",
            "close quote",
        }
        assert expected == set(ptt.VOICE_COMMANDS.keys())

    def test_voice_command_select_all(self):
        """Voice command select all."""
        assert ptt.VOICE_COMMANDS["select all"] == [("key", "29:1 30:1 30:0 29:0")]

    def test_voice_command_new_paragraph(self):
        """Voice command new paragraph."""
        assert ptt.VOICE_COMMANDS["new paragraph"] == [
            ("key", "28:1 28:0"),
            ("key", "28:1 28:0"),
        ]

    def test_voice_command_period(self):
        """Voice command period."""
        assert ptt.VOICE_COMMANDS["period"] == [("type", ".")]

    def test_default_app_contexts(self):
        """Default app contexts."""
        assert "org.mozilla.Thunderbird.desktop" in ptt.DEFAULT_APP_CONTEXTS
        thunderbird = ptt.DEFAULT_APP_CONTEXTS["org.mozilla.Thunderbird.desktop"]
        assert thunderbird == {"tones": ["professional"]}
        assert ptt.DEFAULT_APP_CONTEXTS["org.gnome.Terminal.desktop"] == {"tones": []}

    def test_dbus_bus_names(self):
        """Dbus bus names."""
        assert ptt.DBUS_BUS_NAMES == ["org.gnome.Shell", "com.whisper.LanguageBuddy"]

    def test_dbus_object_path(self):
        """Dbus object path."""
        assert ptt.DBUS_OBJECT_PATH == "/com/whisper/LanguageBuddy"

    def test_dbus_interface(self):
        """Dbus interface."""
        assert ptt.DBUS_INTERFACE == "com.whisper.LanguageBuddy"


class TestFindKeyboards:
    """Tests covering findkeyboards."""

    def test_finds_keyboard_with_key_a_and_enter(self):
        """Finds keyboard with key a and enter."""
        dev = MagicMock()
        dev.name = "AT Translated Set 2 keyboard"
        dev.path = "/dev/input/event0"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("KEY_ENTER",), ("KEY_B",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert dev in result

    def test_skips_virtual_device(self):
        """Skips virtual device."""
        virtual_dev = MagicMock()
        virtual_dev.name = "Virtual Keyboard"
        virtual_dev.path = "/dev/input/event0"
        real_dev = MagicMock()
        real_dev.name = "Real keyboard"
        real_dev.path = "/dev/input/event1"
        real_dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("KEY_ENTER",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = [
            "/dev/input/event0",
            "/dev/input/event1",
        ]
        mock_evdev.InputDevice.side_effect = [virtual_dev, real_dev]
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert real_dev in result
        assert virtual_dev not in result

    def test_returns_empty_when_no_keyboard(self):
        """Returns empty when no keyboard."""
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = []
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert not result

    def test_skips_device_missing_key_enter(self):
        """Skips device missing key enter."""
        dev = MagicMock()
        dev.name = "Mouse"
        dev.path = "/dev/input/event0"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("BTN_LEFT",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert not result

    def test_handles_tuple_key_names(self):
        """evdev sometimes returns key names as tuples of aliases."""
        dev = MagicMock()
        dev.name = "Keyboard"
        dev.path = "/dev/input/event0"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [
                (("KEY_A", "KEY_A_ALIAS"),),
                (("KEY_ENTER", "KEY_RETURN"),),
            ],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert dev in result

    def test_skips_non_ev_key_types(self):
        """Skips non ev key types."""
        dev = MagicMock()
        dev.name = "Keyboard"
        dev.path = "/dev/input/event0"
        dev.capabilities.return_value = {
            ("EV_REL", 2): [("REL_X",), ("REL_Y",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        mock_evdev.InputDevice.return_value = dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert not result

    def test_exclude_skips_known_paths(self):
        """Exclude skips known paths."""
        dev = MagicMock()
        dev.name = "Keyboard"
        dev.path = "/dev/input/event0"
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards(exclude=["/dev/input/event0"])
        mock_evdev.InputDevice.assert_not_called()
        assert not result

    def test_returns_multiple_keyboards(self):
        """Returns multiple keyboards."""
        dev0 = MagicMock()
        dev0.name = "USB Keyboard"
        dev0.path = "/dev/input/event0"
        dev0.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("KEY_ENTER",)],
        }
        dev1 = MagicMock()
        dev1.name = "PS/2 Keyboard"
        dev1.path = "/dev/input/event1"
        dev1.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_A",), ("KEY_ENTER",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = [
            "/dev/input/event0",
            "/dev/input/event1",
        ]
        mock_evdev.InputDevice.side_effect = [dev0, dev1]
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards()
        assert dev0 in result
        assert dev1 in result

    def test_target_key_includes_non_keyboard_device(self):
        """Include a non-keyboard device that exposes the target key."""
        extra_dev = MagicMock()
        extra_dev.name = "ThinkPad Extra Buttons"
        extra_dev.path = "/dev/input/event12"
        extra_dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_VOICECOMMAND",), ("KEY_MICMUTE",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event12"]
        mock_evdev.InputDevice.return_value = extra_dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards(target_key_name="KEY_VOICECOMMAND")
        assert extra_dev in result

    def test_target_key_not_on_device_still_excluded(self):
        """Non-keyboard device without the target key is still excluded."""
        dev = MagicMock()
        dev.name = "ThinkPad Extra Buttons"
        dev.path = "/dev/input/event12"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_MICMUTE",), ("KEY_VOLUMEUP",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event12"]
        mock_evdev.InputDevice.return_value = dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards(target_key_name="KEY_VOICECOMMAND")
        assert not result

    def test_target_key_none_does_not_change_existing_behavior(self):
        """Passing target_key_name=None still requires full keyboard (KEY_A + KEY_ENTER)."""
        dev = MagicMock()
        dev.name = "ThinkPad Extra Buttons"
        dev.path = "/dev/input/event12"
        dev.capabilities.return_value = {
            ("EV_KEY", 1): [("KEY_VOICECOMMAND",), ("KEY_MICMUTE",)],
        }
        mock_evdev = MagicMock()
        mock_evdev.list_devices.return_value = ["/dev/input/event12"]
        mock_evdev.InputDevice.return_value = dev
        with patch.object(dictation_audio, "evdev", mock_evdev):
            result = ptt.find_keyboards(target_key_name=None)
        assert not result


class TestAudioBuffer:
    """Tests covering audiobuffer."""

    def test_init_starts_reader_thread(self):
        """Init starts reader thread."""
        proc = MagicMock()
        proc.stdout.read.return_value = b""
        buf = ptt.AudioBuffer(proc)
        buf.thread.join(timeout=1)
        assert not buf.thread.is_alive()

    def test_reader_accumulates_data(self):
        """Reader accumulates data."""
        chunks = [b"AAAA", b"BBBB", b""]
        proc = MagicMock()
        proc.stdout.read.side_effect = chunks
        buf = ptt.AudioBuffer(proc)
        buf.thread.join(timeout=2)
        assert buf.snapshot() == b"AAAABBBB"

    def test_snapshot_returns_copy(self):
        """Snapshot returns copy."""
        proc = MagicMock()
        proc.stdout.read.return_value = b""
        buf = ptt.AudioBuffer(proc)
        buf.thread.join(timeout=1)
        with buf.lock:
            buf.samples.extend(b"test")
        snap = buf.snapshot()
        assert snap == b"test"
        # Modifying snap should not affect internal buf
        assert isinstance(snap, bytes)

    def test_stop_sends_sigterm_and_returns_data(self):
        """Stop sends sigterm and returns data."""
        chunks = [b"data", b""]
        proc = MagicMock()
        proc.stdout.read.side_effect = chunks
        proc.wait.return_value = 0
        buf = ptt.AudioBuffer(proc)
        buf.thread.join(timeout=2)
        result = buf.stop()
        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.wait.assert_called_once()
        assert result == b"data"


class TestRecordAudio:
    """Tests covering recordaudio."""

    def test_record_audio_returns_audio_buffer(self, mock_popen):
        """Record audio returns audio buffer."""
        popen_cls, _proc = mock_popen
        with patch.object(dictation_audio, "AudioBuffer") as mock_buf_cls:
            mock_buf_cls.return_value = MagicMock()
            ptt.record_audio()
        popen_cls.assert_called_once()
        args = popen_cls.call_args
        assert "parec" in args[0][0]
        assert "--format=s16le" in args[0][0]
        assert "--rate=16000" in args[0][0]

    def test_record_audio_passes_the_process_to_the_buffer(self, mock_popen):
        """AudioBuffer is constructed with the parec process."""
        _popen_cls, proc = mock_popen
        with patch.object(dictation_audio, "AudioBuffer") as mock_buf_cls:
            ptt.record_audio()
        mock_buf_cls.assert_called_once_with(proc)


class TestStopRecording:
    """Tests covering stoprecording."""

    def test_returns_none_for_all_silence(self):
        """Returns none for all silence."""
        audio_buf = MagicMock()
        audio_buf.stop.return_value = make_silent_raw(duration_s=1.0)
        result = ptt.stop_recording(audio_buf, vad_threshold=-40)
        assert result is None

    def test_returns_wav_for_speech(self):
        """Returns wav for speech."""
        audio_buf = MagicMock()
        audio_buf.stop.return_value = _loud_raw(duration_s=0.5)
        result = ptt.stop_recording(audio_buf, vad_threshold=-40)
        assert result is not None
        # Verify it's a valid WAV file
        wav_io = io.BytesIO(result)
        with wave.open(wav_io, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_returns_none_for_empty_audio(self):
        """Returns none for empty audio."""
        audio_buf = MagicMock()
        audio_buf.stop.return_value = b""
        result = ptt.stop_recording(audio_buf, vad_threshold=-40)
        assert result is None


class TestTrimSilence:
    """Pure logic function — test every branch and edge case."""

    def test_empty_input(self):
        """Empty input."""
        assert ptt.trim_silence(b"") == b""

    def test_none_like_falsy(self):
        """None like falsy."""
        assert ptt.trim_silence(b"") == b""

    def test_short_data_below_frame_size(self):
        # Less than one frame — returned as-is
        """Short data below frame size."""
        short = b"\x00\x01"
        assert ptt.trim_silence(short) == short

    def test_all_silence_returns_empty(self):
        """All silence returns empty."""
        raw = make_silent_raw(duration_s=1.0)
        result = ptt.trim_silence(raw, threshold_db=-40)
        assert result == b""

    def test_loud_audio_preserved(self):
        """Loud audio preserved."""
        raw = _loud_raw(duration_s=0.5)
        result = ptt.trim_silence(raw, threshold_db=-40)
        assert len(result) > 0

    def test_leading_silence_trimmed(self):
        """Leading silence trimmed."""
        silence = make_silent_raw(duration_s=0.5)
        loud = _loud_raw(duration_s=0.5)
        raw = silence + loud
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        # Result should be shorter than original (silence removed)
        assert len(result) < len(raw)

    def test_trailing_silence_trimmed(self):
        """Trailing silence trimmed."""
        loud = _loud_raw(duration_s=0.5)
        silence = make_silent_raw(duration_s=0.5)
        raw = loud + silence
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        assert len(result) < len(raw)

    def test_both_sides_trimmed(self):
        """Both sides trimmed."""
        silence = make_silent_raw(duration_s=0.5)
        loud = _loud_raw(duration_s=0.5)
        raw = silence + loud + silence
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        assert len(result) < len(raw)

    def test_pad_frames_preserves_context(self):
        """Pad frames preserves context."""
        silence = make_silent_raw(duration_s=0.5)
        loud = _loud_raw(duration_s=0.5)
        raw = silence + loud + silence
        result_no_pad = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        result_with_pad = ptt.trim_silence(raw, threshold_db=-40, pad_frames=3)
        # With padding, we should keep more data
        assert len(result_with_pad) >= len(result_no_pad)

    def test_pad_frames_clamped_to_bounds(self):
        """Padding shouldn't go before index 0 or past last frame."""
        loud = _loud_raw(duration_s=0.1)
        result = ptt.trim_silence(loud, threshold_db=-40, pad_frames=100)
        assert len(result) > 0

    def test_custom_threshold(self):
        # Very quiet audio that would pass at -80 dB but fail at -20 dB
        """Custom threshold."""
        quiet = _quiet_raw(duration_s=0.5, amplitude=5)
        result_loose = ptt.trim_silence(quiet, threshold_db=-100)
        result_strict = ptt.trim_silence(quiet, threshold_db=-10)
        assert len(result_loose) > 0
        assert result_strict == b""

    def test_custom_frame_ms(self):
        """Custom frame ms."""
        loud = _loud_raw(duration_s=0.5)
        result_20 = ptt.trim_silence(loud, threshold_db=-40, frame_ms=20)
        result_10 = ptt.trim_silence(loud, threshold_db=-40, frame_ms=10)
        # Both should return non-empty
        assert len(result_20) > 0
        assert len(result_10) > 0

    def test_single_frame_of_audio(self):
        """Exactly one frame of loud audio."""
        frame_samples = int(16000 * 20 / 1000)  # 320 samples for 20ms
        samples = _sine_samples(frame_samples, 16000, 16000)
        raw = struct.pack(f"<{len(samples)}h", *samples)
        result = ptt.trim_silence(raw, threshold_db=-40, pad_frames=0)
        assert len(result) > 0

    def test_rms_zero_treated_as_silence(self):
        """Frame of all zeros should be treated as -100 dB (silence)."""
        frame_samples = int(16000 * 20 / 1000)
        raw = b"\x00\x00" * frame_samples
        result = ptt.trim_silence(raw, threshold_db=-40)
        assert result == b""


class TestTranscribeStream:
    """Tests covering transcribe stream."""

    @pytest.mark.asyncio
    async def test_streams_text_and_types(self):
        """Streams text and types."""
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines(
            [
                b'data: {"text": "hello "}\n',
                b'data: {"text": "world"}\n',
                b'data: {"done": true}\n',
            ],
        )
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with (
            patch.object(dictation_audio, "aiohttp", mock_aiohttp),
            patch.object(
                dictation_output,
                "type_text",
            ) as mock_type,
            patch.object(dictation_audio, "transcribe_batch", new_callable=AsyncMock),
        ):
            result = await ptt.transcribe_stream(b"wavdata", 5000, 2)
        assert result is True
        assert mock_type.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_to_batch_on_non_200(self):
        """Fallback to batch on non 200."""
        resp = MagicMock()
        resp.status = 404
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with (
            patch.object(dictation_audio, "aiohttp", mock_aiohttp),
            patch.object(
                dictation_audio,
                "transcribe_batch",
                new_callable=AsyncMock,
                return_value="fallback",
            ) as mock_batch,
        ):
            result = await ptt.transcribe_stream(b"wavdata", 5000, 2)
        mock_batch.assert_awaited_once_with(b"wavdata", 5000, language=None)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_skips_non_sse_lines(self):
        """Skips non sse lines."""
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines(
            [
                b": comment\n",
                b"\n",
                b"event: ping\n",
                b'data: {"text": "only"}\n',
                b'data: {"done": true}\n',
            ],
        )
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with (
            patch.object(dictation_audio, "aiohttp", mock_aiohttp),
            patch.object(
                dictation_output,
                "type_text",
            ) as mock_type,
        ):
            await ptt.transcribe_stream(b"wavdata", 5000, 2)
        mock_type.assert_called_once_with("only", delay_ms=2)

    @pytest.mark.asyncio
    async def test_skips_invalid_json(self):
        """Skips invalid json."""
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines(
            [
                b"data: {invalid json}\n",
                b'data: {"text": "valid"}\n',
                b'data: {"done": true}\n',
            ],
        )
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with (
            patch.object(dictation_audio, "aiohttp", mock_aiohttp),
            patch.object(
                dictation_output,
                "type_text",
            ) as mock_type,
        ):
            await ptt.transcribe_stream(b"wavdata", 5000, 2)
        mock_type.assert_called_once_with("valid", delay_ms=2)

    @pytest.mark.asyncio
    async def test_language_appended_to_url(self):
        """Language appended to url."""
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([b'data: {"done": true}\n'])
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            await ptt.transcribe_stream(b"wav", 5000, 2, language="de")
        url_arg = mock_session.post.call_args[0][0]
        assert "?language=de" in url_arg

    @pytest.mark.asyncio
    async def test_no_text_returns_false(self):
        """No text returns false."""
        resp = MagicMock()
        resp.status = 200
        resp.content = _AsyncIterLines([b'data: {"done": true}\n'])
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.transcribe_stream(b"wav", 5000, 2)
        assert result is False


class TestTranscribeBatch:
    """Tests covering transcribe batch."""

    @pytest.mark.asyncio
    async def test_success_returns_text(self):
        """Success returns text."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "  hello world  "})
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.transcribe_batch(b"wav", 5000)
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        """Non 200 returns empty."""
        resp = AsyncMock()
        resp.status = 500
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.transcribe_batch(b"wav", 5000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_language_param(self):
        """Language param."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "hallo"})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            await ptt.transcribe_batch(b"wav", 5000, language="de")
        url_arg = mock_session.post.call_args[0][0]
        assert "?language=de" in url_arg


class TestTranscribeChunk:
    """Tests covering transcribe chunk."""

    @pytest.mark.asyncio
    async def test_success_returns_text(self):
        """Success returns text."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": " chunk text "})
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.transcribe_chunk(b"wav", 5000)
        assert result == "chunk text"

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        """Non 200 returns empty."""
        resp = AsyncMock()
        resp.status = 500
        mock_aiohttp, _mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.transcribe_chunk(b"wav", 5000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        """When ClientSession raises TimeoutError, return empty string."""
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientTimeout = MagicMock()
        mock_aiohttp.ClientError = Exception

        class _RaisingCtx:
            async def __aenter__(self):
                raise asyncio.TimeoutError

            async def __aexit__(self, *a):
                return False

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_RaisingCtx())
        mock_aiohttp.ClientSession = MagicMock(return_value=_AsyncCtx(mock_session))

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            result = await ptt.transcribe_chunk(b"wav", 5000)
        assert result == ""

    @pytest.mark.asyncio
    async def test_language_param(self):
        """Language param."""
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value={"text": "hallo"})
        mock_aiohttp, mock_session = _make_aiohttp_mocks(resp)

        with patch.object(dictation_audio, "aiohttp", mock_aiohttp):
            await ptt.transcribe_chunk(b"wav", 5000, language="fr")
        url_arg = mock_session.post.call_args[0][0]
        assert "?language=fr" in url_arg
