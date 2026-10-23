"""Keyboard discovery, capture, and HTTP transcription for dictation."""

import asyncio
import io
import json
import logging
import math
import signal
import struct
import subprocess
import threading
import wave

from dictation_config import (
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
)
from dictation_deps import aiohttp, evdev
from dictation_output import type_text_async

log = logging.getLogger("ptt")


def _client_errors():
    """HTTP client errors, ignoring stand-ins that are not exception types."""
    found = [asyncio.TimeoutError, OSError]
    client_error = getattr(aiohttp, "ClientError", None) if aiohttp else None
    if isinstance(client_error, type) and issubclass(client_error, BaseException):
        found.insert(0, client_error)
    return tuple(found)


def _list_source_outputs():
    """Return parsed JSON list of PulseAudio/PipeWire source-outputs."""
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "source-outputs"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            log.warning("pactl list source-outputs failed: %s", result.stderr.strip())
            return []
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Cannot list source-outputs: %s", e)
        return []


def _get_volume_percent(out):
    """Extract volume percentage from a source-output's volume dict."""
    vol = out.get("volume", {})
    for ch in vol.values():
        if isinstance(ch, dict):
            return ch.get("value_percent", "100%")
    return "100%"


def mute_other_streams(own_pid):
    """Mute all recording streams except the one belonging to own_pid.
    Uses both mute flag and volume=0 for reliable PipeWire/pavucontrol visibility.
    Returns a list of (index, was_muted, original_volume) for later restoration."""
    outputs = _list_source_outputs()
    muted = []
    own_pid_str = str(own_pid)
    for out in outputs:
        props = out.get("properties", {})
        pid = props.get("application.process.id", "")
        if pid == own_pid_str:
            continue
        if props.get("media.name") == "Peak detect":
            continue
        idx = out.get("index")
        if idx is None:
            continue
        was_muted = out.get("mute", False)
        orig_vol = _get_volume_percent(out)
        app = props.get("application.name", f"PID {pid}")
        try:
            subprocess.run(
                ["pactl", "set-source-output-mute", str(idx), "1"],
                check=True,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["pactl", "set-source-output-volume", str(idx), "0%"],
                check=True,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            muted.append((idx, was_muted, orig_vol))
            log.info("Muted recording stream: %s (index %d)", app, idx)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("Failed to mute recording stream %d (%s): %s", idx, app, e)
    return muted


def unmute_streams(muted_list):
    """Restore recording streams that were muted by mute_other_streams()."""
    for idx, was_muted, orig_vol in muted_list:
        try:
            subprocess.run(
                ["pactl", "set-source-output-volume", str(idx), orig_vol],
                check=True,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not was_muted:
                subprocess.run(
                    ["pactl", "set-source-output-mute", str(idx), "0"],
                    check=True,
                    timeout=2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            log.info("Restored recording stream (index %d)", idx)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("Failed to restore recording stream %d: %s", idx, e)


def find_keyboards(exclude=(), target_key_name=None):
    """Find all keyboard devices in /dev/input/, skipping paths in exclude.

    Multiple devices advertise a full key range (e.g. gaming mice expose a
    phantom keyboard interface), so listen on every real keyboard rather
    than guessing which one the user will press the hold key on.

    target_key_name: if the hotkey lives on a non-keyboard device (e.g.
    ThinkPad Extra Buttons with KEY_VOICECOMMAND), that device is also
    included so its events reach the event loop.
    """
    if evdev is None:
        return []
    kbds = []
    for path in evdev.list_devices():
        if path in exclude:
            continue
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            # Device vanished or udev hasn't set permissions yet (hotplug race)
            continue
        if "virtual" in dev.name.lower():
            dev.close()
            continue
        caps = dev.capabilities(verbose=True)
        matched = False
        for (etype, _), codes in caps.items():
            if etype == "EV_KEY":
                key_names = [c[0] if isinstance(c[0], str) else c[0][0] for c in codes]
                if "KEY_A" in key_names and "KEY_ENTER" in key_names:
                    matched = True
                    break
                if target_key_name and target_key_name in key_names:
                    matched = True
                    break
        if matched:
            kbds.append(dev)
        else:
            dev.close()
    return kbds


class AudioBuffer:
    """Continuously reads from parec into a growing buffer so the pipe never blocks."""

    def __init__(self, proc):
        """Start a daemon reader so the parec pipe never blocks."""
        self.proc = proc
        self.samples = bytearray()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        """Append parec stdout to the sample buffer until the pipe closes."""
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            with self.lock:
                self.samples.extend(chunk)

    def snapshot(self):
        """Return a copy of all audio accumulated so far."""
        with self.lock:
            return bytes(self.samples)

    def stop(self):
        """Stop recording and return the final raw audio bytes."""
        self.proc.send_signal(signal.SIGTERM)
        self.thread.join(timeout=2)
        self.proc.wait()
        with self.lock:
            return bytes(self.samples)


def _open_capture():
    """Start parec and return the process for the caller to own."""
    return subprocess.Popen(
        [
            "parec",
            "--format=s16le",
            "--rate=16000",
            "--channels=1",
            "--latency-msec=10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def record_audio():
    """Record audio from the default PipeWire or PulseAudio source."""
    return AudioBuffer(_open_capture())


def stop_recording(audio_buf, vad_threshold=-40):
    """Stop recording and return WAV bytes, or None if no speech detected."""
    raw_audio = audio_buf.stop()
    raw_audio = trim_silence(raw_audio, threshold_db=vad_threshold)
    if not raw_audio:
        return None

    buf = io.BytesIO()
    with wave.Wave_write(buf) as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)
    return buf.getvalue()


def trim_silence(raw_audio, threshold_db=-40, pad_frames=3, frame_ms=20):
    """Trim leading/trailing silence using energy-based VAD."""
    if not raw_audio:
        return raw_audio
    frame_size = int(SAMPLE_RATE * frame_ms / 1000) * SAMPLE_WIDTH
    n_frames = len(raw_audio) // frame_size
    if n_frames == 0:
        return raw_audio

    first_speech = -1
    last_speech = -1
    for i in range(n_frames):
        chunk = raw_audio[i * frame_size : (i + 1) * frame_size]
        samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        db = 20 * math.log10(rms / 32768) if rms > 0 else -100
        if db > threshold_db:
            if first_speech < 0:
                first_speech = i
            last_speech = i

    if first_speech < 0:
        return b""

    start = max(0, first_speech - pad_frames) * frame_size
    end = min(n_frames, last_speech + pad_frames + 1) * frame_size
    return raw_audio[start:end]


async def transcribe_stream(wav_bytes, port, type_delay_ms, language=None):
    """Stream a transcription and type each chunk as it arrives."""
    if aiohttp is None:
        log.error("aiohttp required: pip install aiohttp")
        return False
    url = f"http://127.0.0.1:{port}/transcribe/stream"
    if language:
        url += f"?language={language}"
    typed_any = False
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=wav_bytes) as resp:
                if resp.status != 200:
                    log.warning(
                        "Stream endpoint returned %d, falling back to batch",
                        resp.status,
                    )
                    return await transcribe_batch(wav_bytes, port, language=language)
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if data.get("done"):
                        break
                    chunk = data.get("text", "")
                    if chunk:
                        await type_text_async(chunk, delay_ms=type_delay_ms)
                        typed_any = True
    except _client_errors() as exc:
        log.warning("Streaming transcription failed: %s", exc)
    return typed_any


async def transcribe_batch(wav_bytes, port, language=None):
    """Send audio to the whisper server and return the full text."""
    if aiohttp is None:
        log.error("aiohttp required: pip install aiohttp")
        return ""
    url = f"http://127.0.0.1:{port}/transcribe"
    if language:
        url += f"?language={language}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=wav_bytes) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", "").strip()
                log.error("Server returned %d", resp.status)
                return ""
    except _client_errors() as exc:
        log.warning("Transcription request failed: %s", exc)
        return ""


async def transcribe_chunk(wav_bytes, port, language=None):
    """Send one audio chunk to the whisper.cpp server."""
    if aiohttp is None:
        log.error("aiohttp required: pip install aiohttp")
        return ""
    url = f"http://127.0.0.1:{port}/transcribe/stream"
    if language:
        url += f"?language={language}"
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=wav_bytes) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", "").strip()
                log.error("Stream chunk returned %d", resp.status)
                return ""
    except _client_errors() as exc:
        log.warning("Transcription request failed: %s", exc)
        return ""


async def auto_punctuate(text, port):
    """Ask the server to restore punctuation, or return the original text."""
    if aiohttp is None:
        return text
    url = f"http://127.0.0.1:{port}/punctuate"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"text": text}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", text).strip()
    except _client_errors() as exc:
        log.warning("Auto-punctuate failed: %s", exc)
    return text


async def translate_text(text, target_language, port):
    """Ask the server to translate text, or return the original text."""
    if aiohttp is None:
        return text
    url = f"http://127.0.0.1:{port}/translate"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"text": text, "target_language": target_language},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("text", text).strip()
    except _client_errors() as exc:
        log.warning("Translation failed: %s", exc)
    return text
