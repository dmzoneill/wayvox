# Wayvox

<div align="center">
  <img src="branding/wayvox-mark.svg" alt="Wayvox" width="180"/>
  <p><strong>Private voice, local power.</strong></p>
  <p>On-device dictation for Linux. Speak naturally, keep audio local, and type into any focused Wayland application.</p>
  <p>
    <a href="https://github.com/dmzoneill/wayvox"><img src="https://img.shields.io/github/stars/dmzoneill/wayvox?style=flat&color=5DE4FF" alt="GitHub stars"></a>
    <a href="https://github.com/dmzoneill/wayvox/blob/main/LICENSE"><img src="https://img.shields.io/github/license/dmzoneill/wayvox?color=FF8066" alt="License"></a>
    <img src="https://img.shields.io/badge/platform-Linux-101827" alt="Linux">
    <img src="https://img.shields.io/badge/inference-OpenVINO-5DE4FF" alt="OpenVINO">
  </p>
</div>

---

## What is Wayvox?

Wayvox is a local speech-to-text and push-to-talk dictation stack for Linux. It runs Whisper through OpenVINO and can use Intel NPU, Intel GPU, NVIDIA CUDA, or CPU inference. Audio stays on your machine.

Wayvox integrates with GNOME and KDE, provides a local HTTP API, and includes vocabulary tools for technical terms such as Git, Kubernetes, APIs, and Linux commands.

## Key features

### 🔒 Private by default

Speech processing runs locally. No cloud account or remote transcription service is required.

### ⚡ Hardware acceleration

Use Intel NPU, Intel GPU, NVIDIA CUDA, or CPU backends through one install and configuration flow.

### 🎙️ Push-to-talk dictation

Hold a key to dictate into the focused application. Tap for live streaming transcription and incremental typing.

### 🖥️ Desktop integrations

Use the GNOME Shell extension or KDE Plasma widget for status, settings, models, vocabulary, and diagnostics.

### 🧠 Technical vocabulary

Bias Whisper with project terms and apply replacement rules when the model still mishears a phrase.

### 🔌 Local API

Transcribe files or stream token results over a local HTTP API. See the [API reference](docs/api.md).

---

## Quick start

### Install

```bash
git clone git@github.com:dmzoneill/wayvox.git
cd wayvox
make install
```

### Start

```bash
make start
make status
```

`make install` detects hardware, installs dependencies, downloads the default model, creates a Wayvox checkout path when upgrading an old checkout, migrates config/cache and service names, and enables the primary Wayvox services. The first NPU model load can take several minutes while its compiled cache is created.

### Try dictation

Press and hold Right Ctrl, speak, then release. Wayvox transcribes and types into the focused application.

For service logs:

```bash
make logs
```

---

## Desktop integrations

- **GNOME Shell:** tray controls, preferences, vocabulary editor, model management, and About diagnostics.
- **KDE Plasma:** panel widget, settings, model management, and Language Buddy overlay.

See [development.md](docs/development.md) for installation and preview commands.

## Documentation

- [Architecture](docs/architecture.md)
- [API reference](docs/api.md)
- [Configuration and models](docs/configuration.md)
- [Services and migration](docs/services.md)
- [Development and testing](docs/development.md)
- [Branding guide](branding/README.md)

## Requirements

- Linux with Wayland or X11
- Python 3
- PipeWire audio tools
- An Intel NPU, Intel GPU, NVIDIA GPU, or CPU

Intel NPU users need `/dev/accel/accel0` and a working `intel_vpu` driver. Hardware-specific Python packages are selected by `make install`.

## License

Distributed under the [Apache License 2.0](LICENSE).

<div align="center">
  <sub>Built for local voice, fast hardware, and focused work.</sub>
</div>
