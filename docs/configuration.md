# Wayvox configuration

## Install-time variables

Override defaults with `make install VAR=value`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_DEVICE` | `auto` | OpenVINO device: `NPU`, `GPU`, or `CPU` |
| `WHISPER_MODEL` | `whisper-small.en-fp16-ov` | Default OpenVINO model |
| `WHISPER_CPP_DEVICE` | `NPU` | whisper.cpp encoder device |
| `WHISPER_CPP_PORT` | `5001` | whisper.cpp server port |
| `WHISPER_NPU_WORD_TIMESTAMPS` | `1` | Compile a wider NPU prompt slot |
| `WHISPER_CACHE_DIR` | `~/.cache/wayvox` | Compiled model cache |

Example:

```bash
make install WHISPER_DEVICE=CPU WHISPER_MODEL=whisper-base.en
```

Technical environment variables and paths retain their `WHISPER_*` names for compatibility.

## Models

OpenVINO models live in `~/.whisper/models/`. The default model is downloaded by `make install`.

Install additional OpenVINO models:

```bash
cd ~/.whisper/models
for model in whisper-small.en-fp16-ov whisper-base.en whisper-tiny.en; do
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/OpenVINO/$model
  cd "$model" && git lfs pull && cd ..
done
```

The optional whisper.cpp model cache is `~/.cache/whisper/`.

## Vocabulary

Wayvox combines prompt biasing with regex replacements. The bundled vocabulary is installed at:

```text
~/.config/wayvox/vocabulary.json
```

Existing files are never overwritten by `make install`. Edit vocabulary from the GNOME or KDE UI, or edit the JSON file directly. Changes apply on the next transcription.

Replacement rules use this format:

```text
\bpolar quests?\b => pull request
```

Append `[cs]` for case-sensitive patterns. Invalid patterns are rejected on save.

## NPU prompt slot

With `WHISPER_NPU_WORD_TIMESTAMPS=1`, the NPU pipeline compiles `word_timestamps=True` and uses a 447-token prompt slot. The first model load can take several minutes. Later starts reuse `WHISPER_CACHE_DIR`.

Set `WHISPER_NPU_WORD_TIMESTAMPS=0` to keep the four-token slot. Do not set `MAX_PROMPT_LEN`; the NPU encoder rejects that option. If an OpenVINO build rejects prompt bias arguments, Wayvox falls back to regex replacements and reports the state through `/health`.

## About and diagnostics

The GNOME and KDE About views query `/system` from the active server. The response includes Wayvox version, runtime engine, device, accelerators, drivers, Python, library versions, vocabulary state, and failover guidance.

Run diagnostics directly:

```bash
python3 diagnostics.py
python3 about.py --print
```

## Dependencies

Python requirements are split by hardware path:

- `requirements-base.txt`: common runtime packages.
- `requirements-openvino.txt`: OpenVINO NPU, GPU, and CPU support.
- `requirements-cuda.txt`: NVIDIA CUDA support.

System dependencies include PipeWire audio tools, `ydotool`, `wtype`, `wl-clipboard`, `xdotool`, `evdev`, and build tools for whisper.cpp.
