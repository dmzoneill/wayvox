# Wayvox API

Primary server listens on `http://127.0.0.1:5000`. The optional whisper.cpp server listens on port `5001`.

## Primary server

| Endpoint                     | Method | Purpose                                           |
| ---------------------------- | ------ | ------------------------------------------------- |
| `/health`                    | GET    | Runtime, model, device, and vocabulary status     |
| `/system`                    | GET    | Version, hardware, libraries, and runtime details |
| `/models`                    | GET    | List installed OpenVINO models                    |
| `/transcribe`                | POST   | Batch transcription with the default model        |
| `/transcribe/<model>`        | POST   | Batch transcription with a named model            |
| `/transcribe/stream`         | POST   | Streaming transcription with the default model    |
| `/transcribe/stream/<model>` | POST   | Streaming transcription with a named model        |

## Batch transcription

```bash
curl -X POST http://127.0.0.1:5000/transcribe \
  --data-binary @audio.wav
```

```json
{ "text": "The transcribed text appears here." }
```

## Streaming transcription

```bash
curl -N -X POST http://127.0.0.1:5000/transcribe/stream \
  --data-binary @audio.wav
```

```text
data: {"text": "The "}
data: {"text": "transcribed "}
data: {"text": "text."}
data: {"done": true, "full_text": "The transcribed text."}
```

## Model selection

```bash
curl http://127.0.0.1:5000/models
curl -X POST http://127.0.0.1:5000/transcribe/whisper-base.en \
  --data-binary @audio.wav
```

## Health and system information

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/system
```

`/health` reports the selected engine, device, model, vocabulary bias mode, and failover state. `/system` reports the Wayvox version, Python runtime, drivers, accelerators, and installed inference libraries.

## whisper.cpp server

The optional `wayvox-cpp-server.service` exposes:

| Endpoint             | Method | Purpose                            |
| -------------------- | ------ | ---------------------------------- |
| `/health`            | GET    | Health check                       |
| `/transcribe`        | POST   | Batch transcription                |
| `/transcribe/stream` | POST   | Transcribe the latest audio window |

Run it manually when needed:

```bash
python3 server-whisper-cpp.py \
  --port 5001 \
  --model ~/.cache/whisper/ggml-base.bin \
  --device NPU
```
