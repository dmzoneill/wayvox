# Wayvox development

## Project layout

- `server_native.py`: primary OpenVINO server.
- `server_whisper_cpp.py`: optional whisper.cpp server.
- `push_to_talk.py`: desktop dictation client.
- `gnome-extension/`: GNOME Shell extension.
- `kde-plasmoid/`: KDE Plasma widget and D-Bus overlay.
- `branding/`: product mark, social preview, and brand guidance.
- `docs/`: user and technical documentation.
- `tests/`: unit, smoke, and integration tests.

## Test suite

Install test dependencies in the active Python environment, then run:

```bash
pytest -q
```

Useful focused runs:

```bash
pytest -q tests/test_diagnostics.py tests/test_vocab.py
pytest -q tests/test_push_to_talk_audio.py tests/test_push_to_talk_output.py
```

The live service path can be checked without importing server dependencies into the system Python:

```bash
make test
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/system
```

## GNOME extension

Install and enable the extension from the repository root:

```bash
make extension-install
make extension-enable
```

Useful development targets:

```bash
make extension-reload
make extension-shell
make extension-logs
```

The extension uses Wayvox IDs and paths. The installer migrates legacy GNOME settings and extension registrations before enabling the new extension.

## KDE plasmoid

Install the plasmoid and its D-Bus service:

```bash
make -C kde-plasmoid install
make -C kde-plasmoid install-service
```

Preview without adding the widget to a panel:

```bash
make -C kde-plasmoid preview
make -C kde-plasmoid preview-dbus
```

## Service unit generation

Service files are generated from the root `Makefile` into `~/.config/systemd/user/`. Do not edit generated units directly. Use Make variables or drop-ins instead.

```bash
make -n migrate-services
make migrate-services
```

## Branding assets

- `branding/wayvox-mark.svg`: app and repository mark.
- `branding/wayvox-social-preview.svg`: GitHub social preview.
- `gnome-extension/icons/wayvox-symbolic.svg`: GNOME symbolic icon.
- `kde-plasmoid/contents/icons/wayvox.svg`: KDE icon.

Keep user-facing product copy as Wayvox. Retain `WHISPER_*` environment variables, Whisper model paths, and D-Bus/API compatibility names. `make migrate-branding` moves legacy config/cache data and desktop integration settings to new Wayvox names.
