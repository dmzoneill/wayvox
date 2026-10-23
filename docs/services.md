# Wayvox services

Wayvox uses systemd user services. The primary workflow needs `wayvox-server.service` and `wayvox-push-to-talk.service`. The whisper.cpp service is optional.

## Service names

| Service | Role | Default |
| --- | --- | --- |
| `wayvox-server.service` | OpenVINO Whisper server on port 5000 | Enabled |
| `wayvox-push-to-talk.service` | Keyboard dictation client | Enabled |
| `wayvox-cpp-server.service` | whisper.cpp fallback on port 5001 | Disabled |
| `wayvox-kde.service` | KDE Language Buddy D-Bus service | Optional |

## Install

```bash
make install
```

The installer detects hardware, installs packages, downloads the default model, installs vocabulary, generates service units, migrates legacy service names, and enables the primary workflow.

Start, stop, inspect, and test services:

```bash
make start
make stop
make status
make test
make logs
```

Target-specific logs:

```bash
make logs-server
make logs-ptt
make logs-cpp
```

## Migrate an existing install

Run this after pulling the Wayvox rebrand into an existing checkout:

```bash
make migrate-services
systemctl --user enable wayvox-server.service wayvox-push-to-talk.service
make start
```

Migration:

- Disables and removes `whisper-server.service`, `whisper-cpp-server.service`, `push-to-talk.service`, and the old KDE unit.
- Moves service drop-ins to their `wayvox-*` names.
- Preserves enabled and active states.
- Keeps `WHISPER_*` variables, model paths, and configuration paths unchanged.

`make install-services` and `make install` run the same migration automatically.

## Configuration overrides

The GNOME and KDE integrations write systemd drop-ins. Manual overrides live under:

```text
~/.config/systemd/user/wayvox-server.service.d/
~/.config/systemd/user/wayvox-push-to-talk.service.d/
```

Reload after editing:

```bash
systemctl --user daemon-reload
systemctl --user restart wayvox-server.service
systemctl --user restart wayvox-push-to-talk.service
```

## Uninstall

```bash
make uninstall
```

This stops, disables, and removes current and legacy Wayvox service units. Model and cache data are kept unless removed with `make clean`.
