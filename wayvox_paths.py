"""Wayvox user-data paths with one-release legacy fallbacks."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR = Path(os.path.expanduser(os.environ.get("WAYVOX_CONFIG_DIR", "~/.config/wayvox")))
LEGACY_CONFIG_DIR = Path(os.path.expanduser("~/.config/whisper-npu"))
CACHE_DIR = Path(os.path.expanduser("~/.cache/wayvox"))
LEGACY_CACHE_DIR = Path(os.path.expanduser("~/.cache/whisper-npu"))


def current_config_dir() -> Path:
    return Path(os.path.expanduser(os.environ.get("WAYVOX_CONFIG_DIR", "~/.config/wayvox")))


def legacy_config_dir() -> Path:
    return Path(os.path.expanduser("~/.config/whisper-npu"))


def config_file(name: str) -> Path:
    """Return new config path, falling back to legacy data until migrated."""
    current = current_config_dir() / name
    legacy = legacy_config_dir() / name
    if current.exists() or not legacy.exists():
        return current
    return legacy


def history_file() -> Path:
    return config_file("history.db")


def vocabulary_file() -> Path:
    return config_file("vocabulary.json")


def app_contexts_file() -> Path:
    return config_file("app-contexts.json")


def cache_dir() -> str:
    """Return configured cache, defaulting to Wayvox's cache directory."""
    raw = os.environ.get("WHISPER_CACHE_DIR")
    if raw is None:
        return str(CACHE_DIR)
    raw = raw.strip()
    if not raw:
        return ""
    return os.path.expanduser(raw)
