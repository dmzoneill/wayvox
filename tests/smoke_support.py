"""Fixtures that load Wayvox modules for smoke tests."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def server_smoke():
    """Import the transcription server with filesystem and device libraries mocked."""
    mock_ov = MagicMock()
    mock_librosa = MagicMock()
    saved = {}
    for name in ("openvino_genai", "librosa"):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock_ov if name == "openvino_genai" else mock_librosa
    exists = patch("os.path.exists", return_value=True)
    listing = patch("os.listdir", return_value=["whisper-small-en"])
    directory = patch("os.path.isdir", return_value=True)
    expanduser = patch(
        "os.path.expanduser",
        side_effect=lambda path: path.replace("~", "/tmp/smoke"),
    )
    with exists, listing, directory, expanduser:
        module = importlib.import_module("server_native")
        importlib.reload(module)
    client = module.app.test_client()
    yield module, client
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def kde_service():
    """Load the KDE service module with dbus and gi replaced by mocks."""
    for name in ("dbus", "dbus.service", "dbus.mainloop", "dbus.mainloop.glib"):
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
    mock_gi = MagicMock()
    for name in ("gi", "gi.repository"):
        if name not in sys.modules:
            sys.modules[name] = mock_gi
    spec = importlib.util.spec_from_file_location(
        "wayvox_kde_service",
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "kde-plasmoid",
            "wayvox_kde_service.py",
        ),
        submodule_search_locations=[],
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
