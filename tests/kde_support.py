"""Load wayvox_kde_service.py for tests without a session bus.

The module lives at kde-plasmoid/wayvox_kde_service.py, outside any package,
so this helper imports it by file path after installing dbus and gi mocks.
It also provides the shared QApplication fixture.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

_QtWidgets = importlib.import_module("PyQt6.QtWidgets")
QApplication = _QtWidgets.QApplication


class MockDbusServiceObject:
    """Stand-in for dbus.service.Object when no session bus exists."""

    def __init__(self, *_args, **_kwargs):
        """Accept the bus connection arguments and do nothing."""

    def bind(self):
        """Return this object so the caller can hold the registration."""
        return self

    def object_path(self):
        """Return the Language Buddy object path this stand-in represents."""
        return "/com/whisper/LanguageBuddy"


def _passthrough_method(*_args, **_kwargs):
    """Return a decorator that leaves a D-Bus method unchanged."""

    def decorate(func):
        """Return the method without registering it on a bus."""
        return func

    return decorate


def install_runtime_mocks():
    """Replace dbus and gi with mocks so the service module can import."""
    for name in (
        "dbus",
        "dbus.service",
        "dbus.mainloop",
        "dbus.mainloop.glib",
        "gi",
        "gi.repository",
    ):
        sys.modules.pop(name, None)

    mock_dbus = MagicMock()
    stand_in = MockDbusServiceObject().bind()
    mock_dbus.service = MagicMock()
    mock_dbus.service.Object = MockDbusServiceObject
    mock_dbus.service.method = _passthrough_method
    mock_dbus.service.BusName = MagicMock
    mock_dbus.service.bound_path = stand_in.object_path()
    mock_dbus.SessionBus = MagicMock
    sys.modules["dbus"] = mock_dbus
    sys.modules["dbus.service"] = mock_dbus.service
    sys.modules["dbus.mainloop"] = MagicMock()
    sys.modules["dbus.mainloop.glib"] = MagicMock()

    mock_gi = MagicMock()
    sys.modules["gi"] = mock_gi
    sys.modules["gi.repository"] = mock_gi.repository


def _service_path():
    """Return the path to kde-plasmoid/wayvox_kde_service.py."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "kde-plasmoid", "wayvox_kde_service.py")


def _load_service():
    """Import wayvox_kde_service.py by file path."""
    path = _service_path()
    spec = importlib.util.spec_from_file_location("wayvox_npu_kde_service", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ignore_selected(_text):
    """Accept a card selection without typing it."""


install_runtime_mocks()
kde_svc = _load_service()


@pytest.fixture(scope="session")
def qapp():
    """Create one QApplication for every KDE widget test in the session."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def history_overlay(request):
    """Create a history overlay after the shared QApplication exists."""
    app = request.getfixturevalue("qapp")
    assert app is not None
    return kde_svc.HistoryOverlay()


@pytest.fixture
def service(request):
    """Build a LanguageBuddyService on in-memory overlays."""
    app = request.getfixturevalue("qapp")
    assert app is not None
    overlay = kde_svc.LanguageBuddyOverlay()
    history = kde_svc.HistoryOverlay()
    buddy = kde_svc.LanguageBuddyService(MagicMock(), overlay, history)
    overlay.card_selected.connect(_ignore_selected)
    return buddy
