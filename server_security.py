"""Shared network-safety helpers for the wayvox HTTP servers.

The servers are local JSON APIs used by the dictation client, the GNOME
extension and the KDE plasmoid. None of them use cookies or browser sessions,
so the realistic CSRF risk is a web page in the user's browser posting to
http://127.0.0.1:<port>. ``install_csrf_guard`` rejects state-changing requests
that a browser marks as cross-origin, while native clients (which send no
Origin / Sec-Fetch-Site headers) keep working.
"""

import os
from urllib.parse import urlsplit

from flask import abort, request

DEFAULT_BIND_HOST = "127.0.0.1"
BIND_HOST_ENV = "WAYVOX_BIND_HOST"

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_CROSS_SITE_FETCH = frozenset({"cross-site", "same-site"})


def bind_host(override=None):
    """Return the address to listen on.

    Defaults to loopback. Set ``WAYVOX_BIND_HOST`` (or pass ``override``) to
    ``0.0.0.0`` or a specific address to serve other machines on the network.
    """
    if override:
        return override
    return os.environ.get(BIND_HOST_ENV, "").strip() or DEFAULT_BIND_HOST


def _origin_matches_host(origin, host):
    parts = urlsplit(origin)
    return bool(parts.netloc) and parts.netloc.lower() == (host or "").lower()


def is_cross_origin_request():
    """True when a browser indicates the request came from another origin."""
    if request.method in _SAFE_METHODS:
        return False
    if request.headers.get("Sec-Fetch-Site", "").lower() in _CROSS_SITE_FETCH:
        return True
    origin = request.headers.get("Origin")
    if origin is None:
        return False
    return not _origin_matches_host(origin, request.host)


def install_csrf_guard(app):
    """Reject cross-origin state-changing browser requests with 403."""

    @app.before_request
    def _reject_cross_origin():
        if is_cross_origin_request():
            abort(403, description="Cross-origin requests are not allowed")

    return app
