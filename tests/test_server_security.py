"""Tests for the bind-host and cross-origin guard helpers."""

from flask import Flask

import server_security


def _client():
    app = server_security.install_csrf_guard(Flask(__name__))

    @app.route("/thing", methods=["GET", "POST"])
    def thing():
        return "ok"

    return app.test_client()


def test_bind_host_defaults_to_loopback(monkeypatch):
    """Bind host defaults to loopback when unset."""
    monkeypatch.delenv("WAYVOX_BIND_HOST", raising=False)
    assert server_security.bind_host() == "127.0.0.1"


def test_bind_host_env_and_override(monkeypatch):
    """Env var and explicit argument override the bind host."""
    monkeypatch.setenv("WAYVOX_BIND_HOST", "0.0.0.0")
    assert server_security.bind_host() == "0.0.0.0"
    assert server_security.bind_host("192.168.1.5") == "192.168.1.5"


def test_native_client_post_allowed():
    """POST without Origin headers is allowed."""
    assert _client().post("/thing").status_code == 200


def test_same_origin_post_allowed():
    """Same-origin POST is allowed."""
    resp = _client().post("/thing", headers={"Origin": "http://localhost"})
    assert resp.status_code == 200


def test_cross_origin_post_rejected():
    """Cross-origin POST is rejected."""
    resp = _client().post("/thing", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 403


def test_cross_site_fetch_metadata_rejected():
    """Cross-site Sec-Fetch-Site POST is rejected."""
    resp = _client().post("/thing", headers={"Sec-Fetch-Site": "cross-site"})
    assert resp.status_code == 403


def test_cross_origin_get_allowed():
    """Cross-origin GET is allowed."""
    resp = _client().get("/thing", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
