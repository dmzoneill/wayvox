"""Tests for the push-to-talk command line and startup."""

from unittest.mock import MagicMock, patch

import push_to_talk as ptt


def _main(argv, loaded=None):
    """Run ptt.main with the event loop and config file stubbed out."""
    with (
        patch.object(ptt.dictation_audio, "aiohttp", MagicMock()),
        patch.object(ptt, "load_config", return_value=loaded),
        patch.object(ptt, "apply_config") as apply,
        patch.object(ptt, "run_dictation", MagicMock(return_value="coro")) as start,
        patch.object(ptt.asyncio, "run") as run,
        patch.object(ptt, "build_settings", side_effect=lambda args: vars(args).copy()),
    ):
        assert ptt.main(argv) == 0
    return run.call_args, apply, start


def test_defaults():
    call, apply, _start = _main([])
    apply.assert_not_called()
    assert call.args == ("coro",)


def test_overrides_and_config():
    argv = ["--backend", "whisper-cpp", "--port", "5100", "--language", "de"]
    _call, apply, start = _main(argv, loaded={"sound": False})
    settings = start.call_args.args[0]
    assert settings["backend"] == "whisper-cpp"
    assert settings["port"] == 5100
    assert settings["language"] == "de"
    apply.assert_called_once()


def test_requires_aiohttp():
    with patch.object(ptt.dictation_audio, "aiohttp", None):
        assert ptt.main([]) == 1
