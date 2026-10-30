"""Tests for technical vocabulary bias and mishear replacements."""

import json
import logging

import pytest
import vocab


@pytest.fixture(name="vocab_path")
def _fixture_vocab_path(tmp_path, monkeypatch):
    """Vocab path."""
    path = tmp_path / "vocabulary.json"
    monkeypatch.setenv("WHISPER_VOCAB_PATH", str(path))
    with vocab.STATE.lock:
        vocab.STATE.cache["key"] = None
    yield path
    with vocab.STATE.lock:
        vocab.STATE.cache["key"] = None


@pytest.mark.usefixtures("vocab_path")
def test_polar_quest_becomes_pull_request():
    """Test that polar quest becomes pull request."""
    assert vocab.apply("open a polar quest please") == "open a pull request please"
    assert vocab.apply("Polar Quest merged") == "Pull request merged"
    assert vocab.apply("POLAR QUEST") == "PULL REQUEST"


@pytest.mark.usefixtures("vocab_path")
def test_other_computer_mishears():
    """Test that other computer mishears."""
    assert vocab.apply("kube cuddle get logs") == "kubectl get logs"
    assert vocab.apply("cooper netties cluster") == "Kubernetes cluster"
    assert vocab.apply("engine x config") == "nginx config"
    assert vocab.apply("jay son file") == "JSON file"
    assert vocab.apply("the sequel query") == "the SQL query"


@pytest.mark.usefixtures("vocab_path")
def test_ordinary_words_stay():
    """Test that ordinary words stay."""
    text = "the red barn is bright and the sequel is not here"
    # "sequel" is intentionally mapped to SQL. Other prose stays.
    assert vocab.apply("the red barn is bright") == "the red barn is bright"
    assert "barn" in text


@pytest.mark.usefixtures("vocab_path")
def test_disabled_skips_rules_and_prompt():
    """Test that disabled skips rules and prompt."""
    vocab.save(
        {
            "enabled": False,
            "terms": ["pull request"],
            "replacements": [
                {
                    "pattern": r"\bpolar quest\b",
                    "replacement": "pull request",
                },
            ],
        },
    )
    assert vocab.apply("polar quest") == "polar quest"
    assert vocab.bias() == {"initial_prompt": "", "hotwords": ""}


@pytest.mark.usefixtures("vocab_path")
def test_prompt_includes_priority_terms_and_stays_short():
    """Test that prompt includes priority terms and stays short."""
    prompts = vocab.bias()
    assert prompts["initial_prompt"].startswith("Technical computer dictation")
    assert "pull request" in prompts["initial_prompt"]
    assert "GitHub" in prompts["initial_prompt"]
    assert len(prompts["initial_prompt"]) <= vocab.PROMPT_CHAR_BUDGET
    assert "kubectl" in prompts["hotwords"]
    assert len(prompts["hotwords"]) <= vocab.HOTWORD_CHAR_BUDGET


def test_custom_file_round_trip(vocab_path):
    """Test that custom file round trip."""
    rules, errors = vocab.parse_rules(r"\bfoo bar\b => baz  [cs]")
    assert not errors
    assert rules[0]["ignore_case"] is False
    config = {
        "enabled": True,
        "preamble": "Only these:",
        "terms": ["baz", "baz", "qux"],
        "replacements": rules,
    }
    vocab.save(config)
    loaded = json.loads(vocab_path.read_text())
    assert loaded["terms"] == ["baz", "qux"]
    assert loaded["replacements"][0]["pattern"] == r"\bfoo bar\b"
    assert vocab.apply("foo bar") == "baz"
    assert vocab.apply("Foo Bar") == "Foo Bar"
    assert vocab.rules_to_text(loaded["replacements"]).endswith("[cs]")


def test_invalid_regex_is_refused_and_skipped(vocab_path):
    """Test that invalid regex is refused and skipped."""
    bad = {
        "enabled": True,
        "terms": ["pull request"],
        "replacements": [{"pattern": r"(", "replacement": "x"}],
    }
    with pytest.raises(ValueError):
        vocab.save(bad)
    vocab_path.write_text(json.dumps(bad))
    with vocab.STATE.lock:
        vocab.STATE.cache["key"] = None
    # Broken file rule is skipped. Built-in rules are not mixed back in.
    assert vocab.apply("polar quest") == "polar quest"


def test_missing_file_uses_builtin_defaults(tmp_path, monkeypatch):
    """Test that missing file uses builtin defaults."""
    missing = tmp_path / "nope" / "vocabulary.json"
    monkeypatch.setenv("WHISPER_VOCAB_PATH", str(missing))
    with vocab.STATE.lock:
        vocab.STATE.cache["key"] = None
    info = vocab.describe()
    assert info["using_defaults"] is True
    assert info["terms"] > 40
    assert info["replacements"] > 20
    assert vocab.apply("polar quest") == "pull request"


@pytest.mark.usefixtures("vocab_path")
def test_generate_biased_drops_rejected_hotwords(monkeypatch):
    """Test that generate biased drops rejected hotwords."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())

    def backend(*_args, **kwargs):
        """Backend."""
        if "hotwords" in kwargs:
            raise TypeError("unexpected keyword argument 'hotwords'")
        assert "initial_prompt" in kwargs
        assert kwargs["language"] == "<|en|>"
        return "ok"

    assert vocab.generate_biased(backend, language="<|en|>") == "ok"


@pytest.mark.usefixtures("vocab_path")
def test_generate_biased_retries_when_npu_rejects_prompt(monkeypatch):
    """Test that generate biased retries when npu rejects prompt."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())
    calls = []

    def backend(*_args, **kwargs):
        """Backend."""
        calls.append(set(kwargs))
        if "initial_prompt" in kwargs:
            raise RuntimeError(
                "Check failed: 'initial_prompt' parameter is not supported on NPU device.",
            )
        if "hotwords" in kwargs:
            raise RuntimeError("'hotwords' parameter is not supported on NPU device.")
        assert kwargs["language"] == "<|en|>"
        return "ok"

    assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    assert calls[-1] == {"language"}
    assert vocab.STATE.rejected_bias == {"initial_prompt", "hotwords"}


@pytest.mark.usefixtures("vocab_path")
def test_configure_backend_keeps_npu_bias_until_failover(
    monkeypatch,
    caplog,
):
    """Test that configure backend keeps npu bias until failover."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())
    monkeypatch.setattr(vocab.STATE, "failover_logged", set())
    monkeypatch.setattr(vocab.STATE, "failover_reasons", [])
    monkeypatch.setattr(vocab.STATE, "bias_confirmed", False)
    vocab.configure_backend("CPU")
    assert vocab.STATE.rejected_bias == set()
    with caplog.at_level(logging.INFO, logger="vocab"):
        vocab.configure_backend("NPU")
    assert "first transcription" in caplog.text
    assert "Failover" not in caplog.text
    assert vocab.bias_status()["mode"] == "pending"

    def backend(*_args, **kwargs):
        """Backend."""
        assert "initial_prompt" in kwargs
        assert "hotwords" in kwargs
        return "ok"

    assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    assert vocab.bias_status()["mode"] == "on"


@pytest.mark.usefixtures("vocab_path")
def test_npu_reject_log_names_upgrade(monkeypatch, caplog):
    """Test that npu reject log names upgrade."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())
    monkeypatch.setattr(vocab.STATE, "failover_logged", set())
    monkeypatch.setattr(vocab.STATE, "failover_reasons", [])
    vocab.configure_backend("CPU")

    def backend(*_args, **kwargs):
        """Backend."""
        if "initial_prompt" in kwargs or "hotwords" in kwargs:
            raise RuntimeError(
                "Check failed: 'initial_prompt' parameter is not supported on NPU device.",
            )
        return "ok"

    with caplog.at_level(logging.WARNING, logger="vocab"):
        assert vocab.generate_biased(backend, language="<|en|>") == "ok"
        assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].startswith("Failover:")
    assert "2026.4.1" in warnings[0]
    assert "regex" in warnings[0].lower()
    assert "restart wayvox-server" in warnings[0]
    assert vocab.bias_status()["mode"] == "regex"
    assert vocab.bias_status()["failover"] == ["unsupported"]


@pytest.mark.usefixtures("vocab_path")
def test_prompt_slot_log_names_max_prompt_len(monkeypatch, caplog):
    """Test that prompt slot log names max prompt len."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())
    monkeypatch.setattr(vocab.STATE, "failover_logged", set())
    monkeypatch.setattr(vocab.STATE, "failover_reasons", [])
    vocab.configure_backend("NPU")

    def backend(*_args, **kwargs):
        """Backend."""
        if "initial_prompt" in kwargs or "hotwords" in kwargs:
            raise RuntimeError("Check '*roi_end <= *max_dim' failed")
        return "ok"

    with caplog.at_level(logging.WARNING, logger="vocab"):
        assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    text = caplog.text
    assert "Failover:" in text
    assert "4 tokens" in text
    assert "MAX_PROMPT_LEN" in text
    assert "replacement rule" in text
    assert vocab.bias_status()["mode"] == "regex"
    assert vocab.bias_status()["failover"] == ["prompt_slot"]


def test_load_whisper_pipeline_falls_back_when_static_flag_rejected(
    monkeypatch,
    caplog,
    tmp_path,
):
    """Test that load whisper pipeline falls back when static flag rejected."""
    cache_dir = str(tmp_path / "cache")
    monkeypatch.setenv("WHISPER_NPU_WORD_TIMESTAMPS", "1")
    monkeypatch.setenv("WHISPER_CACHE_DIR", cache_dir)
    monkeypatch.setattr(vocab.STATE, "failover_logged", set())
    monkeypatch.setattr(vocab.STATE, "failover_reasons", [])
    calls = []

    def factory(_path, **kwargs):
        """Factory."""
        calls.append(kwargs)
        if "STATIC_PIPELINE" in kwargs:
            raise TypeError("unexpected keyword argument 'STATIC_PIPELINE'")
        return "pipe"

    with caplog.at_level(logging.WARNING, logger="vocab"):
        assert vocab.load_whisper_pipeline(factory, "/m", "NPU") == "pipe"
    wide = {
        "device": "NPU",
        "word_timestamps": True,
        "CACHE_DIR": cache_dir,
    }
    assert calls == [
        {**wide, "STATIC_PIPELINE": False},
        wide,
    ]
    assert "Failover:" in caplog.text
    assert "2026.4.1" in caplog.text
    assert vocab.bias_status()["pipeline"] == "default"
    assert vocab.bias_status()["failover"] == ["static_pipeline"]


def test_load_whisper_pipeline_keeps_stateful_on_npu(monkeypatch):
    """Test that load whisper pipeline keeps stateful on npu."""
    monkeypatch.setattr(vocab.STATE, "failover_logged", set())
    monkeypatch.setattr(vocab.STATE, "failover_reasons", [])

    def factory(_path, **kwargs):
        """Factory."""
        assert kwargs["STATIC_PIPELINE"] is False
        return "pipe"

    assert vocab.load_whisper_pipeline(factory, "/m", "NPU") == "pipe"
    assert vocab.bias_status()["pipeline"] == "stateful"


def test_load_whisper_pipeline_cpu_skips_static_flag():
    """Test that load whisper pipeline cpu skips static flag."""

    def factory(_path, **kwargs):
        """Factory."""
        assert "STATIC_PIPELINE" not in kwargs
        return "pipe"

    assert vocab.load_whisper_pipeline(factory, "/m", "CPU") == "pipe"
    assert vocab.bias_status()["pipeline"] == "default"


def test_load_whisper_pipeline_reraises_unrelated():
    """Test that load whisper pipeline reraises unrelated."""

    def factory(_path, **kwargs):
        """Factory."""
        raise RuntimeError("npu missing")

    with pytest.raises(RuntimeError, match="npu missing"):
        vocab.load_whisper_pipeline(factory, "/m", "NPU")


def test_load_skips_word_timestamps_when_disabled(monkeypatch):
    """Test that load skips word timestamps when disabled."""
    monkeypatch.setenv("WHISPER_NPU_WORD_TIMESTAMPS", "0")
    monkeypatch.setenv("WHISPER_CACHE_DIR", "")

    def factory(_path, **kwargs):
        """Factory."""
        assert "word_timestamps" not in kwargs
        assert "CACHE_DIR" not in kwargs
        assert kwargs["STATIC_PIPELINE"] is False
        return "pipe"

    assert vocab.load_whisper_pipeline(factory, "/m", "NPU") == "pipe"
    status = vocab.bias_status()
    assert status["word_timestamps"] is False
    assert status["word_timestamps_requested"] is False
    assert status["pipeline"] == "stateful"


def test_load_drops_word_timestamps_when_rejected(monkeypatch, caplog):
    """Test that load drops word timestamps when rejected."""
    monkeypatch.setenv("WHISPER_NPU_WORD_TIMESTAMPS", "1")
    monkeypatch.setenv("WHISPER_CACHE_DIR", "")
    monkeypatch.setattr(vocab.STATE, "failover_logged", set())
    monkeypatch.setattr(vocab.STATE, "failover_reasons", [])
    calls = []

    def factory(_path, **kwargs):
        """Factory."""
        calls.append(set(kwargs))
        if kwargs.get("word_timestamps"):
            raise TypeError("unexpected keyword argument 'word_timestamps'")
        return "pipe"

    with caplog.at_level(logging.WARNING, logger="vocab"):
        assert vocab.load_whisper_pipeline(factory, "/m", "NPU") == "pipe"
    assert "word_timestamps" not in calls[-1]
    assert "STATIC_PIPELINE" in calls[-1]
    assert vocab.bias_status()["word_timestamps"] is False
    assert vocab.bias_status()["failover"] == ["word_timestamps"]
    assert "Failover:" in caplog.text
    assert "2026.4.1" in caplog.text


@pytest.mark.usefixtures("vocab_path")
def test_generate_biased_retries_when_npu_prompt_slot_is_short(monkeypatch):
    """Test that generate biased retries when npu prompt slot is short."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())
    calls = []

    def backend(*_args, **kwargs):
        """Backend."""
        calls.append(set(kwargs))
        if "initial_prompt" in kwargs or "hotwords" in kwargs:
            raise RuntimeError("Check '*roi_end <= *max_dim' failed")
        return "ok"

    assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    assert calls[-1] == {"language"}


@pytest.mark.usefixtures("vocab_path")
def test_generate_biased_sends_prompt_and_hotwords(monkeypatch):
    """Test that generate biased sends prompt and hotwords."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())
    seen = {}

    def backend(*_args, **kwargs):
        """Backend."""
        seen.update(kwargs)
        return "ok"

    assert vocab.generate_biased(backend, language="<|en|>") == "ok"
    assert seen["initial_prompt"]
    assert "pull request" in seen["hotwords"]


@pytest.mark.usefixtures("vocab_path")
def test_generate_biased_does_not_swallow_other_errors(monkeypatch):
    """Test that generate biased does not swallow other errors."""
    monkeypatch.setattr(vocab.STATE, "rejected_bias", set())

    def backend(*_args, **kwargs):
        """Backend."""
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        vocab.generate_biased(backend)


def test_parse_rules_reports_bad_lines():
    """Test that parse rules reports bad lines."""
    _rules, errors = vocab.parse_rules("not a rule\n\\bfoo\\b => bar")
    assert errors
    assert "line 1" in errors[0]


def test_bundled_vocab_matches_builtin_defaults():
    """Test that bundled vocab matches builtin defaults."""
    with open(vocab.bundled_path(), encoding="utf-8") as handle:
        bundled = json.load(handle)
    assert vocab.normalize(bundled) == vocab.default_config()


def test_install_default_writes_once_and_does_not_clobber(tmp_path):
    """Test that install default writes once and does not clobber."""
    dest = tmp_path / "vocabulary.json"
    assert vocab.install_default(str(dest)) is True
    loaded = json.loads(dest.read_text())
    assert loaded["terms"][0] == "pull request"
    assert any(rule["replacement"] == "pull request" for rule in loaded["replacements"])
    assert vocab.install_default(str(dest)) is False
    custom = {
        "enabled": True,
        "preamble": "mine",
        "terms": ["only-mine"],
        "replacements": [],
    }
    dest.write_text(json.dumps(custom))
    assert vocab.install_default(str(dest)) is False
    assert json.loads(dest.read_text())["terms"] == ["only-mine"]


def test_install_default_falls_back_when_bundle_missing(tmp_path, monkeypatch):
    """Test that install default falls back when bundle missing."""
    dest = tmp_path / "vocabulary.json"
    monkeypatch.setattr(vocab, "bundled_path", lambda: str(tmp_path / "missing.json"))
    assert vocab.install_default(str(dest)) is True
    assert json.loads(dest.read_text())["terms"] == list(vocab.DEFAULT_TERMS)
