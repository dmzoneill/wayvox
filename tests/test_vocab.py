"""Tests for technical vocabulary bias and mishear replacements."""

import json

import pytest

import vocab


@pytest.fixture
def vocab_path(tmp_path, monkeypatch):
    path = tmp_path / "vocabulary.json"
    monkeypatch.setenv("WHISPER_VOCAB_PATH", str(path))
    with vocab._lock:
        vocab._state["key"] = None
    yield path
    with vocab._lock:
        vocab._state["key"] = None


def test_polar_quest_becomes_pull_request(vocab_path):
    assert vocab.apply("open a polar quest please") == "open a pull request please"
    assert vocab.apply("Polar Quest merged") == "Pull request merged"
    assert vocab.apply("POLAR QUEST") == "PULL REQUEST"


def test_other_computer_mishears(vocab_path):
    assert vocab.apply("kube cuddle get logs") == "kubectl get logs"
    assert vocab.apply("cooper netties cluster") == "Kubernetes cluster"
    assert vocab.apply("engine x config") == "nginx config"
    assert vocab.apply("jay son file") == "JSON file"
    assert vocab.apply("the sequel query") == "the SQL query"


def test_ordinary_words_stay(vocab_path):
    text = "the red barn is bright and the sequel is not here"
    # "sequel" is intentionally mapped to SQL. Other prose stays.
    assert vocab.apply("the red barn is bright") == "the red barn is bright"
    assert "barn" in text


def test_disabled_skips_rules_and_prompt(vocab_path):
    vocab.save({"enabled": False, "terms": ["pull request"], "replacements": [
        {"pattern": r"\bpolar quest\b", "replacement": "pull request"}
    ]})
    assert vocab.apply("polar quest") == "polar quest"
    assert vocab.bias() == {"initial_prompt": "", "hotwords": ""}


def test_prompt_includes_priority_terms_and_stays_short(vocab_path):
    prompts = vocab.bias()
    assert prompts["initial_prompt"].startswith("Technical computer dictation")
    assert "pull request" in prompts["initial_prompt"]
    assert "GitHub" in prompts["initial_prompt"]
    assert len(prompts["initial_prompt"]) <= vocab.PROMPT_CHAR_BUDGET
    assert "kubectl" in prompts["hotwords"]
    assert len(prompts["hotwords"]) <= vocab.HOTWORD_CHAR_BUDGET


def test_custom_file_round_trip(vocab_path):
    rules, errors = vocab.parse_rules(r"\bfoo bar\b => baz  [cs]")
    assert errors == []
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
    bad = {"enabled": True, "terms": ["pull request"], "replacements": [
        {"pattern": r"(", "replacement": "x"}
    ]}
    with pytest.raises(ValueError):
        vocab.save(bad)
    vocab_path.write_text(json.dumps(bad))
    with vocab._lock:
        vocab._state["key"] = None
    # Broken file rule is skipped. Built-in rules are not mixed back in.
    assert vocab.apply("polar quest") == "polar quest"


def test_missing_file_uses_builtin_defaults(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "vocabulary.json"
    monkeypatch.setenv("WHISPER_VOCAB_PATH", str(missing))
    with vocab._lock:
        vocab._state["key"] = None
    info = vocab.describe()
    assert info["using_defaults"] is True
    assert info["terms"] > 40
    assert info["replacements"] > 20
    assert vocab.apply("polar quest") == "pull request"


def test_generate_biased_drops_rejected_hotwords(vocab_path, monkeypatch):
    monkeypatch.setattr(vocab, "_rejected_bias", set())

    def backend(*_args, **kwargs):
        if "hotwords" in kwargs:
            raise TypeError("unexpected keyword argument 'hotwords'")
        assert "initial_prompt" in kwargs
        assert kwargs["language"] == "<|en|>"
        return "ok"

    assert vocab.generate_biased(backend, language="<|en|>") == "ok"


def test_parse_rules_reports_bad_lines():
    _rules, errors = vocab.parse_rules("not a rule\n\\bfoo\\b => bar")
    assert errors
    assert "line 1" in errors[0]


def test_bundled_vocab_matches_builtin_defaults():
    bundled = json.loads(open(vocab.bundled_path(), encoding="utf-8").read())
    assert vocab.normalize(bundled) == vocab.default_config()


def test_install_default_writes_once_and_does_not_clobber(tmp_path):
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
    dest = tmp_path / "vocabulary.json"
    monkeypatch.setattr(vocab, "bundled_path", lambda: str(tmp_path / "missing.json"))
    assert vocab.install_default(str(dest)) is True
    assert json.loads(dest.read_text())["terms"] == list(vocab.DEFAULT_TERMS)
