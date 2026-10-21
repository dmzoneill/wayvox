"""Technical vocabulary for Whisper dictation.

Terms bias the decoder through initial_prompt and hotwords on devices
that accept them. OpenVINO NPU compiles a 4 token prompt slot, which is
too small for this list, so those arguments are not sent there. Replacement
rules still fix mishears the model emits, such as "polar quest" -> "pull request".

The editable file is ~/.config/whisper-npu/vocabulary.json. make install
copies share/vocabulary.json there when that file is missing, and leaves an
existing file alone. If the user file is still missing, the built-in lists
below are used. The tray editor writes the file. The server reloads it when
the mtime changes.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading

log = logging.getLogger("vocab")

# Whisper clips the initial prompt to about half the text context. A short
# sentence biases better than a huge list, which the model may echo.
PROMPT_CHAR_BUDGET = 320
HOTWORD_CHAR_BUDGET = 1600
MAX_PATTERN_LEN = 200

DEFAULT_PREAMBLE = "Technical computer dictation. Terms:"

# Order matters. The prompt keeps the front of this list. Hotwords keep more.
DEFAULT_TERMS = (
    "pull request",
    "merge request",
    "GitHub",
    "GitLab",
    "Bitbucket",
    "Kubernetes",
    "kubectl",
    "Docker",
    "Dockerfile",
    "docker compose",
    "nginx",
    "PostgreSQL",
    "JSON",
    "YAML",
    "TOML",
    "regex",
    "localhost",
    "OAuth",
    "pytest",
    "TypeScript",
    "JavaScript",
    "Python",
    "Rust",
    "Go",
    "Node.js",
    "npm",
    "pnpm",
    "pip",
    "API",
    "REST",
    "GraphQL",
    "gRPC",
    "SQL",
    "SQLite",
    "MySQL",
    "Redis",
    "MongoDB",
    "CI/CD",
    "OpenVINO",
    "Whisper",
    "NPU",
    "CUDA",
    "PyTorch",
    "NumPy",
    "Hugging Face",
    "OpenAI",
    "FastAPI",
    "Django",
    "Flask",
    "React",
    "Vue",
    "webhook",
    "middleware",
    "gitignore",
    "cherry-pick",
    "rebase",
    "commit",
    "branch",
    "sudo",
    "chmod",
    "chown",
    "systemd",
    "SSH",
    "UTF-8",
    "base64",
    "SHA-256",
    "stdin",
    "stdout",
    "stderr",
    "Terraform",
    "Ansible",
    "Helm",
    "Minikube",
    "Podman",
    "Flatpak",
    "ConfigMap",
    "namespace",
    "ingress",
    "Elasticsearch",
    "Kafka",
    "RabbitMQ",
    "Prometheus",
    "Grafana",
    "Neovim",
    "VS Code",
    "Wayland",
    "PipeWire",
    "segfault",
    "stack trace",
    "breakpoint",
    "linter",
    "async",
    "boolean",
    "microservice",
    "repository",
    "kernel",
    "filesystem",
    "endpoint",
    "CORS",
    "JWT",
    "WebSocket",
    "protobuf",
    "cron",
    "cluster",
    "container",
)

# Patterns are the spoken mishear. Replacements are the spelling we want.
# Kept specific so ordinary English is left alone.
_DEFAULT_RULE_PAIRS = (
    (r"\bpolar quests?\b", "pull request"),
    (r"\bpuller quests?\b", "pull request"),
    (r"\bpull quests?\b", "pull request"),
    (r"\bpool requests?\b", "pull request"),
    (r"\bpulsar quests?\b", "pull request"),
    (r"\bpolar requests?\b", "pull request"),
    (r"\bmerge a quest\b", "merge request"),
    (r"\bget ?hub\b", "GitHub"),
    (r"\bgit hub\b", "GitHub"),
    (r"\bgit lab\b", "GitLab"),
    (r"\bbit bucket\b", "Bitbucket"),
    (r"\b(?:kube|cube) (?:cuddle|control|ctl)\b", "kubectl"),
    (r"\bkube c t l\b", "kubectl"),
    (r"\b(?:cooper|kube|cube) net(?:t)?(?:ies|es|ease)\b", "Kubernetes"),
    (r"\bengine[\s-]?x\b", "nginx"),
    (r"\bn gin x\b", "nginx"),
    (r"\bpie tests?\b", "pytest"),
    (r"\bpie torch\b", "PyTorch"),
    (r"\bnum pie\b", "NumPy"),
    (r"\brejex\b", "regex"),
    (r"\breg ex\b", "regex"),
    (r"\bjay son\b", "JSON"),
    (r"\byam+els?\b", "YAML"),
    (r"\blocal host\b", "localhost"),
    (r"\boh auth\b", "OAuth"),
    (r"\bo auth\b", "OAuth"),
    (r"\bjava script\b", "JavaScript"),
    (r"\btype script\b", "TypeScript"),
    (r"\bpost ?gres(?:ql)?\b", "PostgreSQL"),
    (r"\bsequel lite\b", "SQLite"),
    (r"\bs q lite\b", "SQLite"),
    (r"\bsequel\b", "SQL"),
    (r"\bgraph q l\b", "GraphQL"),
    (r"\belastic search\b", "Elasticsearch"),
    (r"\bterra form\b", "Terraform"),
    (r"\bdocker file\b", "Dockerfile"),
    (r"\bgit ignore\b", "gitignore"),
    (r"\bneo vim\b", "Neovim"),
    (r"\bv s code\b", "VS Code"),
    (r"\bsue do\b", "sudo"),
    (r"\b(?:see|c) mod\b", "chmod"),
    (r"\bch own\b", "chown"),
    (r"\bweb pack\b", "webpack"),
    (r"\bnext j s\b", "Next.js"),
    (r"\bnode j s\b", "Node.js"),
    (r"\bn p m\b", "npm"),
    (r"\bp n p m\b", "pnpm"),
    (r"\bfast a p i\b", "FastAPI"),
    (r"\brabbit m q\b", "RabbitMQ"),
    (r"\bmongo d b\b", "MongoDB"),
    (r"\bbase 64\b", "base64"),
    (r"\bsha 256\b", "SHA-256"),
    (r"\bu t f 8\b", "UTF-8"),
    (r"\butf 8\b", "UTF-8"),
    (r"\bc i c d\b", "CI/CD"),
    (r"\bci cd\b", "CI/CD"),
    (r"\bopen a i\b", "OpenAI"),
    (r"\bopen ai\b", "OpenAI"),
    (r"\bhugging face\b", "Hugging Face"),
    (r"\bweb hook\b", "webhook"),
    (r"\bseg fault\b", "segfault"),
    (r"\bgit re base\b", "git rebase"),
    (r"\bconfig map\b", "ConfigMap"),
    (r"\bstateful set\b", "StatefulSet"),
    (r"\bdaemon set\b", "DaemonSet"),
    (r"\bcron job\b", "CronJob"),
    (r"\bopen vino\b", "OpenVINO"),
    (r"\bpipe wire\b", "PipeWire"),
    (r"\bflat pack\b", "Flatpak"),
    (r"\bpod man\b", "Podman"),
    (r"\bmini cube\b", "Minikube"),
    (r"\bdot env\b", ".env"),
    (r"\bdot git\b", ".git"),
    (r"\bcherry pick\b", "cherry-pick"),
)

_lock = threading.Lock()
_state = {"key": None, "data": None, "rules": None, "errors": ()}
_logged_errors = set()
_rejected_bias = set()


def default_path():
    override = os.environ.get("WHISPER_VOCAB_PATH")
    if override:
        return override
    return os.path.join(os.path.expanduser("~/.config/whisper-npu"), "vocabulary.json")


def bundled_path():
    """Shipped default vocabulary. make install copies this for a new user."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "share", "vocabulary.json")


def default_config():
    return {
        "enabled": True,
        "preamble": DEFAULT_PREAMBLE,
        "terms": list(DEFAULT_TERMS),
        "replacements": [
            {"pattern": pattern, "replacement": repl, "ignore_case": True}
            for pattern, repl in _DEFAULT_RULE_PAIRS
        ],
    }


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def normalize(data):
    """Return a config dict with the keys the server and editor expect."""
    src = data or {}
    rules = []
    for rule in src.get("replacements") or []:
        if not isinstance(rule, dict):
            continue
        pattern = str(rule.get("pattern", "")).strip()
        replacement = str(rule.get("replacement", ""))
        if not pattern:
            continue
        rules.append({
            "pattern": pattern,
            "replacement": replacement,
            "ignore_case": rule.get("ignore_case") is not False,
        })
    preamble = str(src.get("preamble") or DEFAULT_PREAMBLE).strip() or DEFAULT_PREAMBLE
    return {
        "enabled": bool(src.get("enabled", True)),
        "preamble": preamble,
        "terms": _dedupe(src.get("terms") or []),
        "replacements": rules,
    }


def _compile_rules(rules):
    compiled = []
    errors = []
    for index, rule in enumerate(rules, 1):
        pattern = rule.get("pattern", "")
        if len(pattern) > MAX_PATTERN_LEN:
            errors.append(f"rule {index}: pattern longer than {MAX_PATTERN_LEN} characters")
            continue
        flags = 0 if rule.get("ignore_case") is False else re.IGNORECASE
        try:
            compiled.append((re.compile(pattern, flags), rule.get("replacement", "")))
        except re.error as exc:
            errors.append(f"rule {index}: {exc}")
    return compiled, tuple(errors)


def validate_config(data):
    """Return human-readable errors. Empty list means the config can be saved."""
    _compiled, errors = _compile_rules(normalize(data)["replacements"])
    return list(errors)


def parse_terms(text):
    return _dedupe(
        line.strip()
        for line in str(text).splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def rules_to_text(rules):
    lines = []
    for rule in rules or []:
        line = f"{rule.get('pattern', '')} => {rule.get('replacement', '')}"
        if rule.get("ignore_case") is False:
            line += "  [cs]"
        lines.append(line)
    return "\n".join(lines)


def parse_rules(text):
    """Parse editor lines. Returns (rules, errors)."""
    rules = []
    errors = []
    for lineno, line in enumerate(str(text).splitlines(), 1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if " => " not in raw:
            errors.append(f"line {lineno}: expected pattern => replacement")
            continue
        pattern, replacement = raw.split(" => ", 1)
        pattern = pattern.strip()
        replacement = replacement.strip()
        ignore_case = True
        if replacement.endswith("[cs]"):
            replacement = replacement[:-4].rstrip()
            ignore_case = False
        if not pattern:
            errors.append(f"line {lineno}: empty pattern")
            continue
        rules.append({
            "pattern": pattern,
            "replacement": replacement,
            "ignore_case": ignore_case,
        })
    return rules, errors


def _file_key(path):
    try:
        return (path, os.stat(path).st_mtime_ns)
    except OSError:
        return (path, None)


def _read_file(path):
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError:
        return default_config(), ()
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return default_config(), (f"{path}: {exc}. Using built-in vocabulary.",)
    if not isinstance(loaded, dict):
        return default_config(), (f"{path}: expected a JSON object. Using built-in vocabulary.",)
    return normalize(loaded), ()


def _cached(path):
    key = _file_key(path)
    if _state["key"] == key and _state["data"] is not None:
        return _state["data"], _state["rules"], _state["errors"]
    if key[1] is None:
        data, errors = default_config(), ()
    else:
        data, errors = _read_file(path)
    rules, rule_errors = _compile_rules(data["replacements"])
    errors = tuple(errors) + tuple(rule_errors)
    _state["key"] = key
    _state["data"] = data
    _state["rules"] = rules
    _state["errors"] = errors
    return data, rules, errors


def _log_errors(path, errors):
    token = (path, errors)
    if not errors or token in _logged_errors:
        return
    _logged_errors.add(token)
    for message in errors:
        log.warning("Vocabulary: %s", message)


def load(path=None):
    """Return a copy of the active config. Missing file means built-in defaults."""
    path = path or default_path()
    with _lock:
        data, _rules, errors = _cached(path)
        _log_errors(path, errors)
        return copy.deepcopy(data)


def save(data, path=None):
    """Atomically write vocabulary.json. Raises ValueError if a regex is invalid."""
    path = path or default_path()
    config = normalize(data)
    errors = validate_config(config)
    if errors:
        raise ValueError(errors[0])
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = json.dumps(config, indent=2) + "\n"
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    with _lock:
        if _state["key"] and _state["key"][0] == path:
            _state["key"] = None


def install_default(path=None):
    """Write the shipped vocabulary if the destination is missing.

    An existing file is left unchanged so a later install does not wipe edits.
    Returns True when a file was written.
    """
    path = path or default_path()
    if os.path.isfile(path):
        return False
    source = bundled_path()
    loaded = None
    try:
        with open(source, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        log.warning("Bundled vocabulary %s unusable (%s). Using built-in lists.", source, exc)
    if not isinstance(loaded, dict):
        loaded = default_config()
    save(loaded, path)
    return True


def _join_capped(items, budget):
    if budget <= 0:
        return ""
    parts = []
    used = 0
    for item in items:
        extra = len(item) if not parts else len(item) + 2
        if used + extra > budget:
            break
        parts.append(item)
        used += extra
    return ", ".join(parts)


def bias(path=None):
    """Return initial_prompt and hotwords. Empty strings when vocabulary is off."""
    data = load(path)
    if not data.get("enabled", True):
        return {"initial_prompt": "", "hotwords": ""}
    terms = data.get("terms") or []
    preamble = data.get("preamble") or DEFAULT_PREAMBLE
    room = PROMPT_CHAR_BUDGET - len(preamble) - 1
    body = _join_capped(terms, room)
    prompt = f"{preamble} {body}".strip() if body else preamble
    if len(prompt) > PROMPT_CHAR_BUDGET:
        prompt = prompt[:PROMPT_CHAR_BUDGET].rsplit(" ", 1)[0].rstrip(",")
    return {
        "initial_prompt": prompt,
        "hotwords": _join_capped(terms, HOTWORD_CHAR_BUDGET),
    }


def _match_case(sample, replacement):
    if not sample or not replacement:
        return replacement
    if sample.isupper():
        return replacement.upper()
    if sample[0].isupper() and replacement[0].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply(text, path=None):
    """Run replacement rules. Disabled or empty input is returned unchanged."""
    if not isinstance(text, str) or not text:
        return text
    path = path or default_path()
    with _lock:
        data, rules, errors = _cached(path)
        _log_errors(path, errors)
        if not data.get("enabled", True):
            return text
        rules = list(rules)
    for cre, replacement in rules:
        try:
            text = cre.sub(lambda match, repl=replacement: _match_case(match.group(0), repl), text)
        except re.error as exc:
            log.warning("Vocabulary rule %s failed: %s", cre.pattern, exc)
    return text


_BIAS_KEYS = ("hotwords", "initial_prompt")


def configure_backend(device):
    """Skip decoder bias on NPU.

    GenAI 2026.4 runs NPU Whisper on the stateful pipeline, so it no longer
    rejects initial_prompt up front. The compiled prompt slot is still 4
    tokens. MAX_PROMPT_LEN would raise it, but that key is forwarded to the
    encoder compile and the NPU plugin rejects the option. A vocabulary does
    not fit in the 4 tokens, and a longer prefill fails during inference.
    Regex replacements still run.
    """
    if not str(device or "").upper().startswith("NPU"):
        return
    fresh = [key for key in _BIAS_KEYS if key not in _rejected_bias]
    _rejected_bias.update(_BIAS_KEYS)
    if fresh:
        log.info(
            "NPU Whisper prompt slot is 4 tokens. Skipping %s. Vocabulary replacements still apply.",
            " and ".join(fresh),
        )


def _remember_rejected(key):
    if key not in _rejected_bias:
        _rejected_bias.add(key)
        log.warning("Whisper backend does not accept %s. Continuing without it.", key)


def _rejected_keys_in_error(exc, attempt):
    """Return bias kwargs this exception says the backend refused."""
    message = str(exc)
    named = [key for key in _BIAS_KEYS if key in attempt and key in message]
    lower = message.lower()
    if named and (isinstance(exc, TypeError) or "not supported" in lower or "unexpected" in lower):
        return named
    if isinstance(exc, TypeError) and "unexpected keyword" in lower:
        present = [key for key in _BIAS_KEYS if key in attempt]
        return present[:1]
    # Prefill longer than the compiled NPU prompt slot.
    if "roi_end" in lower:
        return [key for key in _BIAS_KEYS if key in attempt]
    return []


def generate_biased(fn, *args, **kwargs):
    """Call a Whisper generate/transcribe function with prompt and hotwords.

    Backends that reject one of those arguments are retried without it.
    OpenVINO GenAI 2025.4's static NPU pipeline raises a runtime error, not
    TypeError, for both arguments. GenAI 2026.4 accepts the arguments, then
    fails inference when the prefill is longer than the 4 token NPU slot.
    Any other failure is raised as-is.
    """
    prompt = bias()
    attempt = dict(kwargs)
    for key in _BIAS_KEYS:
        if key in _rejected_bias:
            continue
        value = prompt.get(key)
        if value:
            attempt[key] = value

    while True:
        try:
            return fn(*args, **attempt)
        except Exception as exc:
            dropped = _rejected_keys_in_error(exc, attempt)
            if not dropped:
                raise
            for key in dropped:
                attempt.pop(key, None)
                _remember_rejected(key)


def describe(path=None):
    path = path or default_path()
    data = load(path)
    prompts = bias(path)
    return {
        "path": path,
        "enabled": bool(data.get("enabled", True)),
        "terms": len(data.get("terms") or []),
        "replacements": len(data.get("replacements") or []),
        "initial_prompt_chars": len(prompts["initial_prompt"]),
        "hotwords_chars": len(prompts["hotwords"]),
        "using_defaults": not os.path.isfile(path),
    }
