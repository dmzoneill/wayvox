"""Technical vocabulary for Whisper dictation.

Terms bias the decoder through initial_prompt and hotwords when the
backend accepts them. The first transcription tries both. GenAI 2025.4's
static NPU pipeline rejects the arguments. GenAI 2026.4 accepts them.

The normal NPU Whisper compile sets whisper_max_prompt_size to 4.
English-only prefill spends 3 of those tokens, so "pull request" does not
fit. word_timestamps=True on the WhisperPipeline constructor compiles a
447 token slot instead. make install persists that as
WHISPER_NPU_WORD_TIMESTAMPS=1 and stores the compiled model under
WHISPER_CACHE_DIR (default ~/.cache/wayvox). Set the flag to 0 to
keep the 4 token compile. Do not set MAX_PROMPT_LEN. The encoder rejects
that name. A rejection still drops the arguments for this process.
WHISPER_DEVICE=GPU also sends the full list. Replacement rules still fix
mishears the model emits, such as "polar quest" -> "pull request".

The editable file is ~/.config/wayvox/vocabulary.json. make install
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

from wayvox_paths import cache_dir, vocabulary_file

try:
    import openvino_genai as ovg
except ImportError:
    ovg = None

_OV_ERROR = None
if ovg is not None:
    try:
        from openvino.exceptions import OVError as _OV_ERROR
    except ImportError:
        _OV_ERROR = None

log = logging.getLogger("vocab")

# Whisper clips the initial prompt to about half the text context. A short
# sentence biases better than a huge list, which the model may echo.
PROMPT_CHAR_BUDGET = 320
HOTWORD_CHAR_BUDGET = 1600
MAX_PATTERN_LEN = 200

DEFAULT_PREAMBLE = "Technical computer dictation. Terms:"

# Order matters. The prompt keeps the front of this list. Hotwords keep more.
_TERM_TEXT = (
    "pull request|merge request|GitHub|GitLab|Bitbucket|Kubernetes|kubectl|Docker|"
    "Dockerfile|docker compose|nginx|PostgreSQL|JSON|YAML|TOML|regex|localhost|OAuth|"
    "pytest|TypeScript|JavaScript|Python|Rust|Go|Node.js|npm|pnpm|pip|API|REST|GraphQL|"
    "gRPC|SQL|SQLite|MySQL|Redis|MongoDB|CI/CD|OpenVINO|Whisper|NPU|CUDA|PyTorch|NumPy|"
    "Hugging Face|OpenAI|FastAPI|Django|Flask|React|Vue|webhook|middleware|gitignore|"
    "cherry-pick|rebase|commit|branch|sudo|chmod|chown|systemd|SSH|UTF-8|base64|SHA-256|"
    "stdin|stdout|stderr|Terraform|Ansible|Helm|Minikube|Podman|Flatpak|ConfigMap|"
    "namespace|ingress|Elasticsearch|Kafka|RabbitMQ|Prometheus|Grafana|Neovim|VS Code|"
    "Wayland|PipeWire|segfault|stack trace|breakpoint|linter|async|boolean|microservice|"
    "repository|kernel|filesystem|endpoint|CORS|JWT|WebSocket|protobuf|cron|cluster|"
    "container"
)
DEFAULT_TERMS = tuple(_TERM_TEXT.split("|"))

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


class VocabState:
    """Process-wide vocabulary cache and backend bias."""

    def __init__(self):
        """Create the lock, vocabulary cache, and backend bias fields."""
        self.lock = threading.Lock()
        self.cache = {"key": None, "data": None, "rules": None, "errors": ()}
        self.logged_errors = set()
        self.rejected_bias = set()
        self.failover_logged = set()
        self.failover_reasons = []
        self._scalars = {
            "backend_device": "",
            "pipeline_mode": "",
            "bias_confirmed": False,
            "compiled_word_timestamps": False,
        }

    @property
    def backend_device(self):
        """Whisper device recorded for later bias decisions."""
        return self._scalars["backend_device"]

    @backend_device.setter
    def backend_device(self, value):
        self._scalars["backend_device"] = value

    @property
    def pipeline_mode(self):
        """Loaded pipeline mode, or empty before a model is opened."""
        return self._scalars["pipeline_mode"]

    @pipeline_mode.setter
    def pipeline_mode(self, value):
        self._scalars["pipeline_mode"] = value

    @property
    def bias_confirmed(self):
        """True after a transcription accepts decoder bias."""
        return self._scalars["bias_confirmed"]

    @bias_confirmed.setter
    def bias_confirmed(self, value):
        self._scalars["bias_confirmed"] = value

    @property
    def compiled_word_timestamps(self):
        """True when the pipeline compiled word timestamps."""
        return self._scalars["compiled_word_timestamps"]

    @compiled_word_timestamps.setter
    def compiled_word_timestamps(self, value):
        self._scalars["compiled_word_timestamps"] = value


STATE = VocabState()


def _call_errors():
    errors = (TypeError, ValueError, RuntimeError, OSError)
    if _OV_ERROR is None:
        return errors
    return errors + (_OV_ERROR,)


_FALSE_ENV = frozenset({"0", "false", "no", "off"})


def npu_prompt_slot_enabled():
    """True when NPU should compile with word_timestamps (447 token slot).

    The installer writes WHISPER_NPU_WORD_TIMESTAMPS=1. Unset means the
    same default, so a hand-started server matches an installed one.
    """
    raw = os.environ.get("WHISPER_NPU_WORD_TIMESTAMPS")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _FALSE_ENV


def whisper_cache_dir():
    """Directory for the compiled NPU model. Empty env disables the cache."""
    raw = cache_dir()
    return raw or None


def default_path():
    """Return the vocabulary path, honoring WHISPER_VOCAB_PATH."""
    override = os.environ.get("WHISPER_VOCAB_PATH")
    if override:
        return override
    return str(vocabulary_file())


def bundled_path():
    """Shipped default vocabulary. make install copies this for a new user."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "share",
        "vocabulary.json",
    )


def _default_replacements():
    """Build the shipped replacement rules from the default pattern pairs."""
    rules = []
    for pattern, repl in _DEFAULT_RULE_PAIRS:
        rules.append(
            {
                "pattern": pattern,
                "replacement": repl,
                "ignore_case": True,
            },
        )
    return rules


def default_config():
    """Return the built-in vocabulary used when no file is available."""
    return {
        "enabled": True,
        "preamble": DEFAULT_PREAMBLE,
        "terms": list(DEFAULT_TERMS),
        "replacements": _default_replacements(),
    }


def _dedupe(items):
    """Handle dedupe."""
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
        rules.append(
            {
                "pattern": pattern,
                "replacement": replacement,
                "ignore_case": rule.get("ignore_case") is not False,
            },
        )
    preamble = str(src.get("preamble") or DEFAULT_PREAMBLE).strip() or DEFAULT_PREAMBLE
    return {
        "enabled": bool(src.get("enabled", True)),
        "preamble": preamble,
        "terms": _dedupe(src.get("terms") or []),
        "replacements": rules,
    }


def _compile_rules(rules):
    """Handle compile rules."""
    compiled = []
    errors = []
    for index, rule in enumerate(rules, 1):
        pattern = rule.get("pattern", "")
        if len(pattern) > MAX_PATTERN_LEN:
            errors.append(
                f"rule {index}: pattern longer than {MAX_PATTERN_LEN} characters",
            )
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
    """Split editor text into unique terms, skipping blanks and comments."""
    terms = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            terms.append(stripped)
    return _dedupe(terms)


def rules_to_text(rules):
    """Render replacement rules as lines the editor can display."""
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
        rules.append(
            {
                "pattern": pattern,
                "replacement": replacement,
                "ignore_case": ignore_case,
            },
        )
    return rules, errors


def _file_key(path):
    """Handle file key."""
    try:
        return (path, os.stat(path).st_mtime_ns)
    except OSError:
        return (path, None)


def _read_file(path):
    """Handle read file."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError:
        return default_config(), ()
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        return default_config(), (f"{path}: {exc}. Using built-in vocabulary.",)
    if not isinstance(loaded, dict):
        message = f"{path}: expected a JSON object. Using built-in vocabulary."
        return default_config(), (message,)
    return normalize(loaded), ()


def _cached(path):
    """Handle cached."""
    key = _file_key(path)
    if STATE.cache["key"] == key and STATE.cache["data"] is not None:
        return STATE.cache["data"], STATE.cache["rules"], STATE.cache["errors"]
    if key[1] is None:
        data, errors = default_config(), ()
    else:
        data, errors = _read_file(path)
    rules, rule_errors = _compile_rules(data["replacements"])
    errors = tuple(errors) + tuple(rule_errors)
    STATE.cache["key"] = key
    STATE.cache["data"] = data
    STATE.cache["rules"] = rules
    STATE.cache["errors"] = errors
    return data, rules, errors


def _log_errors(path, errors):
    """Handle log errors."""
    token = (path, errors)
    if not errors or token in STATE.logged_errors:
        return
    STATE.logged_errors.add(token)
    for message in errors:
        log.warning("Vocabulary: %s", message)


def load(path=None):
    """Return a copy of the active config. Missing file means built-in defaults."""
    path = path or default_path()
    with STATE.lock:
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
    with STATE.lock:
        cached_key = STATE.cache["key"]
        path_key = tuple(cached_key) if isinstance(cached_key, tuple) else ()
        if path_key and path_key[0] == path:
            STATE.cache["key"] = None


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
        log.warning(
            "Bundled vocabulary %s unusable (%s). Using built-in lists.",
            source,
            exc,
        )
    if not isinstance(loaded, dict):
        loaded = default_config()
    save(loaded, path)
    return True


def _join_capped(items, budget):
    """Handle join capped."""
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
    """Handle match case."""
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
    with STATE.lock:
        data, rules, errors = _cached(path)
        _log_errors(path, errors)
        if not data.get("enabled", True):
            return text
        rules = list(rules)
    for cre, replacement in rules:
        try:
            text = cre.sub(
                lambda match, repl=replacement: _match_case(match.group(0), repl),
                text,
            )
        except re.error as exc:
            log.warning("Vocabulary rule %s failed: %s", cre.pattern, exc)
    return text


_BIAS_KEYS = ("hotwords", "initial_prompt")


def configure_backend(device):
    """Remember the Whisper device. Bias stays enabled until a generate fails.

    Skipping NPU bias up front hid builds that can accept initial_prompt.
    The first transcription tries it. A rejection fails over to regex and
    logs the upgrade that enables decoder bias.
    """
    setattr(STATE, "backend_device", str(device or ""))
    if not STATE.backend_device.upper().startswith("NPU"):
        return
    wide = npu_prompt_slot_enabled()
    log.info(
        "NPU Whisper will try initial_prompt and hotwords on the first transcription. "
        "WHISPER_NPU_WORD_TIMESTAMPS=%s. "
        "1 compiles word_timestamps=True into WhisperPipeline so the prompt slot is 447. "
        "0 keeps the 4 token slot and fails a term list over to regex. "
        "Compiled models stay in %s. Vocabulary file: %s",
        "1" if wide else "0",
        whisper_cache_dir() or "(disabled)",
        default_path(),
    )


def _installed_genai_version():
    if ovg is None:
        return "not installed"
    return getattr(ovg, "__version__", "unknown")


def _option_rejected(exc, name):
    """True when an exception is about one named argument or pipeline option."""
    text = str(exc)
    if name not in text:
        return False
    lower = text.lower()
    tokens = ("unexpected", "not supported", "unsupported", "unknown")
    matched = any(token in lower for token in tokens)
    return isinstance(exc, TypeError) or matched


_LOAD_OPTIONS = ("word_timestamps", "STATIC_PIPELINE", "CACHE_DIR")


def _pipeline_kwargs(device):
    """Constructor options. NPU adds the wide prompt slot and a persistent cache."""
    npu = str(device or "").upper().startswith("NPU")
    kwargs = {"device": device}
    cache = whisper_cache_dir()
    if cache:
        try:
            os.makedirs(cache, exist_ok=True)
        except OSError as exc:
            log.warning(
                "Failover: could not create Whisper cache %s (%s). "
                "Compiling without CACHE_DIR. The next start will compile again. "
                "Action: make the directory writable, or set WHISPER_CACHE_DIR to one that is, "
                "then restart wayvox-server.",
                cache,
                exc,
            )
        else:
            kwargs["CACHE_DIR"] = cache
    if npu:
        kwargs["STATIC_PIPELINE"] = False
        if npu_prompt_slot_enabled():
            kwargs["word_timestamps"] = True
    return kwargs


def _rejected_load_option(exc, pending):
    """Handle rejected load option."""
    for name in _LOAD_OPTIONS:
        if name in pending and _option_rejected(exc, name):
            return name
    return None


def load_whisper_pipeline(factory, model_path, device):
    """Open WhisperPipeline.

    On NPU, ask for the stateful pipeline and, unless
    WHISPER_NPU_WORD_TIMESTAMPS=0, compile with word_timestamps=True.
    That constructor flag is what raises the NPU prompt slot from 4 to 447.
    CACHE_DIR keeps the compiled blob so the next start does not rebuild it.
    A binding that rejects one of those options is retried without it.
    Other load errors propagate.
    """
    npu = str(device or "").upper().startswith("NPU")
    kwargs = _pipeline_kwargs(device)
    while True:
        try:
            pipe = factory(str(model_path), **kwargs)
        except _call_errors() as exc:
            dropped = _rejected_load_option(exc, kwargs)
            if not dropped:
                raise
            kwargs.pop(dropped, None)
            if dropped == "STATIC_PIPELINE":
                _log_pipeline_failover(exc)
            elif dropped == "word_timestamps":
                _log_word_timestamps_failover(exc)
            else:
                _log_cache_failover(exc)
            continue
        break
    enabled = bool(kwargs.get("word_timestamps"))
    setattr(STATE, "compiled_word_timestamps", enabled)
    mode = "stateful" if npu and "STATIC_PIPELINE" in kwargs else "default"
    setattr(STATE, "pipeline_mode", mode)
    if npu:
        if STATE.compiled_word_timestamps:
            log.info(
                "NPU Whisper pipeline is stateful and compiled with word_timestamps=True. "
                "Prompt slot is 447 tokens, so initial_prompt and hotwords are sent. "
                "OpenVINO GenAI %s. Compiled model cache: %s. "
                "The first start after enabling this compiles for a few minutes. "
                "Later starts reuse the cache. "
                "Set WHISPER_NPU_WORD_TIMESTAMPS=0 and restart wayvox-server "
                "to go back to the 4 token slot.",
                _installed_genai_version(),
                kwargs.get("CACHE_DIR") or "(disabled)",
            )
        else:
            log.info(
                "NPU Whisper pipeline is %s. "
                "word_timestamps is off, so the prompt slot stays 4 tokens. "
                "A term list fails over to regex. OpenVINO GenAI %s. "
                "Action: set WHISPER_NPU_WORD_TIMESTAMPS=1 and restart wayvox-server. "
                "make install writes that default.",
                STATE.pipeline_mode,
                _installed_genai_version(),
            )
    return pipe


def _one_line(exc):
    return " ".join(str(exc).split())


def _sentence_block(parts):
    return " ".join(parts) + " "


def _log_pipeline_failover(exc):
    """Handle log pipeline failover."""
    if "static_pipeline" not in STATE.failover_reasons:
        STATE.failover_reasons.append("static_pipeline")
    signature = ("pipeline", ("STATIC_PIPELINE",))
    if signature in STATE.failover_logged:
        return
    STATE.failover_logged.add(signature)
    log.warning(
        "Failover: WhisperPipeline rejected STATIC_PIPELINE (%s). "
        "Loaded the default pipeline instead. "
        "GenAI 2025.4 defaults NPU to a static pipeline that rejects initial_prompt and hotwords. "
        "Action: upgrade openvino, openvino-genai, and openvino-tokenizers "
        "to 2026.4.1 or newer, then restart wayvox-server. "
        "That release uses the stateful pipeline, which accepts decoder bias. "
        "Regex replacements in %s still apply. Installed openvino-genai is %s.",
        _one_line(exc),
        default_path(),
        _installed_genai_version(),
    )


def _log_named_load_failover(reason, signature, message):
    """Handle log named load failover."""
    if reason not in STATE.failover_reasons:
        STATE.failover_reasons.append(reason)
    if signature in STATE.failover_logged:
        return
    STATE.failover_logged.add(signature)
    log.warning(message)


def _log_word_timestamps_failover(exc):
    detail = _one_line(exc)
    path = default_path()
    version = _installed_genai_version()
    message = (
        f"Failover: WhisperPipeline rejected word_timestamps ({detail}). "
        "The NPU prompt slot stays 4 tokens, so a term list will not fit. "
        "Action: upgrade openvino, openvino-genai, and openvino-tokenizers "
        "to 2026.4.1 or newer, then restart wayvox-server. "
        "Set WHISPER_NPU_WORD_TIMESTAMPS=0 to skip this compile option. "
        f"Regex replacements in {path} still apply. "
        f"Installed openvino-genai is {version}."
    )
    _log_named_load_failover(
        "word_timestamps",
        ("pipeline", ("word_timestamps",)),
        message,
    )


def _log_cache_failover(exc):
    detail = _one_line(exc)
    version = _installed_genai_version()
    message = (
        f"Failover: WhisperPipeline rejected CACHE_DIR ({detail}). "
        "Loaded without a compiled-model cache, so the next start compiles again. "
        "Action: unset WHISPER_CACHE_DIR or point it at a writable directory, "
        f"then restart wayvox-server. Installed openvino-genai is {version}."
    )
    _log_named_load_failover(
        "cache_dir",
        ("pipeline", ("CACHE_DIR",)),
        message,
    )


def _failure_reason(exc, attempt):
    """Classify a generate error as a bias rejection, or return (None, [])."""
    text = str(exc)
    lower = text.lower()
    present = [key for key in _BIAS_KEYS if key in attempt]
    if not present:
        return None, []
    named = [key for key in present if key in text]
    markers = ("not supported", "unexpected", "has_value")
    text_match = any(marker in lower for marker in markers)
    unsupported = isinstance(exc, TypeError) or text_match
    if named and unsupported:
        # The static NPU pipeline rejects every bias argument the same way.
        # Drop them together so the retry does not spend another inference.
        if "npu" in lower and "not supported" in lower:
            return "unsupported", list(present)
        return "unsupported", named
    if isinstance(exc, TypeError) and "unexpected keyword" in lower:
        return "unsupported", present[:1]
    # Prefill longer than the compiled NPU prompt slot.
    if "roi_end" in lower:
        return "prompt_slot", list(present)
    return None, []


def _log_failover(reason, dropped, exc):
    """Handle log failover."""
    if reason and reason not in STATE.failover_reasons:
        STATE.failover_reasons.append(reason)
    signature = (reason, tuple(dropped))
    if signature in STATE.failover_logged:
        return
    STATE.failover_logged.add(signature)
    keys = " and ".join(dropped)
    version = _installed_genai_version()
    path = default_path()
    npu = "npu" in str(exc).lower() or STATE.backend_device.upper().startswith("NPU")
    if reason == "prompt_slot":
        if STATE.compiled_word_timestamps:
            action = _sentence_block(
                [
                    "word_timestamps=True was compiled and the slot is still 4 tokens.",
                    "Action: do not set MAX_PROMPT_LEN. The NPU encoder rejects that name.",
                    "Set WHISPER_DEVICE=GPU and restart wayvox-server to send the full term list.",
                    "This OpenVINO build did not raise whisper_max_prompt_size.",
                ],
            )
        else:
            action = _sentence_block(
                [
                    "word_timestamps was not compiled, so the slot stayed 4 tokens.",
                    "Action: set WHISPER_NPU_WORD_TIMESTAMPS=1 and restart wayvox-server.",
                    "make install writes that into the user service.",
                    "The flag passes word_timestamps=True to the WhisperPipeline constructor",
                    "and compiles a 447 token slot.",
                    "Do not set MAX_PROMPT_LEN. The NPU encoder rejects that name.",
                ],
            )
        log.warning(
            "Failover: NPU Whisper prompt slot is 4 tokens, shorter than %s. "
            "Error: %s. Dropped %s for the rest of this process so the next "
            "utterance skips the failed inference. "
            "Regex replacements in %s still correct misheard terms. "
            "%s"
            "Installed openvino-genai is %s. Until then, add a replacement rule "
            "for any term the model still mishears.",
            keys,
            _one_line(exc),
            keys,
            path,
            action,
            version,
        )
        return
    if reason == "unsupported" and npu:
        log.warning(
            "Failover: this OpenVINO build rejected %s (%s). "
            "Transcription continues without decoder bias. "
            "Regex replacements in %s still correct misheard terms. "
            "Action: upgrade openvino, openvino-genai, and openvino-tokenizers "
            "to 2026.4.1 or newer, then restart wayvox-server. "
            "GenAI 2025.4 uses a static NPU pipeline that rejects initial_prompt and hotwords. "
            "GenAI 2026.4 accepts those arguments. "
            "Installed openvino-genai is %s. "
            "If a later log says the prompt slot is 4 tokens, a full term list "
            "still cannot be sent and regex remains the fix.",
            keys,
            _one_line(exc),
            path,
            version,
        )
        return
    log.warning(
        "Failover: Whisper backend rejected %s (%s). "
        "Transcription continues without those arguments. "
        "Regex replacements in %s still apply. "
        "Action: none if the other bias argument was accepted. "
        "CPU, GPU, and faster-whisper accept initial_prompt and hotwords. "
        "An older binding raises TypeError for an unknown argument.",
        keys,
        _one_line(exc),
        path,
    )


def generate_biased(fn, *args, **kwargs):
    """Call a Whisper generate/transcribe function with prompt and hotwords.

    Backends that reject one of those arguments are retried without it.
    The rejection is remembered so the next utterance does not pay a failed
    inference. OpenVINO GenAI 2025.4's static NPU pipeline raises a runtime
    error, not TypeError, for both arguments. GenAI 2026.4 accepts the
    arguments, then fails inference when the prefill is longer than the
    4 token NPU slot. Any other failure is raised as-is.
    """
    prompt = bias()
    attempt = dict(kwargs)
    for key in _BIAS_KEYS:
        if key in STATE.rejected_bias:
            continue
        value = prompt.get(key)
        if value:
            attempt[key] = value

    while True:
        try:
            result = fn(*args, **attempt)
        except _call_errors() as exc:
            reason, dropped = _failure_reason(exc, attempt)
            if not dropped:
                raise
            for key in dropped:
                attempt.pop(key, None)
                STATE.rejected_bias.add(key)
            _log_failover(reason, dropped, exc)
            continue
        if any(key in attempt for key in _BIAS_KEYS):
            setattr(STATE, "bias_confirmed", True)
        return result


def bias_status():
    """Runtime bias state for /health and startup logs."""
    rejected = sorted(STATE.rejected_bias)
    active = [key for key in _BIAS_KEYS if key not in STATE.rejected_bias]
    if rejected and not active:
        mode = "regex"
    elif rejected:
        mode = "partial"
    elif STATE.bias_confirmed or not STATE.backend_device.upper().startswith("NPU"):
        mode = "on"
    else:
        mode = "pending"
    npu = STATE.backend_device.upper().startswith("NPU")
    return {
        "mode": mode,
        "active": active,
        "rejected": rejected,
        "pipeline": STATE.pipeline_mode or "unloaded",
        "device": STATE.backend_device,
        "failover": list(STATE.failover_reasons),
        "word_timestamps": STATE.compiled_word_timestamps,
        "word_timestamps_requested": bool(npu and npu_prompt_slot_enabled()),
        "cache_dir": whisper_cache_dir() or "",
    }


def describe(path=None):
    """Summarize the active vocabulary file and decoder-bias state."""
    path = path or default_path()
    data = load(path)
    prompts = bias(path)
    status = bias_status()
    return {
        "path": path,
        "enabled": bool(data.get("enabled", True)),
        "terms": len(data.get("terms") or []),
        "replacements": len(data.get("replacements") or []),
        "initial_prompt_chars": len(prompts["initial_prompt"]),
        "hotwords_chars": len(prompts["hotwords"]),
        "using_defaults": not os.path.isfile(path),
        "bias_mode": status["mode"],
        "bias_rejected": status["rejected"],
    }
