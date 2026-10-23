"""Hardware, library, and inference facts for the About dialog.

The transcription server publishes this on GET /system so the tray shows the
process that actually runs Whisper. about.py falls back to a local collect
when that server is down.
"""

import os
import platform
import re
import subprocess
import sys

_PCI_KIND = {
    "03": "GPU",
    "12": "NPU",
}

_OV_PROPS = (
    "FULL_DEVICE_NAME",
    "DEVICE_ARCHITECTURE",
    "DEVICE_TYPE",
)

_NPU_PROPS = (
    "NPU_DRIVER_VERSION",
    "NPU_COMPILER_VERSION",
    "DRIVER_VERSION",
)

_LIBRARIES = (
    "openvino",
    "openvino-genai",
    "openvino-tokenizers",
    "faster-whisper",
    "ctranslate2",
    "flask",
)

_MODULES = ("intel_vpu", "xe", "i915", "nvidia")


def app_version():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return "unknown"
    for line in text.splitlines():
        if line.startswith("version="):
            return line.split("=", 1)[1].strip() or "unknown"
    return text.strip() or "unknown"


def _dist_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _run(cmd):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout or ""


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _cpu_model():
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or ""


def parse_lspci_mm(text):
    """Parse `lspci -mm -nn` and keep GPU and NPU rows."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        slot, _, rest = line.partition(" ")
        fields = re.findall(r'"([^"]*)"', rest)
        if len(fields) < 3:
            continue
        klass, vendor, device = fields[0], fields[1], fields[2]
        kind = _pci_kind_from_class(klass)
        if not kind:
            continue
        rows.append({
            "kind": kind,
            "slot": slot,
            "class": klass,
            "vendor": vendor,
            "device": device,
        })
    return rows


def _pci_kind_from_class(klass):
    text = klass.lower()
    if "vga" in text or "3d controller" in text or "display" in text:
        return "GPU"
    if "processing accelerat" in text:
        return "NPU"
    match = re.search(r"\[([0-9a-f]{2,4})", text)
    if not match:
        return None
    return _PCI_KIND.get(match.group(1)[:2])


def pci_from_sysfs(base="/sys/bus/pci/devices"):
    if not os.path.isdir(base):
        return []
    rows = []
    for slot in sorted(os.listdir(base)):
        code = _read(os.path.join(base, slot, "class")).lower()
        if not code.startswith("0x") or len(code) < 4:
            continue
        kind = _PCI_KIND.get(code[2:4])
        if not kind:
            continue
        vendor = _read(os.path.join(base, slot, "vendor"))
        device = _read(os.path.join(base, slot, "device"))
        rows.append({
            "kind": kind,
            "slot": slot,
            "class": code,
            "vendor": vendor,
            "device": device,
        })
    return rows


def _pci_devices():
    parsed = parse_lspci_mm(_run(["lspci", "-mm", "-nn"]))
    if parsed:
        return parsed
    return pci_from_sysfs()


def _kernel_module(name):
    base = os.path.join("/sys/module", name)
    if not os.path.isdir(base):
        return {"loaded": False, "version": None}
    version = _read(os.path.join(base, "version")) or None
    return {"loaded": True, "version": version}


def _nvidia():
    text = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,pci.bus_id",
        "--format=csv,noheader",
    ])
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        rows.append({
            "name": parts[0],
            "driver": parts[1] if len(parts) > 1 else "",
            "pci": parts[2] if len(parts) > 2 else "",
        })
    return rows


def _openvino_devices():
    version = _dist_version("openvino")
    try:
        import openvino as ov
    except Exception as exc:
        return {
            "importable": False,
            "version": version,
            "error": f"{exc.__class__.__name__}: {exc}",
            "devices": [],
        }
    version = getattr(ov, "__version__", None) or version
    try:
        core = ov.Core()
        names = list(core.available_devices)
    except Exception as exc:
        return {
            "importable": True,
            "version": version,
            "error": f"{exc.__class__.__name__}: {exc}",
            "devices": [],
        }
    devices = []
    for dev in names:
        entry = {"id": dev, "properties": {}, "plugins": {}}
        props = list(_OV_PROPS)
        if str(dev).upper().startswith("NPU"):
            props.extend(_NPU_PROPS)
        for prop in props:
            try:
                value = _jsonable(core.get_property(dev, prop))
            except Exception:
                continue
            if isinstance(value, str) and value.startswith("Type."):
                value = value.split(".", 1)[1].lower()
            entry["properties"][prop] = value
        try:
            versions = core.get_versions(dev)
        except Exception as exc:
            entry["plugins_error"] = f"{exc.__class__.__name__}: {exc}"
            versions = {}
        for name, info in (versions or {}).items():
            entry["plugins"][str(name)] = {
                "description": str(getattr(info, "description", "") or ""),
                "build": str(getattr(info, "build_number", "") or ""),
            }
        devices.append(entry)
    return {
        "importable": True,
        "version": version,
        "devices": devices,
    }


def collect():
    """Facts visible to this Python process. Does not load a Whisper model."""
    libraries = {name: _dist_version(name) for name in _LIBRARIES}
    return {
        "app": {"name": "Voxforge", "version": app_version()},
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "libraries": libraries,
        "openvino": _openvino_devices(),
        "pci": _pci_devices(),
        "modules": {name: _kernel_module(name) for name in _MODULES},
        "nvidia": _nvidia(),
    }


def inference_interface(runtime):
    if not runtime:
        return "unknown"
    if runtime.get("interface"):
        return runtime["interface"]
    engine = runtime.get("engine") or ""
    device = runtime.get("device") or "unset"
    if engine == "openvino":
        return f"OpenVINO GenAI WhisperPipeline, device plugin {device}"
    if engine == "faster-whisper":
        return f"faster-whisper through CTranslate2, device {device}"
    if engine == "whisper.cpp":
        return f"whisper.cpp libwhisper.so, OpenVINO encoder on {device}"
    return engine or "unknown"


def bias_action(bias):
    """What the user should do for the current decoder-bias state."""
    if not bias:
        return "No bias state. The server did not report it."
    mode = bias.get("mode") or "unknown"
    reasons = list(bias.get("failover") or [])
    device = str(bias.get("device") or "")
    notes = []
    if "static_pipeline" in reasons or "unsupported" in reasons:
        notes.append(
            "Failover: this OpenVINO build rejected decoder bias "
            "(initial_prompt and hotwords). "
            "Action: upgrade openvino, openvino-genai, and openvino-tokenizers "
            "to 2026.4.1 or newer, then restart whisper-server. "
            "GenAI 2025.4 uses a static NPU pipeline that rejects those arguments. "
            "GenAI 2026.4 accepts them. Regex replacements still correct misheard terms."
        )
    if "prompt_slot" in reasons:
        if bias.get("word_timestamps"):
            slot = (
                "word_timestamps=True was compiled and the slot is still 4 tokens. "
                "Action: do not set MAX_PROMPT_LEN. The NPU encoder rejects that name. "
                "Set WHISPER_DEVICE=GPU and restart whisper-server to send the full term list. "
            )
        else:
            slot = (
                "word_timestamps was not compiled, so the slot stayed 4 tokens. "
                "Action: set WHISPER_NPU_WORD_TIMESTAMPS=1 and restart whisper-server. "
                "make install writes that into the user service. "
                "The flag passes word_timestamps=True to the WhisperPipeline constructor and compiles a 447 token slot. "
                "Do not set MAX_PROMPT_LEN. The NPU encoder rejects that name. "
            )
        notes.append(
            "Failover: NPU Whisper prompt slot is 4 tokens, so a technical term list does not fit. "
            "initial_prompt and hotwords were dropped for this process. Regex replacements still run. "
            + slot +
            "Until then, add a replacement rule for any term the model still mishears."
        )
    if notes:
        return " ".join(notes)
    if mode == "pending" and device.upper().startswith("NPU"):
        return (
            "Decoder bias is armed and has not run yet. The next transcription tries "
            "initial_prompt and hotwords. If the NPU rejects them, this dialog and the journal "
            "(a line starting with Failover:) name the action. Reopen About after one dictation."
        )
    if mode == "on":
        return (
            "Decoder bias is active. The term list is sent as initial_prompt and hotwords. "
            "Regex replacements still run afterwards."
        )
    if mode == "partial":
        rejected = ", ".join(bias.get("rejected") or []) or "one argument"
        return (
            f"Partial failover: {rejected} was rejected. The other bias argument is still sent. "
            "Regex replacements still run."
        )
    if mode == "regex":
        return (
            "Only regex replacements are active. Transcribe once, then reopen About, so the server "
            "can record why bias was dropped. Also check the journal for a line starting with Failover:."
        )
    return f"Bias mode is {mode}."


def health_hints(report):
    """Extra checks that are not already covered by the bias action."""
    hints = []
    ov = report.get("openvino") or {}
    ids = [str(item.get("id") or "") for item in ov.get("devices") or []]
    pci = report.get("pci") or []
    modules = report.get("modules") or {}
    has_pci_npu = any(row.get("kind") == "NPU" for row in pci)
    has_ov_npu = any(device.upper().startswith("NPU") for device in ids)
    vpu = modules.get("intel_vpu") or {}
    if has_pci_npu and not vpu.get("loaded"):
        hints.append(
            "NPU PCI device is present and kernel module intel_vpu is not loaded. "
            "Action: load the NPU driver, then restart whisper-server."
        )
    if has_pci_npu and ov.get("importable") and not has_ov_npu:
        hints.append(
            "PCI shows an NPU but OpenVINO did not list an NPU device. "
            "Action: confirm this Python's OpenVINO build includes the NPU plugin, "
            "then restart whisper-server."
        )
    libs = report.get("libraries") or {}
    runtime = report.get("runtime") or {}
    if runtime.get("engine") == "openvino" and not libs.get("openvino-genai"):
        hints.append(
            "The server selected OpenVINO but openvino-genai is not installed in that process. "
            "Action: install openvino, openvino-genai, and openvino-tokenizers, then restart whisper-server."
        )
    if runtime.get("engine") == "openvino" and not libs.get("openvino"):
        hints.append(
            "The server selected OpenVINO but the openvino package version could not be read. "
            "Action: reinstall the OpenVINO wheel in the whisper-server environment and restart it."
        )
    return hints


def _prop_lines(entry):
    lines = []
    props = entry.get("properties") or {}
    name = props.get("FULL_DEVICE_NAME") or ""
    lines.append(f"  {entry.get('id')}: {name}".rstrip())
    for key, value in props.items():
        if key == "FULL_DEVICE_NAME" or value in (None, ""):
            continue
        lines.append(f"    {key}: {value}")
    for name, info in (entry.get("plugins") or {}).items():
        desc = (info or {}).get("description") or ""
        build = (info or {}).get("build") or ""
        detail = " ".join(part for part in (desc, f"build {build}" if build else "") if part)
        lines.append(f"    plugin {name}: {detail}".rstrip())
    if entry.get("plugins_error"):
        lines.append(f"    plugin error: {entry['plugins_error']}")
    return lines


def format_report(report):
    """Plain text shown in the About dialog."""
    app = report.get("app") or {}
    lines = [
        f"Voxforge {app.get('version', 'unknown')}",
        f"Source: {report.get('source', 'local')}",
    ]
    if report.get("warning"):
        lines.append(report["warning"])
    lines.append("")
    lines.append("Runtime")
    runtime = report.get("runtime") or {}
    if not runtime:
        lines.append("  Server did not answer. Inference engine, model, and bias state are unknown.")
        lines.append(
            "  Action: start whisper-server, confirm the tray host and port, then reopen About. "
            "Library versions below are this Python, not the service."
        )
    else:
        lines.append(f"  Engine: {runtime.get('engine') or 'unknown'}")
        lines.append(f"  Interface: {inference_interface(runtime)}")
        lines.append(f"  Device: {runtime.get('device') or 'unknown'}")
        if runtime.get("requested_device"):
            lines.append(f"  Requested device: {runtime['requested_device']}")
        lines.append(f"  Model: {runtime.get('model') or 'unknown'}")
        if runtime.get("llm_model"):
            lines.append(f"  LLM model: {runtime['llm_model']}")
        if runtime.get("llm_device"):
            lines.append(f"  LLM device: {runtime['llm_device']}")
        if runtime.get("whisper_cpp"):
            lines.append(f"  whisper.cpp: {runtime['whisper_cpp']}")
        if runtime.get("pid"):
            lines.append(f"  PID: {runtime['pid']}")
        bias = runtime.get("bias") or {}
        lines.append(f"  Bias mode: {bias.get('mode', 'unknown')}")
        active = ", ".join(bias.get("active") or []) or "none"
        rejected = ", ".join(bias.get("rejected") or []) or "none"
        lines.append(f"  Bias active: {active}")
        lines.append(f"  Bias rejected: {rejected}")
        lines.append(f"  Pipeline: {bias.get('pipeline', 'unknown')}")
        if str(bias.get("device") or "").upper().startswith("NPU") or bias.get("word_timestamps") is not None:
            if bias.get("word_timestamps"):
                compile_state = "word_timestamps compiled, 447 token slot"
            elif bias.get("word_timestamps_requested"):
                compile_state = "word_timestamps enabled, compiles on first model load"
            elif str(bias.get("device") or "").upper().startswith("NPU"):
                compile_state = "off, 4 token slot. Set WHISPER_NPU_WORD_TIMESTAMPS=1 and restart whisper-server"
            else:
                compile_state = ""
            if compile_state:
                lines.append(f"  NPU prompt compile: {compile_state}")
        if bias.get("cache_dir"):
            lines.append(f"  Whisper cache: {bias['cache_dir']}")
        vocab_info = runtime.get("vocab") or {}
        if vocab_info:
            lines.append(
                f"  Vocabulary: {vocab_info.get('terms', '?')} terms, "
                f"{vocab_info.get('replacements', '?')} rules"
            )
            if vocab_info.get("path"):
                lines.append(f"  Vocabulary file: {vocab_info['path']}")
        lines.append(f"  Action: {bias_action(bias)}")
    lines.append("")
    lines.append("OpenVINO devices")
    ov = report.get("openvino") or {}
    lines.append(f"  Importable: {bool(ov.get('importable'))}")
    lines.append(f"  Version: {ov.get('version') or 'not installed'}")
    if ov.get("error"):
        lines.append(f"  Error: {ov['error']}")
    devices = ov.get("devices") or []
    if not devices:
        lines.append("  Devices: none reported")
    for entry in devices:
        lines.extend(_prop_lines(entry))
    lines.append("")
    lines.append("PCI accelerators")
    pci = report.get("pci") or []
    if not pci:
        lines.append("  No GPU or NPU PCI device found")
    for row in pci:
        lines.append(
            f"  {row.get('kind')} {row.get('slot')}: {row.get('vendor')} {row.get('device')}".rstrip()
        )
    lines.append("")
    lines.append("Drivers")
    for name, info in (report.get("modules") or {}).items():
        if info.get("loaded"):
            lines.append(f"  {name}: {info.get('version') or 'loaded'}")
        else:
            lines.append(f"  {name}: not loaded")
    for gpu in report.get("nvidia") or []:
        lines.append(
            f"  nvidia-smi: {gpu.get('name')} driver {gpu.get('driver')} {gpu.get('pci')}".rstrip()
        )
    lines.append("")
    lines.append("Libraries")
    lines.append(f"  Python: {report.get('python') or 'unknown'}")
    lines.append(f"  Executable: {report.get('python_executable') or 'unknown'}")
    lines.append(f"  Platform: {report.get('platform') or 'unknown'}")
    if report.get("cpu"):
        lines.append(f"  CPU: {report['cpu']}")
    for name, version in (report.get("libraries") or {}).items():
        lines.append(f"  {name}: {version or 'not installed'}")
    hints = health_hints(report)
    if hints:
        lines.append("")
        lines.append("Checks")
        for hint in hints:
            lines.append(f"  {hint}")
    lines.append("")
    lines.append("Copy this text when reporting a device or dictation problem.")
    return "\n".join(lines)
