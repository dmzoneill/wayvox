"""Tests for the About dialog system report."""

import os

import diagnostics


SAMPLE_LSPCI = """
00:02.0 "VGA compatible controller [0300]" "Intel Corporation [8086]" "Meteor Lake-P [Intel Arc Graphics] [7d55]" -r08 "Lenovo [17aa]" "Device [2317]"
00:0b.0 "Processing accelerators [1200]" "Intel Corporation [8086]" "Meteor Lake NPU [7d1d]" -r01 "Lenovo [17aa]" "Device [2317]"
01:00.0 "3D controller [0302]" "NVIDIA Corporation [10de]" "AD107M [GeForce RTX 4060 Max-Q / Mobile] [28a0]" -ra1 "Lenovo [17aa]" "Device [2317]"
02:00.0 "Ethernet controller [0200]" "Intel Corporation [8086]" "Device [5502]"
"""


def test_parse_lspci_keeps_gpu_and_npu():
    rows = diagnostics.parse_lspci_mm(SAMPLE_LSPCI)
    kinds = [(row["kind"], row["slot"]) for row in rows]
    assert kinds == [("GPU", "00:02.0"), ("NPU", "00:0b.0"), ("GPU", "01:00.0")]
    assert "Arc Graphics" in rows[0]["device"]
    assert "NPU" in rows[1]["device"]
    assert "RTX 4060" in rows[2]["device"]


def test_pci_sysfs_class_codes(tmp_path):
    npu = tmp_path / "0000:00:0b.0"
    gpu = tmp_path / "0000:00:02.0"
    net = tmp_path / "0000:02:00.0"
    for path, code, vendor, device in (
        (npu, "0x120000", "0x8086", "0x7d1d"),
        (gpu, "0x030000", "0x8086", "0x7d55"),
        (net, "0x020000", "0x8086", "0x5502"),
    ):
        path.mkdir()
        (path / "class").write_text(code)
        (path / "vendor").write_text(vendor)
        (path / "device").write_text(device)
    rows = diagnostics.pci_from_sysfs(str(tmp_path))
    assert [(row["kind"], row["slot"]) for row in rows] == [
        ("GPU", "0000:00:02.0"),
        ("NPU", "0000:00:0b.0"),
    ]


def test_format_report_names_devices_library_and_failover():
    report = {
        "app": {"name": "Voxforge", "version": "0.0.2"},
        "source": "whisper-server",
        "python": "3.14.0",
        "python_executable": "/venv/bin/python",
        "platform": "Linux",
        "cpu": "Intel Core Ultra 7",
        "libraries": {
            "openvino": "2026.4.1",
            "openvino-genai": "2026.4.1",
            "openvino-tokenizers": None,
        },
        "openvino": {
            "importable": True,
            "version": "2026.4.1",
            "devices": [
                {
                    "id": "NPU",
                    "properties": {"FULL_DEVICE_NAME": "Intel AI Boost", "NPU_DRIVER_VERSION": "1.2"},
                    "plugins": {"NPU": {"description": "NPU plugin", "build": "2026.4.1"}},
                },
                {
                    "id": "GPU",
                    "properties": {"FULL_DEVICE_NAME": "Intel Arc Graphics"},
                    "plugins": {},
                },
            ],
        },
        "pci": [
            {"kind": "GPU", "slot": "00:02.0", "vendor": "Intel", "device": "Arc Graphics"},
            {"kind": "NPU", "slot": "00:0b.0", "vendor": "Intel", "device": "Meteor Lake NPU"},
            {"kind": "GPU", "slot": "01:00.0", "vendor": "NVIDIA", "device": "RTX 4060"},
        ],
        "modules": {
            "intel_vpu": {"loaded": True, "version": "1.2"},
            "nvidia": {"loaded": True, "version": None},
        },
        "nvidia": [{"name": "RTX 4060", "driver": "580.1", "pci": "01:00.0"}],
        "runtime": {
            "engine": "openvino",
            "device": "NPU",
            "requested_device": "auto",
            "model": "whisper-small",
            "interface": "OpenVINO GenAI WhisperPipeline, device plugin NPU",
            "bias": {
                "mode": "regex",
                "active": [],
                "rejected": ["hotwords", "initial_prompt"],
                "pipeline": "stateful",
                "device": "NPU",
                "failover": ["prompt_slot"],
            },
            "vocab": {"terms": 105, "replacements": 74, "path": "/tmp/vocabulary.json"},
        },
    }
    text = diagnostics.format_report(report)
    assert "Voxforge 0.0.2" in text
    assert "OpenVINO GenAI WhisperPipeline, device plugin NPU" in text
    assert "Intel AI Boost" in text
    assert "Intel Arc Graphics" in text
    assert "Meteor Lake NPU" in text
    assert "RTX 4060" in text
    assert "openvino-genai: 2026.4.1" in text
    assert "openvino-tokenizers: not installed" in text
    assert "NPU_DRIVER_VERSION: 1.2" in text
    assert "4 tokens" in text
    assert "MAX_PROMPT_LEN" in text
    assert "restart whisper-server" in text
    assert "/venv/bin/python" in text


def test_bias_action_names_upgrade_for_unsupported_npu():
    text = diagnostics.bias_action({
        "mode": "regex",
        "device": "NPU",
        "failover": ["unsupported"],
        "rejected": ["initial_prompt", "hotwords"],
    })
    assert "2026.4.1" in text
    assert "restart whisper-server" in text


def test_missing_npu_plugin_hint():
    hints = diagnostics.health_hints({
        "openvino": {"importable": True, "devices": [{"id": "CPU"}]},
        "pci": [{"kind": "NPU", "slot": "00:0b.0"}],
        "modules": {"intel_vpu": {"loaded": False}},
        "libraries": {"openvino": "2026.4.1", "openvino-genai": "2026.4.1"},
        "runtime": {"engine": "openvino"},
    })
    assert any("intel_vpu" in hint for hint in hints)
    assert any("did not list an NPU" in hint for hint in hints)


def test_collect_has_report_sections():
    report = diagnostics.collect()
    assert report["app"]["version"]
    assert "python" in report
    assert "openvino" in report
    assert isinstance(report["pci"], list)
    text = diagnostics.format_report(report)
    assert "OpenVINO devices" in text
    assert "PCI accelerators" in text
    assert os.path.basename(report["python_executable"])
