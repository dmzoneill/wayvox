.PHONY: install uninstall install-service uninstall-service enable-service disable-service \
       dev logs logs-service preview preview-viewer preview-dbus preview-stop deps clean

SHELL := /bin/bash
CWD := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PROJECT_DIR := $(realpath $(CWD)/..)
PYTHON ?= $(shell command -v python3 2>/dev/null || printf '%s' python3)
PLUGIN_ID = com.dmzoneill.wayvox
LEGACY_PLUGIN_ID = com.whisper.npu
INSTALL_DIR = ~/.local/share/plasma/plasmoids/$(PLUGIN_ID)
SERVICE_DIR = ~/.config/systemd/user
CONFIG_DIR = ~/.config/wayvox

# Fedora package names — adjust for other distros
KDE_RUNTIME_DEPS = plasma-workspace python3-dbus python3-pyqt6 dbus-tools
KDE_PREVIEW_DEPS = plasma-sdk
KDE_ALL_DEPS = $(KDE_RUNTIME_DEPS) $(KDE_PREVIEW_DEPS)

install:
	@$(CWD)/../scripts/migrate-wayvox.sh "$(CWD)/.."
	mkdir -p $(dir $(INSTALL_DIR))
	-plasmapkg2 --remove $(LEGACY_PLUGIN_ID) 2>/dev/null
	ln -snf $(CWD) $(INSTALL_DIR)
	mkdir -p $(CONFIG_DIR)
	@echo "Plasmoid installed. Add the 'Wayvox' widget to your panel or system tray."
	@echo ""
	@echo "To install the D-Bus service (required for Language Buddy): make install-service"

uninstall:
	rm -f $(INSTALL_DIR)
	-plasmapkg2 --remove $(PLUGIN_ID) 2>/dev/null
	@echo "Plasmoid uninstalled."

install-service:
	mkdir -p $(SERVICE_DIR)
	-systemctl --user disable --now whisper-npu-kde.service 2>/dev/null
	rm -f $(SERVICE_DIR)/whisper-npu-kde.service
	while IFS= read -r line; do line="$${line//@WAYVOX_PROJECT_DIR@/$(PROJECT_DIR)}"; printf '%s\n' "$${line//@WAYVOX_PYTHON@/$(PYTHON)}"; done < $(CWD)/wayvox-kde.service > $(SERVICE_DIR)/wayvox-kde.service
	systemctl --user daemon-reload
	systemctl --user enable wayvox-kde.service
	systemctl --user start wayvox-kde.service
	@echo "D-Bus service installed and started."

uninstall-service: disable-service
	rm -f $(SERVICE_DIR)/wayvox-kde.service
	systemctl --user daemon-reload
	@echo "D-Bus service uninstalled."

enable-service:
	systemctl --user enable wayvox-kde.service
	systemctl --user start wayvox-kde.service

disable-service:
	-systemctl --user stop wayvox-kde.service
	-systemctl --user disable wayvox-kde.service

dev: install install-service
	@echo ""
	@echo "========================================"
	@echo "Wayvox KDE plasmoid installed!"
	@echo "========================================"
	@echo ""
	@echo "1. Right-click panel > Add Widgets > search 'Wayvox'"
	@echo "2. Or add to system tray via System Tray Settings"
	@echo ""
	@echo "Watch logs:   make logs"
	@echo "Service logs: make logs-service"

# ---------- Dependencies ------------------------------------------

deps:
	@echo "Installing KDE plasmoid dependencies..."
	sudo dnf install -y $(KDE_ALL_DEPS)
	@echo "Dependencies installed."

# ---------- Preview / testing (no Plasma session required) --------

preview: deps install
	@echo "Starting plasmoid preview with plasmawindowed..."
	@echo "Close the window or press Ctrl+C to stop."
	@echo ""
	plasmawindowed $(PLUGIN_ID) 2>&1

preview-viewer: deps install
	@echo "Starting plasmoid preview with plasmoidviewer..."
	@echo "Use the toolbar to switch form factor (panel / desktop / sidebar)."
	@echo "Close the window or press Ctrl+C to stop."
	@echo ""
	plasmoidviewer -a $(PLUGIN_ID) 2>&1

preview-dbus: deps
	@echo "Starting D-Bus overlay service in foreground..."
	@echo "Test with:"
	@echo '  dbus-send --session --dest=com.whisper.LanguageBuddy --print-reply \'
	@echo '    /com/whisper/LanguageBuddy com.whisper.LanguageBuddy.HandleTranscription \'
	@echo '    string:"Hello this is a test transcription"'
	@echo ""
	python3 $(CWD)/wayvox_kde_service.py

preview-stop:
	-pkill -f "plasmawindowed $(PLUGIN_ID)" 2>/dev/null
	-pkill -f "plasmoidviewer.*$(PLUGIN_ID)" 2>/dev/null
	-pkill -f "wayvox_kde_service" 2>/dev/null
	@echo "Preview processes stopped."

# ------------------------------------------------------------------

logs:
	journalctl --user -f -u wayvox-push-to-talk.service -u wayvox-server.service

logs-service:
	journalctl --user -f -u wayvox-kde.service

clean:
	@echo "Nothing to clean."
