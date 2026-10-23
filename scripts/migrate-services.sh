#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_DIR=${1:?usage: migrate-services.sh SYSTEMD_DIR}
PROJECT_DIR=${2:-}
SYSTEMCTL=${SYSTEMCTL:-systemctl}
RUNTIME_DIR="$PROJECT_DIR"
if [[ -n "$PROJECT_DIR" && "$(basename "$PROJECT_DIR")" == whisper-npu-server ]]; then
	RUNTIME_DIR="$(dirname "$PROJECT_DIR")/wayvox"
fi

# Legacy service names map to Wayvox names. Keep technical environment names
# unchanged, but move user units, drop-ins, and enablement.
legacy_units=(
	whisper-server.service
	whisper-cpp-server.service
	push-to-talk.service
)
new_units=(
	wayvox-server.service
	wayvox-cpp-server.service
	wayvox-push-to-talk.service
)

OLD_VENV_PREFIX="${HOME:-}/.local/share/virtualenvs/whisper-npu-server"
NEW_VENV_PREFIX="${HOME:-}/.local/share/virtualenvs/wayvox"

was_enabled=()
was_active=()
for i in "${!legacy_units[@]}"; do
	old=${legacy_units[$i]}
	state=$($SYSTEMCTL --user is-enabled "$old" 2>/dev/null || true)
	if [[ "$state" == enabled || "$state" == enabled-runtime || "$state" == linked || "$state" == linked-runtime ]]; then
		was_enabled[i]=1
	else
		was_enabled[i]=0
	fi
	active=$($SYSTEMCTL --user is-active "$old" 2>/dev/null || true)
	if [[ "$active" == active || "$active" == activating ]]; then
		was_active[i]=1
	else
		was_active[i]=0
	fi
done

kde_was_enabled=0
kde_state=$($SYSTEMCTL --user is-enabled whisper-npu-kde.service 2>/dev/null || true)
if [[ "$kde_state" == enabled || "$kde_state" == enabled-runtime || "$kde_state" == linked || "$kde_state" == linked-runtime ]]; then
	kde_was_enabled=1
fi

for old in "${legacy_units[@]}"; do
	$SYSTEMCTL --user disable --now "$old" 2>/dev/null || true
done

for i in "${!legacy_units[@]}"; do
	old=${legacy_units[$i]}
	new=${new_units[$i]}
	old_dropin="$SYSTEMD_DIR/$old.d"
	new_dropin="$SYSTEMD_DIR/$new.d"

	if [[ -d "$old_dropin" ]]; then
		mkdir -p "$new_dropin"
		cp -a "$old_dropin/." "$new_dropin/"
		rm -rf "$old_dropin"
		printf 'Migrated drop-ins: %s -> %s\n' "$old_dropin" "$new_dropin"
	fi

	rm -f "$SYSTEMD_DIR/$old"
done

# KDE's D-Bus unit was installed separately by kde-plasmoid/Makefile. Remove
# its old registration here; that make target installs wayvox-kde.service.
$SYSTEMCTL --user disable --now whisper-npu-kde.service 2>/dev/null || true
rm -f "$SYSTEMD_DIR/whisper-npu-kde.service"
if [[ -n "$PROJECT_DIR" && -f "$PROJECT_DIR/kde-plasmoid/wayvox-kde.service" ]]; then
	tmp_service="$SYSTEMD_DIR/wayvox-kde.service.tmp"
	while IFS= read -r line; do
		printf '%s\n' "${line//@WAYVOX_PROJECT_DIR@/$RUNTIME_DIR}"
	done <"$PROJECT_DIR/kde-plasmoid/wayvox-kde.service" >"$tmp_service"
	mv "$tmp_service" "$SYSTEMD_DIR/wayvox-kde.service"
fi

$SYSTEMCTL --user daemon-reload

for i in "${!legacy_units[@]}"; do
	if [[ "${was_enabled[i]}" == 1 ]]; then
		old=${legacy_units[$i]}
		new=${new_units[$i]}
		$SYSTEMCTL --user enable "$new"
		printf 'Migrated enablement: %s -> %s\n' "$old" "$new"
	fi
	if [[ "${was_active[i]}" == 1 ]]; then
		new=${new_units[$i]}
		$SYSTEMCTL --user start "$new"
		printf 'Migrated active state: %s started\n' "$new"
	fi
done

if [[ "$kde_was_enabled" == 1 && -f "$SYSTEMD_DIR/wayvox-kde.service" ]]; then
	$SYSTEMCTL --user enable wayvox-kde.service
	printf 'Migrated enablement: whisper-npu-kde.service -> wayvox-kde.service\n'
fi

# Rewrite migrated units so old checkout, virtualenv, and script paths work
# after a rebrand without requiring manual systemd overrides.
shopt -s nullglob
for service_file in "$SYSTEMD_DIR"/wayvox*.service "$SYSTEMD_DIR"/wayvox*.service.d/*; do
	if [[ ! -f "$service_file" ]]; then
		continue
	fi
	tmp_service="$service_file.tmp"
	while IFS= read -r line; do
		if [[ -n "$PROJECT_DIR" ]]; then
			line=${line//$PROJECT_DIR/$RUNTIME_DIR}
		fi
		if [[ -n "${HOME:-}" ]]; then
			line=${line//$OLD_VENV_PREFIX/$NEW_VENV_PREFIX}
		fi
		line=${line//server-native.py/server_native.py}
		line=${line//server-whisper-cpp.py/server_whisper_cpp.py}
		line=${line//push-to-talk.py/push_to_talk.py}
		line=${line//wayvox-kde-service.py/wayvox_kde_service.py}
		printf '%s\n' "$line"
	done <"$service_file" >"$tmp_service"
	mv "$tmp_service" "$service_file"
done
shopt -u nullglob

printf 'Wayvox service migration complete.\n'
