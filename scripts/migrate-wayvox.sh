#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${1:?usage: migrate-wayvox.sh PROJECT_DIR}
HOME_DIR=${HOME:?HOME is required}
PROJECT_PARENT=$(dirname "$PROJECT_DIR")
PROJECT_NAME=$(basename "$PROJECT_DIR")
RUNTIME_DIR="$PROJECT_DIR"

if [[ "$PROJECT_NAME" == whisper-npu-server ]]; then
  RUNTIME_DIR="$PROJECT_PARENT/wayvox"
  if [[ ! -e "$RUNTIME_DIR" ]]; then
    ln -s "$PROJECT_DIR" "$RUNTIME_DIR"
    printf 'Created Wayvox checkout path: %s -> %s\n' "$RUNTIME_DIR" "$PROJECT_DIR"
  fi
fi

OLD_VENV_PREFIX="$HOME_DIR/.local/share/virtualenvs/whisper-npu-server"
NEW_VENV_PREFIX="$HOME_DIR/.local/share/virtualenvs/wayvox"
for old_venv in "$OLD_VENV_PREFIX"-*; do
  [[ -e "$old_venv" ]] || continue
  new_venv="${old_venv/$OLD_VENV_PREFIX/$NEW_VENV_PREFIX}"
  if [[ ! -e "$new_venv" ]]; then
    ln -s "$old_venv" "$new_venv"
    printf 'Created Wayvox virtualenv path: %s -> %s\n' "$new_venv" "$old_venv"
  fi
done

move_tree() {
  local old_dir=$1
  local new_dir=$2
  if [[ ! -d "$old_dir" ]]; then
    return
  fi
  if [[ ! -e "$new_dir" ]]; then
    mkdir -p "$(dirname "$new_dir")"
    mv "$old_dir" "$new_dir"
    printf 'Migrated data: %s -> %s\n' "$old_dir" "$new_dir"
    return
  fi
  mkdir -p "$new_dir"
  cp -an "$old_dir"/. "$new_dir"/ 2>/dev/null || true
  printf 'Merged legacy data: %s -> %s\n' "$old_dir" "$new_dir"
}

move_tree "$HOME_DIR/.config/whisper-npu" "$HOME_DIR/.config/wayvox"
move_tree "$HOME_DIR/.cache/whisper-npu" "$HOME_DIR/.cache/wayvox"

# Existing drop-ins often contain absolute paths from the old checkout name.
# Rewrite only those paths. Virtualenv names and technical WHISPER_* variables
# remain unchanged for compatibility.
systemd_dir="$HOME_DIR/.config/systemd/user"
if [[ "$RUNTIME_DIR" != "$PROJECT_DIR" && -d "$systemd_dir" ]]; then
  shopt -s nullglob
  service_files=(
    "$systemd_dir"/wayvox-*.service
    "$systemd_dir"/wayvox-*.service.d/*
  )
  for service_file in "${service_files[@]}"; do
    [[ -f "$service_file" ]] || continue
    tmp_file="$service_file.tmp"
    while IFS= read -r line; do
      line="${line//$PROJECT_DIR/$RUNTIME_DIR}"
      line="${line//$OLD_VENV_PREFIX/$NEW_VENV_PREFIX}"
      printf '%s\n' "$line"
    done < "$service_file" > "$tmp_file"
    mv "$tmp_file" "$service_file"
  done
  shopt -u nullglob
fi

# Migrate GNOME settings while the legacy schema is still installed.
if command -v gsettings >/dev/null 2>&1 && [[ -f "$PROJECT_DIR/gnome-extension/schemas/org.gnome.shell.extensions.wayvox.gschema.xml" ]]; then
  old_schema=org.gnome.shell.extensions.whisper-npu
  new_schema=org.gnome.shell.extensions.wayvox
  schema_dir="$HOME_DIR/.local/share/glib-2.0/schemas"
  mkdir -p "$schema_dir"
  cp "$PROJECT_DIR/gnome-extension/schemas/org.gnome.shell.extensions.wayvox.gschema.xml" "$schema_dir/"
  if command -v glib-compile-schemas >/dev/null 2>&1; then
    glib-compile-schemas "$schema_dir" >/dev/null 2>&1 || true
  fi
  if gsettings list-keys "$old_schema" >/dev/null 2>&1; then
    while IFS= read -r key; do
      value=$(gsettings get "$old_schema" "$key" 2>/dev/null || true)
      if [[ -n "$value" ]]; then
        gsettings set "$new_schema" "$key" "$value" >/dev/null 2>&1 || true
      fi
    done < <(gsettings list-keys "$old_schema" 2>/dev/null)
    printf 'Migrated GNOME settings: %s -> %s\n' "$old_schema" "$new_schema"
  fi
  rm -f "$schema_dir/org.gnome.shell.extensions.whisper-npu.gschema.xml"
  if command -v glib-compile-schemas >/dev/null 2>&1; then
    glib-compile-schemas "$schema_dir" >/dev/null 2>&1 || true
  fi
  gnome-extensions disable whisper-npu@dmz.oneill >/dev/null 2>&1 || true
  rm -rf "$HOME_DIR/.local/share/gnome-shell/extensions/whisper-npu@dmz.oneill"
fi

# Remove old KDE package registration. Config data was migrated above.
plasmapkg2 --remove com.whisper.npu >/dev/null 2>&1 || true

printf 'Wayvox branding migration complete.\n'
