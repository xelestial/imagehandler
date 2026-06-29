#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR/workspace}"

log() { printf '\033[1;34m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$ROOT_DIR"

mkdir -p \
  "$WORKSPACE_DIR/inbox/bg" \
  "$WORKSPACE_DIR/inbox/sheets" \
  "$WORKSPACE_DIR/inbox/items" \
  "$WORKSPACE_DIR/jobs" \
  "$WORKSPACE_DIR/archive" \
  "$WORKSPACE_DIR/_reports"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  warn "Do not run this with sudo unless you intentionally want root-owned output files."
fi

if [[ ! -d "$VENV_DIR" ]]; then
  warn "Virtual environment was not found: $VENV_DIR"
  if [[ -x "$ROOT_DIR/dependency.sh" ]]; then
    log "Running dependency.sh first..."
    "$ROOT_DIR/dependency.sh"
  elif [[ -f "$ROOT_DIR/dependency.sh" ]]; then
    log "Running dependency.sh first through bash..."
    bash "$ROOT_DIR/dependency.sh"
  else
    fail "dependency.sh not found. Run setup first."
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

print_inbox_hint() {
  cat <<EOF

Workspace folders are ready:

  Background removal input:
    $WORKSPACE_DIR/inbox/bg

  Character sheet input:
    $WORKSPACE_DIR/inbox/sheets

  Item/equipment sheet input:
    $WORKSPACE_DIR/inbox/items

Put your image files into the matching folder, then choose a menu action.
When the menu asks for an input path, press Enter to use the default inbox folder.

EOF
}

if [[ $# -eq 0 ]]; then
  print_inbox_hint
  log "Launching interactive menu"
  IMAGEHANDLER_WORKSPACE="$WORKSPACE_DIR" python -m imagehandler.cli menu
  exit $?
fi

case "$1" in
  menu)
    shift
    print_inbox_hint
    log "Launching interactive menu"
    IMAGEHANDLER_WORKSPACE="$WORKSPACE_DIR" python -m imagehandler.cli menu "$@"
    ;;
  help|--help|-h)
    cat <<'USAGE'
ImageHandler runner

Usage:
  ./run.sh
  ./run.sh menu
  ./run.sh <imagehandler CLI args>

Default folders:
  workspace/inbox/bg       background-removal source images
  workspace/inbox/sheets   character-sheet source images
  workspace/inbox/items    item/equipment-sheet source images
  workspace/jobs           per-job results

Examples:
  ./run.sh
  ./run.sh menu
  ./run.sh bg batch-remove workspace/inbox/bg --workspace ./workspace --recursive
  ./run.sh sheet batch-split workspace/inbox/sheets --workspace ./workspace --views 4 --recursive
  ./run.sh items batch-extract workspace/inbox/items --workspace ./workspace --recursive

Notes:
  Do not run ./imagehandler. imagehandler/ is a Python package folder.
USAGE
    ;;
  *)
    IMAGEHANDLER_WORKSPACE="$WORKSPACE_DIR" python -m imagehandler.cli "$@"
    ;;
esac
