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
  "$WORKSPACE_DIR/bg/input" "$WORKSPACE_DIR/bg/complete" "$WORKSPACE_DIR/bg/jobs" \
  "$WORKSPACE_DIR/sheets/input" "$WORKSPACE_DIR/sheets/complete" "$WORKSPACE_DIR/sheets/jobs" \
  "$WORKSPACE_DIR/items/input" "$WORKSPACE_DIR/items/complete" "$WORKSPACE_DIR/items/jobs" \
  "$WORKSPACE_DIR/quality/input" "$WORKSPACE_DIR/quality/complete" "$WORKSPACE_DIR/quality/jobs" \
  "$WORKSPACE_DIR/archive" "$WORKSPACE_DIR/_reports"

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

source "$VENV_DIR/bin/activate"

print_workspace_hint() {
  cat <<EOF

Task-first workspace folders are ready:

  Background removal input:
    $WORKSPACE_DIR/bg/input

  Character sheet input:
    $WORKSPACE_DIR/sheets/input

  Item/equipment sheet input:
    $WORKSPACE_DIR/items/input

  Results:
    $WORKSPACE_DIR/<task>/jobs/<job_name>/output

  Completed source files:
    $WORKSPACE_DIR/<task>/complete

Put your image files into the matching input folder, then choose Quick run.

EOF
}

if [[ $# -eq 0 ]]; then
  print_workspace_hint
  log "Launching interactive menu"
  IMAGEHANDLER_WORKSPACE="$WORKSPACE_DIR" python -m imagehandler.cli menu
  exit $?
fi

case "$1" in
  menu)
    shift
    print_workspace_hint
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

Task-first folders:
  workspace/bg/input        background-removal source images
  workspace/bg/jobs         background-removal results
  workspace/bg/complete     completed background-removal source images

  workspace/sheets/input    character-sheet source images
  workspace/sheets/jobs     character-sheet results
  workspace/sheets/complete completed character-sheet source images

  workspace/items/input     item/equipment-sheet source images
  workspace/items/jobs      item/equipment-sheet results
  workspace/items/complete  completed item/equipment-sheet source images

Examples:
  ./run.sh
  ./run.sh bg batch-remove workspace/bg/input --workspace ./workspace --recursive
  ./run.sh sheet batch-split workspace/sheets/input --workspace ./workspace --views 4 --recursive
  ./run.sh items batch-extract workspace/items/input --workspace ./workspace --recursive

Notes:
  Do not run ./imagehandler. imagehandler/ is a Python package folder.
USAGE
    ;;
  *)
    IMAGEHANDLER_WORKSPACE="$WORKSPACE_DIR" python -m imagehandler.cli "$@"
    ;;
esac
