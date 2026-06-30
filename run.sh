#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR/workspace}"

log() { printf '\033[1;34m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

python_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
v = sys.version_info
raise SystemExit(0 if ((v.major, v.minor) >= (3, 11) and (v.major, v.minor) < (3, 15)) else 1)
PY
}

python_version_text() {
  "$1" - <<'PY' 2>/dev/null || true
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

archive_bad_venv_if_needed() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    return
  fi

  if python_ok "$VENV_DIR/bin/python"; then
    return
  fi

  local version backup
  version="$(python_version_text "$VENV_DIR/bin/python")"
  backup="$ROOT_DIR/.venv.invalid.$(date +%Y%m%d_%H%M%S)"
  warn "Existing venv Python is unsupported: ${version:-unknown}. Python 3.11-3.14 is required."
  warn "Moving bad venv to: $backup"
  mv "$VENV_DIR" "$backup"
}

cd "$ROOT_DIR"

mkdir -p \
  "$WORKSPACE_DIR/bg/input" "$WORKSPACE_DIR/bg/jobs" "$WORKSPACE_DIR/bg/failed" \
  "$WORKSPACE_DIR/sheets/input" "$WORKSPACE_DIR/sheets/jobs" "$WORKSPACE_DIR/sheets/failed" \
  "$WORKSPACE_DIR/items/input" "$WORKSPACE_DIR/items/jobs" "$WORKSPACE_DIR/items/failed" \
  "$WORKSPACE_DIR/reports"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  warn "Do not run this with sudo unless you intentionally want root-owned output files."
fi

archive_bad_venv_if_needed

if [[ ! -d "$VENV_DIR" ]]; then
  warn "Virtual environment was not found or was invalid: $VENV_DIR"
  if [[ -x "$ROOT_DIR/setup.sh" ]]; then
    log "Running setup.sh first..."
    "$ROOT_DIR/setup.sh"
  else
    fail "setup.sh not found. Run setup first."
  fi
fi

source "$VENV_DIR/bin/activate"

if ! python_ok python; then
  version="$(python_version_text python)"
  fail "Activated Python is unsupported: ${version:-unknown}. Run: ./setup.sh"
fi

print_workspace_hint() {
  cat <<EOF

Optimized workspace folders are ready:

  BG input:
    $WORKSPACE_DIR/bg/input

  Sheets input:
    $WORKSPACE_DIR/sheets/input

  Items input:
    $WORKSPACE_DIR/items/input

  Success flow:
    $WORKSPACE_DIR/<task>/input/source.png
    -> $WORKSPACE_DIR/<task>/jobs/<job_name>/input/source.png
    -> $WORKSPACE_DIR/<task>/jobs/<job_name>/output/

  Failure flow:
    $WORKSPACE_DIR/<task>/failed/source.png

Put files into the matching input folder, then choose Quick run.

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

Python requirement:
  Python 3.11-3.14.

Optimized workspace:
  workspace/bg/input       background-removal source images
  workspace/bg/jobs        per-image background-removal jobs
  workspace/bg/failed      failed background-removal source images

  workspace/sheets/input   character-sheet source images
  workspace/sheets/jobs    per-sheet split jobs
  workspace/sheets/failed  failed character-sheet source images

  workspace/items/input    item/equipment-sheet source images
  workspace/items/jobs     per-sheet extraction jobs
  workspace/items/failed   failed item/equipment source images

Examples:
  ./run.sh
  ./run.sh bg batch-remove workspace/bg/input --workspace ./workspace --recursive
  ./run.sh sheet batch-split workspace/sheets/input --workspace ./workspace --views 4 --recursive
  ./run.sh items batch-extract workspace/items/input --workspace ./workspace --recursive
USAGE
    ;;
  *)
    IMAGEHANDLER_WORKSPACE="$WORKSPACE_DIR" python -m imagehandler.cli "$@"
    ;;
esac
