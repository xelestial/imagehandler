#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTHON_BIN_FILE="$ROOT_DIR/.python-bin"
ENV_FILE="$ROOT_DIR/.imagehandler-env"
INSTALL=1
RUN_SETUP=1
FIX_OWNER=0
SETUP_ARGS=()

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
ARCH_NAME="$(uname -m 2>/dev/null || echo unknown)"
IS_MAC=0
[[ "$OS_NAME" == "Darwin" ]] && IS_MAC=1

log() { printf '\033[1;34m[dependency]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./dependency.sh [options]

Default behavior:
  Installs/checks system dependencies, finds Python 3.11-3.13, then runs ./setup.sh.

Options:
  --check-only       Only check dependencies. Do not install and do not run setup.sh.
  --no-setup         Install/check dependencies but do not run setup.sh.
  --fix-owner        Fix local project ownership if previous sudo runs created root-owned files.
  --python 3.12      Preferred Python version. Default: 3.12
  --setup-arg ARG    Pass one argument to setup.sh. Can be repeated.
  -h, --help         Show this help.

Examples:
  ./dependency.sh
  ./dependency.sh --fix-owner
  ./dependency.sh --check-only
  ./dependency.sh --no-setup
  ./dependency.sh --setup-arg --workspace --setup-arg ./workspace
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|--check-only) INSTALL=0; RUN_SETUP=0 ;;
    --install) INSTALL=1 ;; # backward compatible; default
    --run-setup) RUN_SETUP=1 ;; # backward compatible; default
    --no-setup) RUN_SETUP=0 ;;
    --fix-owner) FIX_OWNER=1 ;;
    --python) PYTHON_VERSION="$2"; shift ;;
    --setup-arg) SETUP_ARGS+=("$2"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
  shift
done

if [[ "$IS_MAC" -eq 1 ]]; then
  export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:$PATH"
fi

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  warn "Do not run dependency/setup with sudo unless you intentionally want root-owned files."
fi

if [[ "$FIX_OWNER" -eq 1 ]]; then
  owner="${SUDO_USER:-$(whoami)}"
  log "Fixing project ownership for $owner"
  sudo chown -R "$owner" "$ROOT_DIR" 2>/dev/null || warn "chown failed or not needed"
fi

python_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
v = sys.version_info
raise SystemExit(0 if ((v.major, v.minor) >= (3, 11) and (v.major, v.minor) < (3, 14)) else 1)
PY
}

find_python() {
  local candidates=(
    "python${PYTHON_VERSION}" python3.13 python3.12 python3.11 python3 python
    "/opt/homebrew/bin/python${PYTHON_VERSION}" /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3
    "/opt/homebrew/opt/python@${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" /opt/homebrew/opt/python@3.13/bin/python3.13 /opt/homebrew/opt/python@3.12/bin/python3.12 /opt/homebrew/opt/python@3.11/bin/python3.11
    "/usr/local/bin/python${PYTHON_VERSION}" /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3
  )
  local p
  for p in "${candidates[@]}"; do
    if ([[ -x "$p" ]] || command -v "$p" >/dev/null 2>&1) && python_ok "$p"; then
      command -v "$p" 2>/dev/null || printf '%s\n' "$p"
      return
    fi
  done
}

ensure_brew() {
  [[ "$IS_MAC" -eq 1 ]] || return
  if command -v brew >/dev/null 2>&1; then
    eval "$(brew shellenv)" || true
    log "Homebrew OK: $(brew --version | head -n 1)"
    return
  fi
  warn "Homebrew not found."
  if [[ "$INSTALL" -eq 1 ]]; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:$PATH"
    command -v brew >/dev/null 2>&1 && eval "$(brew shellenv)" || true
  else
    warn 'Install Homebrew or rerun without --check-only.'
  fi
}

log "Detected OS: $OS_NAME / $ARCH_NAME"
if [[ "$IS_MAC" -eq 1 ]]; then
  xcode-select -p >/dev/null 2>&1 || warn "Xcode Command Line Tools missing. Run: xcode-select --install"
  ensure_brew
fi

PY="$(find_python || true)"
if [[ -z "$PY" && "$INSTALL" -eq 1 && "$IS_MAC" -eq 1 ]]; then
  log "Installing python@${PYTHON_VERSION}"
  brew install "python@${PYTHON_VERSION}" || brew upgrade "python@${PYTHON_VERSION}" || true
  PY="$(find_python || true)"
fi

[[ -n "$PY" ]] || fail "Python 3.11-3.13 not found. Run: brew install python@${PYTHON_VERSION}"

printf '%s\n' "$PY" > "$PYTHON_BIN_FILE"
cat > "$ENV_FILE" <<EOF
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:\$PATH"
export PYTHON_BIN="$PY"
EOF

log "Python: $PY"
log "Wrote $PYTHON_BIN_FILE and $ENV_FILE"

"$PY" - <<'PY'
import ensurepip, venv
print('ensurepip OK')
print('venv OK')
PY

if [[ "$RUN_SETUP" -eq 1 ]]; then
  log "Running setup.sh"
  PYTHON_BIN="$PY" "$ROOT_DIR/setup.sh" "${SETUP_ARGS[@]}"
else
  cat <<EOF

Next:
  PYTHON_BIN="$(cat "$PYTHON_BIN_FILE")" ./setup.sh

Or:
  source "$ENV_FILE"
  ./setup.sh
EOF
fi
