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
IS_LINUX=0
IS_WSL=0
[[ "$OS_NAME" == "Darwin" ]] && IS_MAC=1
[[ "$OS_NAME" == "Linux" ]] && IS_LINUX=1
if [[ "$IS_LINUX" -eq 1 ]]; then
  if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null || [[ -n "${WSL_INTEROP:-}" ]] || [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    IS_WSL=1
  fi
fi

log() { printf '\033[1;34m[dependency]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./dependency.sh [options]

Default behavior:
  Installs/checks system dependencies, finds Python 3.11-3.13, then runs ./setup.sh.

Important:
  Python 3.14 is intentionally not used for this project yet. Several image / ML
  wheels may lag behind the newest CPython release. On WSL/Linux this script
  tries to install and use python3.12 first.

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
  ./dependency.sh --python 3.12
  ./dependency.sh --setup-arg --skip-smoke
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|--check-only) INSTALL=0; RUN_SETUP=0 ;;
    --install) INSTALL=1 ;;
    --run-setup) RUN_SETUP=1 ;;
    --no-setup) RUN_SETUP=0 ;;
    --fix-owner) FIX_OWNER=1 ;;
    --python)
      [[ $# -ge 2 ]] || fail "--python requires a value"
      PYTHON_VERSION="$2"
      shift
      ;;
    --setup-arg)
      [[ $# -ge 2 ]] || fail "--setup-arg requires a value"
      SETUP_ARGS+=("$2")
      shift
      ;;
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

python_version_text() {
  "$1" - <<'PY' 2>/dev/null || true
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

find_python() {
  local candidates=(
    "python${PYTHON_VERSION}" python3.13 python3.12 python3.11
    "/opt/homebrew/bin/python${PYTHON_VERSION}" /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11
    "/opt/homebrew/opt/python@${PYTHON_VERSION}/bin/python${PYTHON_VERSION}" /opt/homebrew/opt/python@3.13/bin/python3.13 /opt/homebrew/opt/python@3.12/bin/python3.12 /opt/homebrew/opt/python@3.11/bin/python3.11
    "/usr/local/bin/python${PYTHON_VERSION}" /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11
    python3 python
  )
  local p
  for p in "${candidates[@]}"; do
    if ([[ -x "$p" ]] || command -v "$p" >/dev/null 2>&1) && python_ok "$p"; then
      command -v "$p" 2>/dev/null || printf '%s\n' "$p"
      return
    fi
  done
}

show_rejected_default_python() {
  if command -v python3 >/dev/null 2>&1; then
    local v
    v="$(python_version_text python3)"
    if [[ -n "$v" ]]; then
      warn "Default python3 is $v. Python 3.14 is not supported by this setup yet; using Python 3.12/3.13/3.11 instead."
    fi
  fi
}

have_sudo() {
  command -v sudo >/dev/null 2>&1
}

apt_has_package() {
  local package="$1"
  apt-cache policy "$package" 2>/dev/null | awk '/Candidate:/ {print $2}' | grep -vq '(none)'
}

apt_install() {
  [[ "$INSTALL" -eq 1 ]] || return 1
  command -v apt-get >/dev/null 2>&1 || return 1
  have_sudo || fail "sudo is required for apt install. Install Python 3.12 manually or run in an environment with sudo."

  log "Updating apt package index"
  sudo apt-get update
  log "Installing Linux build/runtime dependencies"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git build-essential pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
}

ensure_linux_python() {
  [[ "$IS_LINUX" -eq 1 ]] || return
  [[ "$INSTALL" -eq 1 ]] || return
  command -v apt-get >/dev/null 2>&1 || return

  show_rejected_default_python
  apt_install || return

  local versions=("$PYTHON_VERSION" 3.13 3.12 3.11)
  local version package
  for version in "${versions[@]}"; do
    package="python${version}"
    if apt_has_package "$package"; then
      log "Installing $package and venv/dev packages"
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "$package" "$package-venv" "$package-dev"
      return
    fi
  done

  warn "apt did not provide python3.11-3.13 packages on this distro."
  warn "For Ubuntu 22.04, install Python 3.12 via deadsnakes or use Ubuntu 24.04+ WSL."
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
if [[ "$IS_WSL" -eq 1 ]]; then
  log "WSL detected"
fi

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

if [[ -z "$PY" && "$INSTALL" -eq 1 && "$IS_LINUX" -eq 1 ]]; then
  ensure_linux_python
  PY="$(find_python || true)"
fi

if [[ -z "$PY" ]]; then
  if [[ "$IS_LINUX" -eq 1 ]]; then
    cat >&2 <<EOF
[error] Python 3.11-3.13 not found.

This project currently avoids Python 3.14 because some image/ML dependencies may
not have stable wheels for it yet.

Try one of these:
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
  ./dependency.sh --python 3.12

If your Ubuntu release does not provide python3.12, use Ubuntu 24.04+ WSL or
install Python 3.12 through a trusted package source.
EOF
    exit 1
  fi
  fail "Python 3.11-3.13 not found. Run: brew install python@${PYTHON_VERSION}"
fi

printf '%s\n' "$PY" > "$PYTHON_BIN_FILE"
cat > "$ENV_FILE" <<EOF
export PYTHON_BIN="$PY"
EOF

log "Python: $PY ($(python_version_text "$PY"))"
log "Wrote $PYTHON_BIN_FILE and $ENV_FILE"

"$PY" - <<'PY'
import venv
print('venv module OK')
try:
    import ensurepip
    print('ensurepip module OK')
except Exception as exc:
    print(f'ensurepip module check skipped/warn: {exc}')
PY

if [[ "$RUN_SETUP" -eq 1 ]]; then
  log "Running setup.sh"
  if [[ "${#SETUP_ARGS[@]}" -gt 0 ]]; then
    PYTHON_BIN="$PY" "$ROOT_DIR/setup.sh" "${SETUP_ARGS[@]}"
  else
    PYTHON_BIN="$PY" "$ROOT_DIR/setup.sh"
  fi
else
  cat <<EOF

Next:
  PYTHON_BIN="$(cat "$PYTHON_BIN_FILE")" ./setup.sh

Or:
  source "$ENV_FILE"
  ./setup.sh
EOF
fi
