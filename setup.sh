#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
MODE="cpu"
WITH_TRANSPARENT=0
WITH_MATTING=0
WITH_DEV=0
SKIP_SMOKE=0

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
ARCH_NAME="$(uname -m 2>/dev/null || echo unknown)"
IS_MAC=0
[[ "$OS_NAME" == "Darwin" ]] && IS_MAC=1

log() { printf '[setup] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
fail() { printf '[error] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./setup.sh [--cpu|--gpu] [--transparent] [--matting] [--dev] [--all] [--skip-smoke]

macOS: use default --cpu. --gpu is for NVIDIA/CUDA and falls back to CPU on normal macOS.
Use PYTHON_BIN=python3.12 ./setup.sh when multiple Python versions exist.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --transparent) WITH_TRANSPARENT=1 ;;
    --matting) WITH_MATTING=1 ;;
    --dev) WITH_DEV=1 ;;
    --all) WITH_TRANSPARENT=1; WITH_MATTING=1; WITH_DEV=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
  shift
done

cd "$ROOT_DIR"
log "Detected OS: $OS_NAME / $ARCH_NAME"

if [[ "$IS_MAC" -eq 1 ]]; then
  [[ "$ARCH_NAME" == "arm64" ]] && log "macOS Apple Silicon detected" || log "macOS Intel detected"
  xcode-select -p >/dev/null 2>&1 || warn "Xcode Command Line Tools not found. Run: xcode-select --install"
  if [[ "$MODE" == "gpu" ]]; then
    warn "rembg GPU mode is NVIDIA/CUDA based, so normal macOS falls back to CPU."
    MODE="cpu"
  fi
fi

python_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
v = sys.version_info
raise SystemExit(0 if ((v.major, v.minor) >= (3, 11) and (v.major, v.minor) < (3, 14)) else 1)
PY
}

find_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "PYTHON_BIN not found: $PYTHON_BIN"
    python_ok "$PYTHON_BIN" || fail "PYTHON_BIN must be Python >=3.11 and <3.14"
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  for p in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$p" >/dev/null 2>&1 && python_ok "$p"; then
      printf '%s\n' "$p"
      return
    fi
  done
  if [[ "$IS_MAC" -eq 1 ]]; then
    fail "Python 3.11-3.13 not found. Try: brew install python@3.12 && PYTHON_BIN=python3.12 ./setup.sh"
  fi
  fail "Python 3.11-3.13 not found."
}

PY="$(find_python)"
"$PY" - <<'PY'
import platform, sys
v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro} / {platform.platform()}")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv: $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
PY="python"

log "Upgrading pip tools"
"$PY" -m pip install --upgrade pip setuptools wheel

log "Installing imagehandler"
"$PY" -m pip install -e .

if [[ "$MODE" == "gpu" ]]; then
  log "Installing rembg GPU backend"
  "$PY" -m pip install "rembg[gpu]>=2.0.0"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || warn "nvidia-smi not found"
else
  log "Installing rembg CPU backend"
  "$PY" -m pip install "rembg[cpu]>=2.0.0"
fi

[[ "$WITH_TRANSPARENT" -eq 1 ]] && "$PY" -m pip install "transparent-background>=1.3.4"
[[ "$WITH_MATTING" -eq 1 ]] && "$PY" -m pip install "pymatting>=1.1"
[[ "$WITH_DEV" -eq 1 ]] && "$PY" -m pip install "pytest>=8.0" "ruff>=0.5"

log "Checking imports"
"$PY" - <<'PY'
import importlib
required = ["numpy", "PIL", "cv2", "scipy", "skimage", "typer", "rich", "pydantic", "imagehandler"]
optional = ["rembg", "transparent_background", "pymatting"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
        print("OK", name)
    except Exception as exc:
        print("FAIL", name, exc)
        missing.append(name)
for name in optional:
    try:
        importlib.import_module(name)
        print("OK optional", name)
    except Exception as exc:
        print("SKIP optional", name, exc)
try:
    import onnxruntime as ort
    print("onnxruntime providers:", ort.get_available_providers())
except Exception as exc:
    print("SKIP onnxruntime", exc)
if missing:
    raise SystemExit(1)
PY

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  log "CLI smoke test"
  imagehandler --help >/dev/null || fail "imagehandler CLI smoke test failed"
fi

log "Setup complete. Activate with: source $VENV_DIR/bin/activate"
