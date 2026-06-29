#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODE="cpu"
WITH_TRANSPARENT=0
WITH_MATTING=0
WITH_DEV=0
SKIP_SMOKE=0

usage() {
  cat <<'USAGE'
Usage: ./setup.sh [--cpu|--gpu] [--transparent] [--matting] [--dev] [--all] [--skip-smoke]

Default installs base package + rembg CPU backend.
USAGE
}

log() { printf '[setup] %s\n' "$*"; }
fail() { printf '[error] %s\n' "$*" >&2; exit 1; }

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
PY="python3"
command -v "$PY" >/dev/null 2>&1 || PY="python"
command -v "$PY" >/dev/null 2>&1 || fail "Python not found"

log "Checking Python version"
"$PY" - <<'PY'
import sys
v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro}")
if not ((v.major, v.minor) >= (3, 11) and (v.major, v.minor) < (3, 14)):
    raise SystemExit("Python >=3.11 and <3.14 is required")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating venv: $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

log "Upgrading pip tools"
python -m pip install --upgrade pip setuptools wheel

log "Installing base package"
python -m pip install -e .

install_extra() {
  local extra="$1"
  local fallback="$2"
  log "Installing extra: $extra"
  python -m pip install -e ".[${extra}]" || python -m pip install ${fallback}
}

if [[ "$MODE" == "gpu" ]]; then
  install_extra "bg-gpu" '"rembg[gpu]>=2.0.0"'
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
else
  install_extra "bg" '"rembg[cpu]>=2.0.0"'
fi

[[ "$WITH_TRANSPARENT" -eq 1 ]] && install_extra "transparent" 'transparent-background>=1.3.4'
[[ "$WITH_MATTING" -eq 1 ]] && install_extra "matting" 'pymatting>=1.1'
[[ "$WITH_DEV" -eq 1 ]] && install_extra "dev" 'pytest>=8.0 ruff>=0.5'

log "Checking imports"
python - <<'PY'
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
  imagehandler --help >/dev/null
fi

log "Setup complete. Activate with: source $VENV_DIR/bin/activate"
