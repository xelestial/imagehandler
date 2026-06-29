#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR/workspace}"
JOB_NAME="${JOB_NAME:-sample_job}"
MODE="cpu"
WITH_TRANSPARENT=0
WITH_MATTING=0
WITH_DEV=0
USE_VENV=1
SKIP_SMOKE=0

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
ARCH_NAME="$(uname -m 2>/dev/null || echo unknown)"
IS_MAC=0
[[ "$OS_NAME" == "Darwin" ]] && IS_MAC=1

usage() {
  cat <<'USAGE'
Usage: ./setup.sh [options]

Options:
  --cpu              Install CPU background-removal stack. Default and recommended on macOS.
  --gpu              Install NVIDIA/CUDA rembg GPU stack. Not supported on normal macOS.
  --transparent      Also install transparent-background backend.
  --matting          Also install pymatting backend.
  --dev              Also install pytest and ruff.
  --all              Install CPU stack plus transparent, matting, and dev tools.
  --no-venv          Install into the currently active Python environment.
  --skip-smoke       Skip smoke tests after installation.
  --workspace DIR    Create workspace folders under DIR. Default: ./workspace
  --job-name NAME    Create one initial per-job folder. Default: sample_job
  -h, --help         Show this help.

Environment variables:
  PYTHON_BIN=/path/to/python   Python executable to use.
  VENV_DIR=.venv               Virtual environment path. Default: ./.venv
  WORKSPACE_DIR=./workspace    Workspace root path.
  JOB_NAME=sample_job          Initial job folder name.

macOS notes:
  - Use Python 3.11, 3.12, or 3.13.
  - Apple Silicon uses CPU/ONNX Runtime path for rembg in this script.
  - --gpu is for NVIDIA/CUDA Linux/Windows systems, not normal macOS.
USAGE
}

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "$IS_MAC" -eq 1 ]]; then
  # Homebrew paths are often missing when a script is launched through sudo.
  export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:$PATH"
fi

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  warn "Running setup with sudo/root is not recommended. Run './setup.sh' as your normal user unless you have a specific reason."
  warn "If Homebrew Python is installed only for your user, sudo may still hide it on some systems."
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --transparent) WITH_TRANSPARENT=1 ;;
    --matting) WITH_MATTING=1 ;;
    --dev) WITH_DEV=1 ;;
    --all) WITH_TRANSPARENT=1 ; WITH_MATTING=1 ; WITH_DEV=1 ; MODE="cpu" ;;
    --no-venv) USE_VENV=0 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --workspace) WORKSPACE_DIR="$2"; shift ;;
    --job-name) JOB_NAME="$2"; shift ;;
    -h|--help) usage ; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
  shift
done

cd "$ROOT_DIR"

log "Detected OS: $OS_NAME / $ARCH_NAME"
if [[ "$IS_MAC" -eq 1 ]]; then
  if [[ "$ARCH_NAME" == "arm64" ]]; then
    log "macOS Apple Silicon detected"
  else
    log "macOS Intel detected"
  fi

  if ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools not found. If pip builds native packages, run: xcode-select --install"
  fi

  if [[ "$MODE" == "gpu" ]]; then
    warn "macOS does not use the rembg NVIDIA/CUDA GPU path. Falling back to --cpu."
    MODE="cpu"
  fi
fi

python_version_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
v = sys.version_info
raise SystemExit(0 if ((v.major, v.minor) >= (3, 11) and (v.major, v.minor) < (3, 14)) else 1)
PY
}

find_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "PYTHON_BIN not found: $PYTHON_BIN"
    python_version_ok "$PYTHON_BIN" || fail "PYTHON_BIN must be Python >=3.11 and <3.14: $PYTHON_BIN"
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  local candidates=(
    python3.13 python3.12 python3.11 python3 python
    /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3
    /opt/homebrew/opt/python@3.13/bin/python3.13 /opt/homebrew/opt/python@3.12/bin/python3.12 /opt/homebrew/opt/python@3.11/bin/python3.11
    /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3
    /usr/local/opt/python@3.13/bin/python3.13 /usr/local/opt/python@3.12/bin/python3.12 /usr/local/opt/python@3.11/bin/python3.11
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && python_version_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  if [[ "$IS_MAC" -eq 1 ]]; then
    fail "Python 3.11-3.13 was not found. Run: brew install python@3.12 && PYTHON_BIN=/opt/homebrew/bin/python3.12 ./setup.sh   Do not use sudo for normal setup."
  else
    fail "Python 3.11-3.13 was not found. Install Python 3.11, 3.12, or 3.13 first."
  fi
}

PY="$(find_python)"

log "Checking Python version"
"$PY" - <<'PY'
import platform, sys
v = sys.version_info
print(f"Python {v.major}.{v.minor}.{v.micro} ({platform.platform()})")
PY

if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment: $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  else
    log "Using existing virtual environment: $VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY="python"
else
  log "Using current Python environment without creating venv"
fi

log "Upgrading packaging tools"
"$PY" -m pip install --upgrade pip setuptools wheel

log "Installing imagehandler base package"
"$PY" -m pip install -e .

install_packages() {
  log "Installing: $*"
  "$PY" -m pip install "$@"
}

case "$MODE" in
  cpu)
    install_packages "rembg[cpu]>=2.0.0"
    ;;
  gpu)
    install_packages "rembg[gpu]>=2.0.0"
    if command -v nvidia-smi >/dev/null 2>&1; then
      log "Detected NVIDIA GPU"
      nvidia-smi || true
    else
      warn "nvidia-smi was not found. GPU mode needs NVIDIA/CUDA and ONNX Runtime GPU configured."
    fi
    ;;
  *) fail "Invalid install mode: $MODE" ;;
esac

if [[ "$WITH_TRANSPARENT" -eq 1 ]]; then
  if [[ "$IS_MAC" -eq 1 ]]; then
    warn "transparent-background may install PyTorch-related dependencies. On Apple Silicon, first install can be slow."
  fi
  install_packages "transparent-background>=1.3.4"
fi

if [[ "$WITH_MATTING" -eq 1 ]]; then
  install_packages "pymatting>=1.1"
fi

if [[ "$WITH_DEV" -eq 1 ]]; then
  install_packages "pytest>=8.0" "ruff>=0.5"
fi

log "Import check"
"$PY" - <<'PY'
import importlib
import platform

print("Platform:", platform.platform())
print("Machine:", platform.machine())

required = {
    "numpy": "numpy",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "typer": "typer",
    "rich": "rich",
    "pydantic": "pydantic",
    "imagehandler": "imagehandler",
}
optional = {
    "rembg": "rembg",
    "transparent_background": "transparent-background",
    "pymatting": "pymatting",
}

missing = []
for module, package in required.items():
    try:
        importlib.import_module(module)
        print(f"OK required: {module}")
    except Exception as exc:
        print(f"FAIL required: {module} ({package}) -> {exc}")
        missing.append(module)

for module, package in optional.items():
    try:
        importlib.import_module(module)
        print(f"OK optional: {module}")
    except Exception as exc:
        print(f"SKIP optional: {module} ({package}) -> {exc}")

try:
    import onnxruntime as ort
    print("ONNX Runtime providers:", ", ".join(ort.get_available_providers()))
except Exception as exc:
    print(f"SKIP onnxruntime provider check -> {exc}")

if missing:
    raise SystemExit(1)
PY

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  log "Running smoke tests with synthetic images"
  "$PY" - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

from imagehandler.bg_remove import remove_background
from imagehandler.extract_items import extract_items
from imagehandler.split_sheet import split_sheet

root = Path(".setup_smoke")
root.mkdir(exist_ok=True)

sheet = Image.new("RGBA", (720, 240), (255, 255, 255, 255))
d = ImageDraw.Draw(sheet)
for box, color in [
    ((40, 40, 140, 215), (30, 60, 200, 255)),
    ((215, 28, 350, 218), (200, 40, 50, 255)),
    ((420, 52, 500, 212), (40, 160, 70, 255)),
    ((570, 35, 690, 220), (160, 60, 180, 255)),
]:
    d.rectangle(box, fill=color)
sheet_path = root / "synthetic_sheet.png"
sheet.save(sheet_path)

report = split_sheet(sheet_path, root / "split", views=4, debug=True)
print("split-sheet:", report.ok, report.mode, len(report.outputs))
if len(report.outputs) != 4:
    raise SystemExit("split-sheet smoke test failed")

items = Image.new("RGBA", (400, 260), (255, 255, 255, 255))
d = ImageDraw.Draw(items)
d.ellipse((25, 30, 95, 100), fill=(220, 50, 50, 255))
d.rectangle((155, 35, 240, 110), fill=(40, 170, 80, 255))
d.polygon([(300, 35), (370, 100), (280, 125)], fill=(60, 80, 210, 255))
d.rectangle((70, 170, 155, 230), fill=(170, 90, 20, 255))
items_path = root / "synthetic_items.png"
items.save(items_path)

report = extract_items(items_path, root / "items", padding=8, debug=True)
print("extract-items:", report.ok, report.mode, len(report.outputs))
if len(report.outputs) < 4:
    raise SystemExit("extract-items smoke test failed")

obj = Image.new("RGBA", (160, 160), (255, 255, 255, 255))
d = ImageDraw.Draw(obj)
d.rectangle((45, 35, 120, 135), fill=(30, 120, 230, 255))
obj_path = root / "synthetic_bg.png"
obj.save(obj_path)

report = remove_background(obj_path, root / "removed.png", backend="classical")
print("remove-bg classical:", report.ok, report.metrics.get("foreground_area_ratio"))
if not (root / "removed.png").exists():
    raise SystemExit("remove-bg smoke test failed")
PY
else
  warn "Smoke tests skipped"
fi

log "Creating recommended workspace structure"
mkdir -p   "$WORKSPACE_DIR/inbox/bg"   "$WORKSPACE_DIR/inbox/sheets"   "$WORKSPACE_DIR/inbox/items"   "$WORKSPACE_DIR/jobs"   "$WORKSPACE_DIR/archive"   "$WORKSPACE_DIR/_reports"

JOB_ROOT="$WORKSPACE_DIR/jobs/$JOB_NAME"
mkdir -p   "$JOB_ROOT/input/bg"   "$JOB_ROOT/input/sheets"   "$JOB_ROOT/input/items"   "$JOB_ROOT/output/bg"   "$JOB_ROOT/output/sheets"   "$JOB_ROOT/output/items"   "$JOB_ROOT/output/quality"   "$JOB_ROOT/output/logs"   "$JOB_ROOT/tmp"

cat > "$JOB_ROOT/README.txt" <<EOF2
ImageHandler job workspace

Why one folder per job?
- avoids mixing outputs from different tasks
- keeps debug, judge, and fallback files together
- makes archiving and re-running easier

Drop source files here:
  $JOB_ROOT/input/bg
  $JOB_ROOT/input/sheets
  $JOB_ROOT/input/items

Find outputs here:
  $JOB_ROOT/output/bg
  $JOB_ROOT/output/sheets
  $JOB_ROOT/output/items
  $JOB_ROOT/output/quality

Examples:
  imagehandler bg remove "$JOB_ROOT/input/bg/example.png" -o "$JOB_ROOT/output/bg/example.png" --retry-on-fail
  imagehandler sheet split "$JOB_ROOT/input/sheets/example.png" -o "$JOB_ROOT/output/sheets/example" --views 4 --retry-on-fail
  imagehandler items extract "$JOB_ROOT/input/items/example.png" -o "$JOB_ROOT/output/items/example" --retry-on-fail
  imagehandler quality judge "$JOB_ROOT/output/sheets/example" --task split-sheet --expected-count 4
EOF2

log "Workspace created: $WORKSPACE_DIR"
log "Initial per-job folder created: $JOB_ROOT"
log "Setup complete"
if [[ "$USE_VENV" -eq 1 ]]; then
  cat <<EOF3

Activate later with:
  source "$VENV_DIR/bin/activate"

Recommended workflow:
  1) Put files into: $JOB_ROOT/input/bg , $JOB_ROOT/input/sheets , $JOB_ROOT/input/items
  2) Run grouped CLI commands:
     imagehandler bg remove ...
     imagehandler sheet split ...
     imagehandler items extract ...
     imagehandler quality judge ...
  3) Find results under: $JOB_ROOT/output/
EOF3
fi
