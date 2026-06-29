#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR/workspace}"
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

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./setup.sh [options]

Options:
  --cpu              Install CPU background-removal stack. Default.
  --gpu              Install NVIDIA/CUDA rembg GPU stack. Not supported on normal macOS.
  --transparent      Also install transparent-background backend.
  --matting          Also install pymatting backend.
  --dev              Also install pytest and ruff.
  --all              Install CPU stack plus transparent, matting, and dev tools.
  --no-venv          Install into the currently active Python environment.
  --skip-smoke       Skip smoke tests after installation.
  --workspace DIR    Create optimized workspace folders under DIR. Default: ./workspace
  -h, --help         Show this help.

Optimized workspace:
  workspace/bg/input
  workspace/bg/jobs/<job>/input
  workspace/bg/jobs/<job>/output
  workspace/bg/failed

  workspace/sheets/input
  workspace/sheets/jobs/<job>/input
  workspace/sheets/jobs/<job>/output
  workspace/sheets/failed

  workspace/items/input
  workspace/items/jobs/<job>/input
  workspace/items/jobs/<job>/output
  workspace/items/failed

  workspace/reports
USAGE
}

if [[ "$IS_MAC" -eq 1 ]]; then
  export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:$PATH"
fi

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  warn "Running setup with sudo/root is not recommended. Run './setup.sh' as your normal user."
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --transparent) WITH_TRANSPARENT=1 ;;
    --matting) WITH_MATTING=1 ;;
    --dev) WITH_DEV=1 ;;
    --all) WITH_TRANSPARENT=1; WITH_MATTING=1; WITH_DEV=1; MODE="cpu" ;;
    --no-venv) USE_VENV=0 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --workspace)
      [[ $# -ge 2 ]] || fail "--workspace requires a path"
      WORKSPACE_DIR="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
  shift
done

cd "$ROOT_DIR"

log "Detected OS: $OS_NAME / $ARCH_NAME"
if [[ "$IS_MAC" -eq 1 && "$MODE" == "gpu" ]]; then
  warn "macOS does not use the rembg NVIDIA/CUDA GPU path. Falling back to --cpu."
  MODE="cpu"
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
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && python_version_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  fail "Python 3.11-3.13 was not found. On macOS run: brew install python@3.12"
}

PY="$(find_python)"
log "Using Python: $PY"

if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment: $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  else
    log "Using existing virtual environment: $VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  PY="python"
fi

log "Upgrading packaging tools"
"$PY" -m pip install --upgrade pip setuptools wheel

log "Installing imagehandler package"
"$PY" -m pip install -e .

install_packages() {
  log "Installing: $*"
  "$PY" -m pip install "$@"
}

case "$MODE" in
  cpu) install_packages "rembg[cpu]>=2.0.0" ;;
  gpu) install_packages "rembg[gpu]>=2.0.0" ;;
  *) fail "Invalid install mode: $MODE" ;;
esac

if [[ "$WITH_TRANSPARENT" -eq 1 ]]; then
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
required = ["numpy", "PIL", "cv2", "scipy", "skimage", "typer", "rich", "pydantic", "imagehandler"]
missing = []
for module in required:
    try:
        importlib.import_module(module)
        print(f"OK required: {module}")
    except Exception as exc:
        print(f"FAIL required: {module} -> {exc}")
        missing.append(module)
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

log "Creating optimized workspace structure"
mkdir -p \
  "$WORKSPACE_DIR/bg/input" "$WORKSPACE_DIR/bg/jobs" "$WORKSPACE_DIR/bg/failed" \
  "$WORKSPACE_DIR/sheets/input" "$WORKSPACE_DIR/sheets/jobs" "$WORKSPACE_DIR/sheets/failed" \
  "$WORKSPACE_DIR/items/input" "$WORKSPACE_DIR/items/jobs" "$WORKSPACE_DIR/items/failed" \
  "$WORKSPACE_DIR/reports"

cat > "$WORKSPACE_DIR/README.txt" <<EOF
ImageHandler optimized workspace

Put source files here:
  bg/input
  sheets/input
  items/input

Success flow:
  <task>/input/source.png
  -> <task>/jobs/<job_name>/input/source.png
  -> <task>/jobs/<job_name>/output/

Failure flow:
  <task>/failed/source.png

Reports:
  reports/
EOF

log "Workspace created: $WORKSPACE_DIR"
log "Setup complete"
if [[ "$USE_VENV" -eq 1 ]]; then
  cat <<EOF

Activate later with:
  source "$VENV_DIR/bin/activate"

Recommended workflow:
  1) Put files into:
     $WORKSPACE_DIR/bg/input
     $WORKSPACE_DIR/sheets/input
     $WORKSPACE_DIR/items/input
  2) Run:
     ./run.sh
EOF
fi
