#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN_FILE="$ROOT_DIR/.python-bin"
ENV_FILE="$ROOT_DIR/.imagehandler-env"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR/workspace}"
MODE="cpu"
WITH_TRANSPARENT=0
WITH_MATTING=0
WITH_DEV=0
USE_VENV=1
SKIP_SMOKE=0
CHECK_ONLY=0
FIX_OWNER=0
PREFERRED_PYTHON="${PYTHON_VERSION:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

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

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./setup.sh [options]

Single installer for ImageHandler. This replaces the old split between
setup.sh and dependency.sh.

Options:
  --python 3.14      Preferred Python version or executable. Default: auto.
  --cpu              Install CPU background-removal stack. Default.
  --gpu              Install NVIDIA/CUDA rembg GPU stack.
  --transparent      Also install transparent-background backend.
  --matting          Also install pymatting backend.
  --dev              Also install pytest and ruff.
  --all              Install CPU stack plus transparent, matting, and dev tools.
  --no-venv          Install into the current Python environment.
  --skip-smoke       Skip smoke tests after installation.
  --workspace DIR    Create workspace folders under DIR. Default: ./workspace
  --check-only       Only check Python and system dependency state.
  --fix-owner        Fix local project ownership if previous sudo runs created root-owned files.
  -h, --help         Show this help.

Python policy:
  Python 3.11 through 3.14 is accepted. Existing .venv/bin/python is used first.

Recommended:
  ./setup.sh
  ./run.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || fail "--python requires a value"
      PREFERRED_PYTHON="$2"
      shift
      ;;
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
    --check|--check-only) CHECK_ONLY=1 ;;
    --fix-owner) FIX_OWNER=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
  shift
done

cd "$ROOT_DIR"

if [[ "$IS_MAC" -eq 1 ]]; then
  export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:$PATH"
fi

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  warn "Running setup with sudo/root is not recommended. Run './setup.sh' as your normal user."
fi

if [[ "$FIX_OWNER" -eq 1 ]]; then
  owner="${SUDO_USER:-$(whoami)}"
  log "Fixing project ownership for $owner"
  sudo chown -R "$owner" "$ROOT_DIR" 2>/dev/null || warn "chown failed or not needed"
fi

python_version_ok() {
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

resolve_python_bin_file() {
  if [[ ! -f "$PYTHON_BIN_FILE" ]]; then
    return 1
  fi
  local p
  p="$(tr -d '\r\n' < "$PYTHON_BIN_FILE")"
  [[ -n "$p" ]] || return 1
  if ([[ -x "$p" ]] || command -v "$p" >/dev/null 2>&1) && python_version_ok "$p"; then
    command -v "$p" 2>/dev/null || printf '%s\n' "$p"
    return 0
  fi
  return 1
}

find_python() {
  if [[ "$USE_VENV" -eq 1 && -x "$VENV_DIR/bin/python" ]] && python_version_ok "$VENV_DIR/bin/python"; then
    printf '%s\n' "$VENV_DIR/bin/python"
    return
  fi

  if [[ -n "$PYTHON_BIN" ]]; then
    if ([[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1) && python_version_ok "$PYTHON_BIN"; then
      command -v "$PYTHON_BIN" 2>/dev/null || printf '%s\n' "$PYTHON_BIN"
      return
    fi
    fail "PYTHON_BIN is not a supported Python 3.11-3.14 executable: $PYTHON_BIN"
  fi

  if resolve_python_bin_file; then
    return
  fi

  local candidates=()
  if [[ -n "$PREFERRED_PYTHON" ]]; then
    if [[ "$PREFERRED_PYTHON" == */* ]]; then
      candidates+=("$PREFERRED_PYTHON")
    else
      candidates+=("python${PREFERRED_PYTHON}")
    fi
  fi
  candidates+=(python3.14 python3.13 python3.12 python3.11 python3 python)
  candidates+=(/usr/local/bin/python3.14 /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11)
  candidates+=(/opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11)

  local p
  for p in "${candidates[@]}"; do
    if ([[ -x "$p" ]] || command -v "$p" >/dev/null 2>&1) && python_version_ok "$p"; then
      command -v "$p" 2>/dev/null || printf '%s\n' "$p"
      return
    fi
  done
}

install_linux_runtime_deps() {
  [[ "$IS_LINUX" -eq 1 ]] || return
  command -v apt-get >/dev/null 2>&1 || return
  command -v sudo >/dev/null 2>&1 || return

  log "Installing Linux/WSL runtime dependencies if needed"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates git build-essential pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
}

archive_bad_venv_if_needed() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    return
  fi
  if python_version_ok "$VENV_DIR/bin/python"; then
    return
  fi
  local version backup
  version="$(python_version_text "$VENV_DIR/bin/python")"
  backup="$ROOT_DIR/.venv.invalid.$(date +%Y%m%d_%H%M%S)"
  warn "Existing venv Python is unsupported: ${version:-unknown}. Moving it to $backup"
  mv "$VENV_DIR" "$backup"
}

log "Detected OS: $OS_NAME / $ARCH_NAME"
if [[ "$IS_WSL" -eq 1 ]]; then
  log "WSL detected"
fi

if [[ "$IS_MAC" -eq 1 && "$MODE" == "gpu" ]]; then
  warn "macOS does not use the rembg NVIDIA/CUDA GPU path. Falling back to --cpu."
  MODE="cpu"
fi

archive_bad_venv_if_needed
install_linux_runtime_deps || warn "Linux runtime dependency install skipped or failed; continuing."

PY="$(find_python || true)"
[[ -n "$PY" ]] || fail "Python 3.11-3.14 was not found. Install Python or run with --python /path/to/python."

log "Using Python: $PY ($(python_version_text "$PY"))"
printf '%s\n' "$PY" > "$PYTHON_BIN_FILE"
cat > "$ENV_FILE" <<EOF
export PYTHON_BIN="$PY"
EOF

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  log "Check-only complete"
  exit 0
fi

if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ "$PY" == "$VENV_DIR/bin/python" ]]; then
    log "Using existing virtual environment: $VENV_DIR"
  else
    if [[ ! -d "$VENV_DIR" ]]; then
      log "Creating virtual environment: $VENV_DIR"
      "$PY" -m venv "$VENV_DIR"
    else
      log "Using existing virtual environment: $VENV_DIR"
    fi
  fi
  source "$VENV_DIR/bin/activate"
  PY="python"
  python_version_ok "$PY" || fail "Activated venv Python is unsupported: $(python_version_text "$PY")"
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

log "Creating workspace structure"
mkdir -p \
  "$WORKSPACE_DIR/bg/input" "$WORKSPACE_DIR/bg/jobs" "$WORKSPACE_DIR/bg/failed" \
  "$WORKSPACE_DIR/sheets/input" "$WORKSPACE_DIR/sheets/jobs" "$WORKSPACE_DIR/sheets/failed" \
  "$WORKSPACE_DIR/items/input" "$WORKSPACE_DIR/items/jobs" "$WORKSPACE_DIR/items/failed" \
  "$WORKSPACE_DIR/reports"

cat > "$WORKSPACE_DIR/README.txt" <<EOF
ImageHandler workspace

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
EOF

log "Setup complete"
cat <<EOF

Recommended workflow:
  ./run.sh
EOF
