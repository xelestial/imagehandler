from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IGNORED_WORKSPACE_PARTS = {"jobs", "failed", "reports", "tmp"}


@dataclass
class BatchResult:
    operation: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    moved_to_job_input: list[str] = field(default_factory=list)
    moved_to_failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _has_ignored_part(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(part in IGNORED_WORKSPACE_PARTS for part in rel.parts)


def iter_image_files(input_path: str | Path, recursive: bool = False, pattern: str | None = None) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        return [root] if _is_supported_image(root) else []
    glob_pattern = pattern or ("**/*" if recursive else "*")
    candidates = sorted(root.glob(glob_pattern))
    files = []
    for p in candidates:
        if not _is_supported_image(p):
            continue
        if _has_ignored_part(p, root):
            continue
        files.append(p)
    return files


def relative_output_file(input_file: Path, input_root: Path, output_root: Path, suffix: str = ".png") -> Path:
    if input_root.is_file():
        rel = Path(input_file.stem + suffix)
    else:
        rel = input_file.relative_to(input_root).with_suffix(suffix)
    return output_root / rel


def relative_output_dir(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        rel = Path(input_file.stem)
    else:
        rel = input_file.relative_to(input_root).with_suffix("")
    return output_root / rel


def _safe_destination(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    counter = 1
    while True:
        candidate = dst.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _relative_source(input_file: Path, input_root: Path) -> Path | None:
    input_file = Path(input_file)
    input_root = Path(input_root)
    if not input_file.exists() or input_root.is_file():
        return None
    try:
        return input_file.relative_to(input_root)
    except ValueError:
        return None


def move_input_to_job_input(input_file: Path, input_root: Path, job_input_root: Path) -> Path | None:
    rel = _relative_source(input_file, input_root)
    if rel is None:
        return None
    if any(part in IGNORED_WORKSPACE_PARTS for part in rel.parts):
        return None
    dst = _safe_destination(Path(job_input_root) / rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(input_file), str(dst))
    return dst


def move_input_to_failed(input_file: Path, input_root: Path) -> Path | None:
    rel = _relative_source(input_file, input_root)
    if rel is None:
        return None
    if any(part in {"jobs", "failed"} for part in rel.parts):
        return None
    failed_root = input_root.parent / "failed" if input_root.name == "input" else input_root / "failed"
    dst = _safe_destination(failed_root / rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(input_file), str(dst))
    return dst
