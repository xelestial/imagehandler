from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class BatchResult:
    operation: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    moved_to_complete: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _is_under_complete(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return "complete" in rel.parts


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
        if _is_under_complete(p, root):
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


def move_input_to_complete(input_file: Path, input_root: Path) -> Path | None:
    input_file = Path(input_file)
    input_root = Path(input_root)
    if not input_file.exists() or input_root.is_file():
        return None
    if _is_under_complete(input_file, input_root):
        return None
    try:
        rel = input_file.relative_to(input_root)
    except ValueError:
        return None
    complete_root = input_root / 'complete'
    dst = complete_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while True:
            candidate = dst.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                dst = candidate
                break
            counter += 1
    shutil.move(str(input_file), str(dst))
    return dst
