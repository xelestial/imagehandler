from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class BatchResult:
    operation: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def iter_image_files(input_path: str | Path, recursive: bool = False, pattern: str | None = None) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        return [root]
    glob_pattern = pattern or ("**/*" if recursive else "*")
    candidates = sorted(root.glob(glob_pattern))
    return [p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]


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
