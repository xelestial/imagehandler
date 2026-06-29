from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageOps


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_image(path: str | Path, mode: str = "RGBA") -> Image.Image:
    """Load image, apply EXIF orientation, and convert mode."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if mode:
        image = image.convert(mode)
    return image


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def iter_images(path: str | Path, recursive: bool = False) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    pattern = "**/*" if recursive else "*"
    return sorted(
        f for f in p.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )


def sidecar_path(output: str | Path, suffix: str) -> Path:
    p = Path(output)
    return p.with_name(f"{p.stem}{suffix}{p.suffix}")
