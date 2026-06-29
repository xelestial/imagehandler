from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .debug import save_mask
from .io import load_image, sidecar_path
from .mask_ops import (
    apply_mask_as_alpha,
    bbox_from_mask,
    clean_mask,
    foreground_mask_from_background,
    mask_metrics,
)
from .reports import OperationReport


@lru_cache(maxsize=8)
def _rembg_session(model: str | None):
    try:
        from rembg import new_session
    except ImportError as exc:
        raise RuntimeError(
            "rembg backend is not installed. Install with: pip install -e '.[bg]'"
        ) from exc
    return new_session(model) if model else new_session()


def remove_background(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = "auto",
    model: str | None = None,
    alpha_matting: bool = False,
    mask_only: bool = False,
    postprocess: bool = True,
    feather: float = 0.0,
) -> OperationReport:
    image = load_image(input_path, "RGBA")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_backend = _select_backend(backend)
    warnings: list[str] = []

    if selected_backend == "rembg":
        rgba, mask = _remove_with_rembg(image, model=model, alpha_matting=alpha_matting)
    elif selected_backend == "transparent":
        rgba, mask = _remove_with_transparent_background(image)
    elif selected_backend == "classical":
        mask = foreground_mask_from_background(image)
        rgba = apply_mask_as_alpha(image, mask)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if postprocess:
        mask = clean_mask(mask, open_size=3, close_size=5, fill_holes=True)
        rgba = apply_mask_as_alpha(rgba, mask)

    if feather > 0:
        rgba = _feather_alpha(rgba, radius=feather)

    metrics = mask_metrics(mask)
    if metrics["foreground_area_ratio"] < 0.005:
        warnings.append("Foreground mask is extremely small.")
    if metrics["foreground_area_ratio"] > 0.98:
        warnings.append("Foreground mask covers almost the entire image.")
    if metrics["touches_border"]:
        warnings.append("Foreground touches image border; crop may be incomplete.")

    mask_path = sidecar_path(output, ".mask")
    report_path = output.with_name(f"{output.stem}.report.json")

    if mask_only:
        save_mask(mask, output)
        outputs = [str(output)]
    else:
        rgba.save(output)
        save_mask(mask, mask_path)
        outputs = [str(output), str(mask_path)]

    bbox = bbox_from_mask(mask)
    report = OperationReport(
        ok=len(warnings) == 0,
        operation="remove-bg",
        source=str(input_path),
        backend=selected_backend,
        warnings=warnings,
        metrics=metrics,
        boxes=[bbox] if bbox else [],
        outputs=outputs,
    )
    report.save(report_path)
    return report


def _select_backend(backend: str) -> str:
    backend = backend.lower()
    if backend != "auto":
        return backend

    try:
        import rembg  # noqa: F401
        return "rembg"
    except ImportError:
        pass

    try:
        import transparent_background  # noqa: F401
        return "transparent"
    except ImportError:
        pass

    return "classical"


def _remove_with_rembg(
    image: Image.Image,
    model: str | None,
    alpha_matting: bool,
) -> tuple[Image.Image, np.ndarray]:
    from rembg import remove

    session = _rembg_session(model)
    out = remove(image, session=session, alpha_matting=alpha_matting)
    if not isinstance(out, Image.Image):
        out = Image.open(out).convert("RGBA")
    out = out.convert("RGBA")
    alpha = np.asarray(out)[:, :, 3]
    return out, alpha > 8


def _remove_with_transparent_background(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    try:
        from transparent_background import Remover
    except ImportError as exc:
        raise RuntimeError(
            "transparent-background backend is not installed. Install with: pip install -e '.[transparent]'"
        ) from exc

    remover = Remover()
    out = remover.process(image.convert("RGB"), type="rgba").convert("RGBA")
    alpha = np.asarray(out)[:, :, 3]
    return out, alpha > 8


def _feather_alpha(image: Image.Image, radius: float) -> Image.Image:
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    rgba.putalpha(a)
    return rgba
