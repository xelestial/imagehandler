from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .bg_refine import refine_foreground_mask_against_background, zero_transparent_rgb
from .debug import save_mask
from .io import load_image, sidecar_path
from .mask_ops import (
    apply_mask_as_alpha,
    bbox_from_mask,
    foreground_mask_from_background,
    mask_metrics,
)
from .reports import OperationReport

DEFAULT_REMBG_MODEL = "birefnet-general"


def normalize_rembg_model(model: str | None) -> str:
    text = (model or "").strip()
    if not text or text.lower() in {"auto", "default", "none"}:
        return DEFAULT_REMBG_MODEL
    return text


@lru_cache(maxsize=12)
def _rembg_session(model: str | None):
    try:
        from rembg import new_session
    except ImportError as exc:
        raise RuntimeError(
            "rembg backend is not installed. Install with: pip install -e '.[bg]'"
        ) from exc
    return new_session(normalize_rembg_model(model))


def remove_background(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = "auto",
    model: str | None = None,
    alpha_matting: bool = False,
    mask_only: bool = False,
    postprocess: bool = True,
    feather: float = 0.0,
    cleanup_mode: str = "safe",
) -> OperationReport:
    source_image = load_image(input_path, "RGBA")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_backend = _select_backend(backend)
    backend_label = selected_backend
    warnings: list[str] = []

    if selected_backend == "rembg":
        rembg_model = normalize_rembg_model(model)
        rgba, soft_alpha = _remove_with_rembg(source_image, model=rembg_model, alpha_matting=alpha_matting)
        backend_label = f"rembg:{rembg_model}"
    elif selected_backend == "transparent":
        rgba, soft_alpha = _remove_with_transparent_background(source_image)
    elif selected_backend == "classical":
        support_mask = foreground_mask_from_background(source_image)
        soft_alpha = (support_mask.astype(np.uint8) * 255)
        rgba = apply_mask_as_alpha(source_image, support_mask)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    support_mask = soft_alpha > 8

    if postprocess:
        # Preserve the model's soft alpha. The support mask is only allowed to
        # zero out pixels that are definitely outside the foreground. Never
        # replace the alpha channel with a binary mask, because that creates
        # jagged edges and destroys hair/skin antialiasing.
        refined_support = refine_foreground_mask_against_background(
            source_image,
            support_mask,
            mode=cleanup_mode,
        )
        soft_alpha = _clip_soft_alpha_to_support(soft_alpha, refined_support)
        support_mask = soft_alpha > 8
        rgba = _apply_soft_alpha(rgba, soft_alpha)

    if feather > 0:
        rgba = _feather_alpha(rgba, radius=feather)
        soft_alpha = _extract_alpha(rgba)
        support_mask = soft_alpha > 8

    rgba = zero_transparent_rgb(rgba)
    soft_alpha = _extract_alpha(rgba)
    mask = soft_alpha > 8

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
        backend=backend_label,
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


def _extract_alpha(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"))[:, :, 3].copy()


def _apply_soft_alpha(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgba = np.array(image.convert("RGBA"), copy=True)
    rgba[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _clip_soft_alpha_to_support(alpha: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.array(alpha, copy=True)
    out[~support.astype(bool)] = 0
    return out


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
    return out, _extract_alpha(out)


def _remove_with_transparent_background(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    try:
        from transparent_background import Remover
    except ImportError as exc:
        raise RuntimeError(
            "transparent-background backend is not installed. Install with: pip install -e '.[transparent]'"
        ) from exc

    remover = Remover()
    out = remover.process(image.convert("RGB"), type="rgba").convert("RGBA")
    return out, _extract_alpha(out)


def _feather_alpha(image: Image.Image, radius: float) -> Image.Image:
    rgba = image.convert("RGBA")
    r, g, b, a = rgba.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    rgba.putalpha(a)
    return rgba
