from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .bg_remove import remove_background
from .extract_items_plus import extract_items as extract_items_plus
from .mask_ops import foreground_mask_from_background, pil_to_rgba_array
from .reports import OperationReport


def _has_meaningful_alpha(image: Image.Image) -> bool:
    rgba = pil_to_rgba_array(image)
    alpha = rgba[:, :, 3]
    return float((alpha < 250).mean()) > 0.005


def _alpha_image_from_bg_remove(source: Path, threshold: float, debug_dir: Path | None = None) -> tuple[Path | None, dict[str, object]]:
    info: dict[str, object] = {"items_preprocess": "rgb_threshold"}
    try:
        with TemporaryDirectory(prefix="imagehandler_items_alpha_") as tmp:
            tmp_root = Path(tmp)
            removed = tmp_root / "removed.png"
            report = remove_background(
                input_path=source,
                output_path=removed,
                backend="auto",
                model=None,
                alpha_matting=False,
                mask_only=False,
                postprocess=True,
                cleanup_mode="safe",
                head_refine=False,
            )
            if not removed.exists():
                info["items_preprocess_error"] = "background removal produced no output"
                return None, info

            original = Image.open(source).convert("RGBA")
            removed_img = Image.open(removed).convert("RGBA")
            if removed_img.size != original.size:
                removed_img = removed_img.resize(original.size, Image.Resampling.BILINEAR)

            original_arr = np.asarray(original).copy()
            alpha = np.asarray(removed_img)[:, :, 3]
            original_arr[:, :, 3] = alpha
            alpha_image = tmp_root / "alpha_source.png"
            Image.fromarray(original_arr, mode="RGBA").save(alpha_image)

            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(alpha, mode="L").save(debug_dir / "debug_preprocess_alpha.png")
                Image.fromarray(original_arr, mode="RGBA").save(debug_dir / "debug_preprocess_rgba.png")

            info.update(
                {
                    "items_preprocess": "alpha_from_background_removal",
                    "items_preprocess_backend": report.backend or "auto",
                    "alpha_foreground_pixels": int((alpha > 8).sum()),
                    "alpha_core_pixels": int((alpha > 128).sum()),
                }
            )
            persistent = debug_dir / "_preprocessed_alpha_source.png" if debug_dir is not None else None
            if persistent is not None:
                Image.fromarray(original_arr, mode="RGBA").save(persistent)
                return persistent, info
            # No debug dir means caller cannot use a path after TemporaryDirectory closes.
            return None, info
    except Exception as exc:
        info["items_preprocess_error"] = str(exc)
        return None, info


def extract_items(
    input_path: str | Path,
    output_dir: str | Path,
    padding: int = 16,
    min_area: int = 120,
    merge_distance: int = 12,
    square_canvas: bool = False,
    normalize: int | None = None,
    transparent_bg: bool = False,
    threshold: float = 28.0,
    debug: bool = False,
) -> OperationReport:
    source = Path(input_path)
    out_dir = Path(output_dir)
    original = Image.open(source).convert("RGBA")
    rgb_mask = foreground_mask_from_background(original, threshold=threshold)
    preprocess_info: dict[str, object] = {
        "rgb_foreground_pixels": int(rgb_mask.sum()),
    }

    if _has_meaningful_alpha(original):
        target = source
        alpha = pil_to_rgba_array(original)[:, :, 3]
        preprocess_info.update(
            {
                "items_preprocess": "existing_alpha",
                "alpha_foreground_pixels": int((alpha > 8).sum()),
                "alpha_core_pixels": int((alpha > 128).sum()),
            }
        )
    else:
        # Use output_dir as a persistent debug/cache location so the alpha image
        # remains available while the downstream extractor reads it.
        target, bg_info = _alpha_image_from_bg_remove(source, threshold=threshold, debug_dir=out_dir if debug else out_dir)
        preprocess_info.update(bg_info)
        if target is None:
            target = source

    report = extract_items_plus(
        input_path=target,
        output_dir=out_dir,
        padding=padding,
        min_area=min_area,
        merge_distance=merge_distance,
        square_canvas=square_canvas,
        normalize=normalize,
        transparent_bg=transparent_bg,
        threshold=threshold,
        debug=debug,
    )
    report.source = str(source)
    report.mode = f"alpha-preprocess+{report.mode}"
    report.metrics.update(preprocess_info)
    report.save(out_dir / "manifest.json")
    return report
