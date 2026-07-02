from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from .fallback import remove_background_with_fallback
from .io import ensure_output_dir, load_image
from .mask_ops import clean_mask, components_bboxes, foreground_mask_from_background, normalize_size, sort_boxes_reading_order, to_square_canvas
from .reports import BBox, OperationReport


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
    out_dir = ensure_output_dir(output_dir)
    image = load_image(source, "RGBA")
    width, height = image.size

    support_mask, preprocess_info = _build_safe_support_mask(source, image, out_dir, threshold, debug)
    safe_mask = _clean_safe_mask(support_mask, width, height)
    boxes = _top_level_boxes(safe_mask, width, height, min_area)

    warnings: list[str] = []
    if not boxes:
        warnings.append("No top-level foreground items were detected.")
    if len(boxes) > 80:
        warnings.append("Many top-level items were detected; review input or increase min_area.")

    outputs: list[str] = []
    for idx, box in enumerate(boxes, start=1):
        crop = _crop_top_level_item(
            image=image,
            mask=safe_mask,
            box=box,
            padding=padding,
            transparent_bg=transparent_bg,
        )
        if square_canvas:
            crop = to_square_canvas(crop)
        if normalize:
            crop = normalize_size(crop, normalize)
        path = out_dir / f"item_{idx:03d}.png"
        crop.save(path)
        outputs.append(str(path))

    coverage = _safe_coverage_metrics(safe_mask, boxes)
    report = OperationReport(
        ok=len(warnings) == 0,
        operation="extract-items",
        source=str(source),
        mode="safe-top-level-components",
        warnings=warnings,
        metrics={
            "items": len(boxes),
            "safe_top_level_components": len(boxes),
            "image_width": width,
            "image_height": height,
            "min_area": min_area,
            "merge_distance_ignored": merge_distance,
            "safe_policy": "top-level-object-islands-no-internal-splitting",
            **preprocess_info,
            **coverage,
        },
        boxes=boxes,
        outputs=outputs,
    )
    report.save(out_dir / "manifest.json")

    if debug:
        _save_mask(safe_mask, out_dir / "debug_safe_support_mask.png")
        _save_boxes_overlay(image, boxes, out_dir / "debug_safe_boxes.png")
    return report


def _build_safe_support_mask(source: Path, image: Image.Image, out_dir: Path, threshold: float, debug: bool) -> tuple[np.ndarray, dict[str, object]]:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    info: dict[str, object] = {}

    if float((alpha < 250).mean()) > 0.005:
        support = alpha > 8
        info.update({
            "items_preprocess": "existing_alpha_safe_support",
            "alpha_foreground_pixels": int(support.sum()),
        })
        return support.astype(bool), info

    try:
        with TemporaryDirectory(prefix="imagehandler_items_safe_") as tmp:
            removed = Path(tmp) / "removed.png"
            report, summary = remove_background_with_fallback(
                input_path=source,
                output_path=removed,
                backend="auto",
                model=None,
                alpha_matting=False,
                mask_only=False,
                postprocess=True,
                feather=0.0,
                accept_verdict="WARN",
                min_score=70.0,
                head_refine=False,
                bisenet_onnx=None,
                head_debug=False,
            )
            if not removed.exists():
                raise RuntimeError("background fallback pipeline produced no output")
            removed_img = Image.open(removed).convert("RGBA")
            if removed_img.size != image.size:
                removed_img = removed_img.resize(image.size, Image.Resampling.BILINEAR)
            alpha2 = np.asarray(removed_img)[:, :, 3]
            support = alpha2 > 8
            if debug:
                Image.fromarray(alpha2, mode="L").save(out_dir / "debug_safe_preprocess_alpha.png")
            info.update({
                "items_preprocess": "alpha_from_bg_fallback_safe_support",
                "items_preprocess_backend": report.backend or "auto",
                "items_preprocess_selected_attempt": summary.selected_attempt,
                "items_preprocess_selected_verdict": summary.selected_verdict,
                "items_preprocess_selected_score": float(summary.selected_score),
                "alpha_foreground_pixels": int(support.sum()),
            })
            return support.astype(bool), info
    except Exception as exc:
        fallback = foreground_mask_from_background(image, threshold=threshold)
        info.update({
            "items_preprocess": "rgb_threshold_safe_support",
            "items_preprocess_error": str(exc),
            "rgb_foreground_pixels": int(fallback.sum()),
        })
        return fallback.astype(bool), info


def _clean_safe_mask(mask: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    short = min(image_width, image_height)
    open_size = max(3, int(short * 0.0018))
    close_size = max(3, int(short * 0.0025))
    if open_size % 2 == 0:
        open_size += 1
    if close_size % 2 == 0:
        close_size += 1
    open_size = min(open_size, 7)
    close_size = min(close_size, 9)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)))
    cleaned = clean_mask(closed.astype(bool), open_size=1, close_size=3, fill_holes=True)
    return cleaned.astype(bool)


def _top_level_boxes(mask: np.ndarray, image_width: int, image_height: int, min_area: int) -> list[BBox]:
    min_pixels = max(min_area, int(image_width * image_height * 0.00018), 64)
    raw = components_bboxes(mask.astype(bool), min_area=min_pixels, min_size=5)
    boxes: list[BBox] = []
    for box, area in raw:
        if area < min_pixels:
            continue
        if box.width < 6 or box.height < 6:
            continue
        if box.area > image_width * image_height * 0.86:
            continue
        boxes.append(_tighten_box_to_mask(mask, box))
    return sort_boxes_reading_order(_dedupe_similar_boxes(boxes))


def _tighten_box_to_mask(mask: np.ndarray, box: BBox) -> BBox:
    local = mask[box.top : box.bottom, box.left : box.right]
    if not local.any():
        return box
    ys, xs = np.where(local)
    return BBox(box.left + int(xs.min()), box.top + int(ys.min()), box.left + int(xs.max()) + 1, box.top + int(ys.max()) + 1)


def _crop_top_level_item(image: Image.Image, mask: np.ndarray, box: BBox, padding: int, transparent_bg: bool) -> Image.Image:
    width, height = image.size
    padded = box.padded(padding, width, height)
    rgba = np.asarray(image.convert("RGBA")).copy()
    local = rgba[padded.top : padded.bottom, padded.left : padded.right].copy()
    local_mask = mask[padded.top : padded.bottom, padded.left : padded.right].astype(bool)
    if transparent_bg:
        local[:, :, 3] = np.where(local_mask, local[:, :, 3], 0).astype(np.uint8)
    else:
        local[~local_mask, 0] = 255
        local[~local_mask, 1] = 255
        local[~local_mask, 2] = 255
        local[~local_mask, 3] = 255
    return Image.fromarray(local, mode="RGBA")


def _safe_coverage_metrics(mask: np.ndarray, boxes: list[BBox]) -> dict[str, float | int]:
    support = mask.astype(bool)
    fg = int(support.sum())
    if fg == 0:
        return {"coverage_ratio": 0.0, "duplication_ratio": 0.0, "covered_pixels": 0, "foreground_pixels": 0}
    covered = np.zeros_like(support, dtype=bool)
    duplicate_sum = 0
    for box in boxes:
        local = support[box.top : box.bottom, box.left : box.right]
        duplicate_sum += int(local.sum())
        covered[box.top : box.bottom, box.left : box.right] |= local
    covered_pixels = int(covered.sum())
    return {
        "coverage_ratio": float(covered_pixels / fg),
        "duplication_ratio": float(duplicate_sum / max(1, covered_pixels)),
        "covered_pixels": covered_pixels,
        "foreground_pixels": fg,
    }


def _dedupe_similar_boxes(boxes: list[BBox]) -> list[BBox]:
    out: list[BBox] = []
    for box in boxes:
        if any(_iou(box, other) > 0.92 for other in out):
            continue
        out.append(box)
    return out


def _iou(a: BBox, b: BBox) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    union = max(1, a.area + b.area - inter)
    return float(inter / union)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _save_boxes_overlay(image: Image.Image, boxes: list[BBox], path: Path) -> None:
    from PIL import ImageDraw

    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle([box.left, box.top, box.right - 1, box.bottom - 1], outline=(255, 0, 0), width=3)
    overlay.save(path)
