from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .debug import save_boxes_overlay, save_mask
from .io import ensure_output_dir, load_image
from .mask_ops import (
    apply_mask_as_alpha,
    clean_mask,
    components_bboxes,
    crop_with_padding,
    crop_mask,
    foreground_mask_from_background,
    merge_close_boxes,
    normalize_size,
    sort_boxes_reading_order,
    to_square_canvas,
)
from .reports import BBox, OperationReport


MAX_RECURSIVE_SPLIT_DEPTH = 4


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
    image = load_image(input_path, "RGBA")
    out_dir = ensure_output_dir(output_dir)
    w, h = image.size

    mask = foreground_mask_from_background(image, threshold=threshold)
    mask = clean_mask(mask, open_size=3, close_size=7, fill_holes=True)

    raw_components = components_bboxes(mask, min_area=min_area, min_size=3)
    initial_boxes = [box for box, _area in raw_components]
    merged_boxes = merge_close_boxes(initial_boxes, w, h, distance=merge_distance)
    merged_boxes = [b for b in merged_boxes if b.area >= min_area]

    # Equipment sheets often contain upper/lower garment views or boot/shoe pairs
    # inside a single coarse component crop. Split such large crops again using
    # internal whitespace projection before saving final items.
    boxes, split_count = _split_boxes_by_internal_whitespace(
        mask=mask,
        boxes=merged_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )
    boxes = sort_boxes_reading_order(boxes)

    warnings: list[str] = []
    if not boxes:
        warnings.append("No foreground items were detected.")
    if len(boxes) > 200:
        warnings.append("Very many components were detected; increase min_area or merge_distance.")

    outputs: list[str] = []
    for idx, box in enumerate(boxes, start=1):
        crop = crop_with_padding(image, box, padding=padding)
        if transparent_bg:
            local_mask = crop_mask(mask, box, padding=padding)
            # crop_with_padding may include border fill if the box touches the image edge.
            if local_mask.shape[1] == crop.width and local_mask.shape[0] == crop.height:
                crop = apply_mask_as_alpha(crop, local_mask)
            else:
                # Edge-clipped case: keep original alpha rather than risk misalignment.
                crop = crop.convert("RGBA")

        if square_canvas:
            crop = to_square_canvas(crop)
        if normalize:
            crop = normalize_size(crop, normalize)

        path = out_dir / f"item_{idx:03d}.png"
        crop.save(path)
        outputs.append(str(path))

    manifest = OperationReport(
        ok=len(warnings) == 0,
        operation="extract-items",
        source=str(input_path),
        mode="component-recursive-whitespace",
        warnings=warnings,
        metrics={
            "items": len(boxes),
            "initial_components": len(initial_boxes),
            "merged_components": len(merged_boxes),
            "recursive_splits": split_count,
            "image_width": w,
            "image_height": h,
            "min_area": min_area,
            "merge_distance": merge_distance,
        },
        boxes=boxes,
        outputs=outputs,
    )
    manifest.save(out_dir / "manifest.json")

    if debug:
        save_mask(mask, out_dir / "debug_mask.png")
        save_boxes_overlay(image, boxes, out_dir / "debug_boxes.png")
        save_boxes_overlay(image, merged_boxes, out_dir / "debug_boxes_before_recursive_split.png")

    return manifest


def _split_boxes_by_internal_whitespace(
    mask: np.ndarray,
    boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> tuple[list[BBox], int]:
    final: list[BBox] = []
    split_count = 0
    for box in boxes:
        pieces, splits = _split_box_recursive(
            mask=mask,
            box=_trim_box_to_foreground(mask, box, image_width, image_height) or box,
            image_width=image_width,
            image_height=image_height,
            min_area=min_area,
            depth=0,
        )
        final.extend(pieces)
        split_count += splits
    return _dedupe_boxes(final), split_count


def _split_box_recursive(
    mask: np.ndarray,
    box: BBox,
    image_width: int,
    image_height: int,
    min_area: int,
    depth: int,
) -> tuple[list[BBox], int]:
    box = _trim_box_to_foreground(mask, box, image_width, image_height) or box
    if depth >= MAX_RECURSIVE_SPLIT_DEPTH:
        return [box], 0
    if box.area < max(min_area * 4, 1024):
        return [box], 0

    local = mask[box.top : box.bottom, box.left : box.right].astype(bool)
    if not local.any():
        return [], 0

    candidate = _best_projection_gap(local, box)
    if candidate is None:
        return [box], 0

    axis, start, end = candidate
    child_boxes = _boxes_from_split(mask, box, axis, start, end, image_width, image_height, min_area)
    if len(child_boxes) < 2:
        return [box], 0

    out: list[BBox] = []
    split_count = 1
    for child in child_boxes:
        pieces, splits = _split_box_recursive(
            mask=mask,
            box=child,
            image_width=image_width,
            image_height=image_height,
            min_area=min_area,
            depth=depth + 1,
        )
        out.extend(pieces)
        split_count += splits
    return out, split_count


def _best_projection_gap(local: np.ndarray, box: BBox) -> tuple[str, int, int] | None:
    h, w = local.shape[:2]
    if h < 24 or w < 24:
        return None

    horizontal = _best_gap_for_projection(
        counts=local.sum(axis=1),
        cross_size=w,
        length=h,
        axis="h",
    )
    vertical = _best_gap_for_projection(
        counts=local.sum(axis=0),
        cross_size=h,
        length=w,
        axis="v",
    )

    candidates = []
    for item in (horizontal, vertical):
        if item is None:
            continue
        axis, start, end, score = item
        if _split_has_enough_foreground(local, axis, start, end, box):
            candidates.append((axis, start, end, score))

    if not candidates:
        return None
    axis, start, end, _score = max(candidates, key=lambda x: x[3])
    return axis, start, end


def _best_gap_for_projection(
    counts: np.ndarray,
    cross_size: int,
    length: int,
    axis: str,
) -> tuple[str, int, int, float] | None:
    low_threshold = max(1, int(cross_size * 0.006))
    emptyish = counts <= low_threshold
    margin = max(4, int(length * 0.04))
    min_gap = max(5, int(length * 0.012))

    best: tuple[str, int, int, float] | None = None
    start: int | None = None
    for idx, is_empty in enumerate(emptyish.tolist() + [False]):
        if is_empty and start is None:
            start = idx
        elif not is_empty and start is not None:
            end = idx
            start_i = start
            start = None
            if start_i < margin or end > length - margin:
                continue
            gap_len = end - start_i
            if gap_len < min_gap:
                continue
            # Longer internal whitespace bands are more reliable. Gaps near the
            # middle are slightly preferred because equipment sheets usually use
            # whitespace between views or upper/lower parts.
            center = (start_i + end) / 2.0
            middle_bonus = 1.0 - min(1.0, abs(center - length / 2.0) / max(1.0, length / 2.0)) * 0.15
            score = (gap_len / max(1, length)) * middle_bonus
            if best is None or score > best[3]:
                best = (axis, start_i, end, score)
    return best


def _split_has_enough_foreground(local: np.ndarray, axis: str, start: int, end: int, box: BBox) -> bool:
    if axis == "h":
        a = local[:start, :]
        b = local[end:, :]
    else:
        a = local[:, :start]
        b = local[:, end:]
    min_pixels = max(24, int(local.size * 0.006))
    if int(a.sum()) < min_pixels or int(b.sum()) < min_pixels:
        return False

    # Avoid splitting a single skinny object due to a small decorative notch.
    if axis == "h":
        return box.height >= 40
    return box.width >= 40


def _boxes_from_split(
    mask: np.ndarray,
    box: BBox,
    axis: str,
    start: int,
    end: int,
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[BBox]:
    if axis == "h":
        slabs = [
            BBox(box.left, box.top, box.right, box.top + start),
            BBox(box.left, box.top + end, box.right, box.bottom),
        ]
    else:
        slabs = [
            BBox(box.left, box.top, box.left + start, box.bottom),
            BBox(box.left + end, box.top, box.right, box.bottom),
        ]

    out: list[BBox] = []
    for slab in slabs:
        trimmed = _trim_box_to_foreground(mask, slab, image_width, image_height)
        if trimmed is None:
            continue
        if trimmed.area < min_area:
            continue
        if trimmed.width < 4 or trimmed.height < 4:
            continue
        out.append(trimmed)
    return out


def _trim_box_to_foreground(mask: np.ndarray, box: BBox, image_width: int, image_height: int) -> BBox | None:
    left = max(0, min(image_width, box.left))
    right = max(0, min(image_width, box.right))
    top = max(0, min(image_height, box.top))
    bottom = max(0, min(image_height, box.bottom))
    if right <= left or bottom <= top:
        return None
    local = mask[top:bottom, left:right].astype(bool)
    if not local.any():
        return None
    ys, xs = np.where(local)
    return BBox(
        left + int(xs.min()),
        top + int(ys.min()),
        left + int(xs.max()) + 1,
        top + int(ys.max()) + 1,
    )


def _dedupe_boxes(boxes: list[BBox]) -> list[BBox]:
    result: list[BBox] = []
    for box in sort_boxes_reading_order(boxes):
        if any(_box_iou(box, existing) > 0.90 for existing in result):
            continue
        result.append(box)
    return result


def _box_iou(a: BBox, b: BBox) -> float:
    x1 = max(a.left, b.left)
    y1 = max(a.top, b.top)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, a.area + b.area - inter)
    return float(inter / union)
