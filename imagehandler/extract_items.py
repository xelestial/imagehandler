from __future__ import annotations

from pathlib import Path

import cv2
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

    # Equipment sheets are frequently connected by thin straps, transparent cloth,
    # decorative lines, or shadows. Do not start from one broad merged component.
    # First create a seed mask where thin bridges are weakened, then expand each
    # seed locally against the original mask.
    seed_mask = _make_seed_mask(mask, w, h)
    seed_boxes = _seed_boxes(seed_mask, min_area=min_area, image_area=w * h)
    seed_expanded_boxes = _expand_seed_boxes_against_original_mask(
        mask=mask,
        seed_boxes=seed_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )
    seed_expanded_boxes = _merge_seed_boxes_safely(
        seed_expanded_boxes,
        image_width=w,
        image_height=h,
        merge_distance=max(2, min(merge_distance, 8)),
    )

    small_independent_boxes = _independent_small_original_boxes(
        raw_boxes=initial_boxes,
        seed_boxes=seed_expanded_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )

    if seed_expanded_boxes:
        candidate_boxes = _dedupe_boxes([*seed_expanded_boxes, *small_independent_boxes])
        mode = "seed-local-expansion"
        split_count = 0
    else:
        merged_boxes = merge_close_boxes(initial_boxes, w, h, distance=max(2, min(merge_distance, 6)))
        merged_boxes = [b for b in merged_boxes if b.area >= min_area]
        candidate_boxes, split_count = _split_boxes_by_internal_whitespace(
            mask=mask,
            boxes=merged_boxes,
            image_width=w,
            image_height=h,
            min_area=min_area,
        )
        mode = "component-recursive-whitespace-fallback"

    # As a final guard, broad sheet-region boxes should not survive as final
    # items when there are smaller alternatives. Try projection splitting first;
    # keep the broad box only when no safe split exists.
    boxes, final_splits = _split_or_reject_broad_boxes(
        mask=mask,
        boxes=candidate_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )
    split_count += final_splits
    boxes = sort_boxes_reading_order(_dedupe_boxes(boxes))

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
            if local_mask.shape[1] == crop.width and local_mask.shape[0] == crop.height:
                crop = apply_mask_as_alpha(crop, local_mask)
            else:
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
        mode=mode,
        warnings=warnings,
        metrics={
            "items": len(boxes),
            "initial_components": len(initial_boxes),
            "seed_components": len(seed_boxes),
            "seed_expanded_components": len(seed_expanded_boxes),
            "independent_small_components": len(small_independent_boxes),
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
        save_mask(seed_mask, out_dir / "debug_seed_mask.png")
        save_boxes_overlay(image, seed_boxes, out_dir / "debug_seed_boxes.png")
        save_boxes_overlay(image, seed_expanded_boxes, out_dir / "debug_seed_expanded_boxes.png")
        save_boxes_overlay(image, boxes, out_dir / "debug_boxes.png")

    return manifest


def _make_seed_mask(mask: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    base = max(3, int(min(image_width, image_height) * 0.004))
    if base % 2 == 0:
        base += 1
    base = min(max(base, 3), 11)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base, base))

    seed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    erode_size = max(3, base - 2)
    if erode_size % 2 == 0:
        erode_size += 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    seed = cv2.erode(seed, erode_kernel, iterations=1)

    # Add a small dilation back so seeds cover the core object but do not restore
    # long thin bridges that caused broad component grouping.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.dilate(seed, dilate_kernel, iterations=1)
    return seed.astype(bool)


def _seed_boxes(seed_mask: np.ndarray, min_area: int, image_area: int) -> list[BBox]:
    area_threshold = max(min_area * 2, int(image_area * 0.00045), 64)
    raw = components_bboxes(seed_mask, min_area=area_threshold, min_size=6)
    boxes = [box for box, _area in raw]
    return sort_boxes_reading_order(boxes)


def _expand_seed_boxes_against_original_mask(
    mask: np.ndarray,
    seed_boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[BBox]:
    out: list[BBox] = []
    for box in seed_boxes:
        expand_x = max(8, int(box.width * 0.12), int(min(image_width, image_height) * 0.006))
        expand_y = max(8, int(box.height * 0.12), int(min(image_width, image_height) * 0.006))
        candidate = BBox(
            max(0, box.left - expand_x),
            max(0, box.top - expand_y),
            min(image_width, box.right + expand_x),
            min(image_height, box.bottom + expand_y),
        )
        trimmed = _trim_box_to_foreground(mask, candidate, image_width, image_height)
        if trimmed is None or trimmed.area < min_area:
            continue
        if _is_extremely_broad(trimmed, image_width, image_height):
            # Do not accept a sheet-sized region as one item.
            continue
        out.append(trimmed)
    return _dedupe_boxes(out)


def _merge_seed_boxes_safely(
    boxes: list[BBox],
    image_width: int,
    image_height: int,
    merge_distance: int,
) -> list[BBox]:
    result = sort_boxes_reading_order(boxes)
    changed = True
    while changed:
        changed = False
        merged: list[BBox] = []
        used = [False] * len(result)
        for i, box in enumerate(result):
            if used[i]:
                continue
            current = box
            used[i] = True
            for j in range(i + 1, len(result)):
                if used[j]:
                    continue
                other = result[j]
                union = current.union(other)
                if _is_too_broad_to_merge(union, image_width, image_height):
                    continue
                if _box_iou(current, other) > 0.08 or _box_distance(current, other) <= merge_distance:
                    current = union
                    used[j] = True
                    changed = True
            merged.append(current)
        result = sort_boxes_reading_order(merged)
    return result


def _independent_small_original_boxes(
    raw_boxes: list[BBox],
    seed_boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[BBox]:
    out: list[BBox] = []
    max_area = max(min_area * 40, int(image_width * image_height * 0.035))
    for box in raw_boxes:
        if box.area < min_area or box.area > max_area:
            continue
        if any(_box_iou(box, seed) > 0.15 or _contained_ratio(box, seed) > 0.65 for seed in seed_boxes):
            continue
        if _is_extremely_broad(box, image_width, image_height):
            continue
        out.append(box)
    return _dedupe_boxes(out)


def _split_or_reject_broad_boxes(
    mask: np.ndarray,
    boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> tuple[list[BBox], int]:
    out: list[BBox] = []
    split_count = 0
    for box in boxes:
        if _is_broad_candidate(box, image_width, image_height):
            pieces, splits = _split_box_recursive(
                mask=mask,
                box=box,
                image_width=image_width,
                image_height=image_height,
                min_area=min_area,
                depth=0,
            )
            if splits > 0 and len(pieces) > 1:
                out.extend(pieces)
                split_count += splits
                continue
        out.append(box)
    return out, split_count


def _is_broad_candidate(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return (
        box.area / image_area > 0.18
        or box.width / max(1, image_width) > 0.60
        or box.height / max(1, image_height) > 0.60
    )


def _is_extremely_broad(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return (
        box.area / image_area > 0.42
        or (box.width / max(1, image_width) > 0.78 and box.height / max(1, image_height) > 0.42)
        or (box.height / max(1, image_height) > 0.78 and box.width / max(1, image_width) > 0.42)
    )


def _is_too_broad_to_merge(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return (
        box.area / image_area > 0.24
        or box.width / max(1, image_width) > 0.68
        or box.height / max(1, image_height) > 0.72
    )


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
    low_threshold = max(1, int(cross_size * 0.010))
    emptyish = counts <= low_threshold
    margin = max(4, int(length * 0.03))
    min_gap = max(4, int(length * 0.010))

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
    min_pixels = max(24, int(local.size * 0.004))
    if int(a.sum()) < min_pixels or int(b.sum()) < min_pixels:
        return False
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
        if any(_box_iou(box, existing) > 0.88 or _mutual_containment(box, existing) > 0.88 for existing in result):
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


def _contained_ratio(a: BBox, b: BBox) -> float:
    x1 = max(a.left, b.left)
    y1 = max(a.top, b.top)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return float(inter / max(1, a.area))


def _mutual_containment(a: BBox, b: BBox) -> float:
    return max(_contained_ratio(a, b), _contained_ratio(b, a))


def _box_distance(a: BBox, b: BBox) -> float:
    dx = max(0, max(a.left, b.left) - min(a.right, b.right))
    dy = max(0, max(a.top, b.top) - min(a.bottom, b.bottom))
    return float((dx * dx + dy * dy) ** 0.5)
