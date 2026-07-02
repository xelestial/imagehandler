from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .debug import save_boxes_overlay, save_mask
from .extract_items import (
    _coverage_metrics,
    _dedupe_boxes,
    _independent_small_original_boxes,
    _is_too_broad_to_merge,
    _make_seed_mask,
    _merge_seed_boxes_safely,
    _recover_missing_foreground,
    _sam_proposal_boxes,
    _save_coverage_debug,
    _seed_boxes,
    _split_or_reject_broad_boxes,
    _watershed_boxes,
)
from .io import ensure_output_dir, load_image
from .mask_ops import (
    apply_mask_as_alpha,
    clean_mask,
    components_bboxes,
    crop_mask,
    crop_with_padding,
    foreground_mask_from_background,
    normalize_size,
    sort_boxes_reading_order,
    to_square_canvas,
)
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
    image = load_image(input_path, "RGBA")
    out_dir = ensure_output_dir(output_dir)
    width, height = image.size
    image_area = max(1, width * height)

    mask = foreground_mask_from_background(image, threshold=threshold)
    mask = clean_mask(mask, open_size=3, close_size=7, fill_holes=True)
    source_fg = int(mask.sum())

    raw_components = components_bboxes(mask, min_area=min_area, min_size=3)
    initial_boxes = [box for box, _area in raw_components]

    seed_mask = _make_seed_mask(mask, width, height)
    seed_core_boxes = _seed_boxes(seed_mask, min_area=min_area, image_area=image_area)
    seed_grown_boxes = _expand_seed_boxes_with_region_growth(
        mask=mask,
        seed_mask=seed_mask,
        seed_boxes=seed_core_boxes,
        image_width=width,
        image_height=height,
        min_area=min_area,
    )

    watershed_boxes = _watershed_boxes(mask, min_area=min_area, image_width=width, image_height=height)
    sam_boxes, sam_info = _sam_proposal_boxes(image, mask, min_area=min_area, image_width=width, image_height=height)

    candidate_boxes = _dedupe_boxes([*seed_grown_boxes, *watershed_boxes, *sam_boxes])
    candidate_boxes = _merge_seed_boxes_safely(
        candidate_boxes,
        image_width=width,
        image_height=height,
        merge_distance=max(2, min(merge_distance, 8)),
    )
    candidate_boxes = _dedupe_boxes(
        [
            *candidate_boxes,
            *_independent_small_original_boxes(
                raw_boxes=initial_boxes,
                seed_boxes=candidate_boxes,
                image_width=width,
                image_height=height,
                min_area=min_area,
            ),
        ]
    )
    candidate_boxes, split_count = _split_or_reject_broad_boxes(mask, candidate_boxes, width, height, min_area)
    candidate_boxes, recovery_count = _recover_missing_foreground(mask, candidate_boxes, width, height, min_area)
    candidate_boxes = sort_boxes_reading_order(_dedupe_boxes(candidate_boxes))

    coverage = _coverage_metrics(mask, candidate_boxes)
    warnings: list[str] = []
    if not candidate_boxes:
        warnings.append("No foreground items were detected.")
    if len(candidate_boxes) > 200:
        warnings.append("Very many components were detected; increase min_area or merge_distance.")
    if source_fg > 0 and coverage["coverage_ratio"] < 0.80:
        warnings.append("Low item coverage; many source foreground pixels were not covered by extracted crops.")
    if coverage["duplication_ratio"] > 1.60:
        warnings.append("High item duplication; extracted crops overlap heavily.")

    outputs: list[str] = []
    for idx, box in enumerate(candidate_boxes, start=1):
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

    report = OperationReport(
        ok=len(warnings) == 0,
        operation="extract-items",
        source=str(input_path),
        mode="sam-watershed-seed-region-grow-coverage",
        warnings=warnings,
        metrics={
            "items": len(candidate_boxes),
            "initial_components": len(initial_boxes),
            "seed_core_components": len(seed_core_boxes),
            "seed_grown_components": len(seed_grown_boxes),
            "watershed_components": len(watershed_boxes),
            "sam_components": len(sam_boxes),
            "sam_status": sam_info.get("status"),
            "recursive_splits": split_count,
            "coverage_recovered_components": recovery_count,
            "image_width": width,
            "image_height": height,
            "min_area": min_area,
            "merge_distance": merge_distance,
            **coverage,
        },
        boxes=candidate_boxes,
        outputs=outputs,
    )
    report.save(out_dir / "manifest.json")

    if debug:
        save_mask(mask, out_dir / "debug_mask.png")
        save_mask(seed_mask, out_dir / "debug_seed_mask.png")
        save_boxes_overlay(image, seed_core_boxes, out_dir / "debug_seed_core_boxes.png")
        save_boxes_overlay(image, seed_grown_boxes, out_dir / "debug_seed_grown_boxes.png")
        save_boxes_overlay(image, watershed_boxes, out_dir / "debug_watershed_boxes.png")
        if sam_boxes:
            save_boxes_overlay(image, sam_boxes, out_dir / "debug_sam_boxes.png")
        save_boxes_overlay(image, candidate_boxes, out_dir / "debug_boxes.png")
        _save_coverage_debug(mask, candidate_boxes, out_dir / "debug_coverage_mask.png")

    return report


def _expand_seed_boxes_with_region_growth(
    mask: np.ndarray,
    seed_mask: np.ndarray,
    seed_boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[BBox]:
    boxes: list[BBox] = []
    for seed_box in seed_boxes:
        box = _grow_one_seed(mask, seed_mask, seed_box, image_width, image_height, min_area)
        if box is not None:
            boxes.append(box)
    return _dedupe_boxes(boxes)


def _grow_one_seed(
    mask: np.ndarray,
    seed_mask: np.ndarray,
    seed_box: BBox,
    image_width: int,
    image_height: int,
    min_area: int,
) -> BBox | None:
    expand_x, expand_y, iterations = _growth_parameters(seed_box, image_width, image_height)
    region_box = BBox(
        max(0, seed_box.left - expand_x),
        max(0, seed_box.top - expand_y),
        min(image_width, seed_box.right + expand_x),
        min(image_height, seed_box.bottom + expand_y),
    )
    fg = mask[region_box.top : region_box.bottom, region_box.left : region_box.right].astype(bool)
    if not fg.any():
        return None

    seed = np.zeros_like(fg, dtype=bool)
    x1 = seed_box.left - region_box.left
    y1 = seed_box.top - region_box.top
    x2 = x1 + seed_box.width
    y2 = y1 + seed_box.height
    seed_part = seed_mask[seed_box.top : seed_box.bottom, seed_box.left : seed_box.right]
    seed[y1:y2, x1:x2] = seed_part
    if not seed.any():
        seed[y1:y2, x1:x2] = fg[y1:y2, x1:x2]

    grown = seed & fg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(iterations):
        nxt = cv2.dilate(grown.astype(np.uint8), kernel, iterations=1).astype(bool) & fg
        if int(nxt.sum()) == int(grown.sum()):
            break
        candidate = _box_from_local_mask(nxt, region_box)
        if candidate is not None and _is_too_broad_to_merge(candidate, image_width, image_height):
            break
        grown = nxt

    out = _box_from_local_mask(grown, region_box)
    if out is None or out.area < min_area:
        return None
    return out


def _growth_parameters(box: BBox, image_width: int, image_height: int) -> tuple[int, int, int]:
    base = min(image_width, image_height)
    aspect = box.height / max(1, box.width)
    inv_aspect = box.width / max(1, box.height)
    if aspect > 2.0:
        expand_x = max(int(box.width * 2.4), int(base * 0.09), 24)
        expand_y = max(int(box.height * 0.20), int(base * 0.025), 16)
    elif inv_aspect > 2.0:
        expand_x = max(int(box.width * 0.20), int(base * 0.025), 16)
        expand_y = max(int(box.height * 2.2), int(base * 0.07), 20)
    else:
        expand_x = max(int(box.width * 0.55), int(base * 0.04), 16)
        expand_y = max(int(box.height * 0.55), int(base * 0.04), 16)
    iterations = max(16, min(90, int(max(expand_x, expand_y) * 0.75)))
    return expand_x, expand_y, iterations


def _box_from_local_mask(local: np.ndarray, region_box: BBox) -> BBox | None:
    if not local.any():
        return None
    ys, xs = np.where(local)
    return BBox(
        region_box.left + int(xs.min()),
        region_box.top + int(ys.min()),
        region_box.left + int(xs.max()) + 1,
        region_box.top + int(ys.max()) + 1,
    )


def _split_box_recursive(
    mask: np.ndarray,
    box: BBox,
    image_width: int,
    image_height: int,
    min_area: int,
    depth: int = 0,
) -> tuple[list[BBox], int]:
    if depth >= 4 or box.area < max(min_area * 4, 1024):
        return [box], 0
    local = mask[box.top : box.bottom, box.left : box.right].astype(bool)
    if not local.any():
        return [], 0
    candidate = _best_gap(local)
    if candidate is None:
        return [box], 0
    axis, start, end = candidate
    if axis == "h":
        slabs = [BBox(box.left, box.top, box.right, box.top + start), BBox(box.left, box.top + end, box.right, box.bottom)]
    else:
        slabs = [BBox(box.left, box.top, box.left + start, box.bottom), BBox(box.left + end, box.top, box.right, box.bottom)]
    pieces: list[BBox] = []
    for slab in slabs:
        sub = mask[slab.top : slab.bottom, slab.left : slab.right].astype(bool)
        if int(sub.sum()) < min_area:
            continue
        ys, xs = np.where(sub)
        trimmed = BBox(slab.left + int(xs.min()), slab.top + int(ys.min()), slab.left + int(xs.max()) + 1, slab.top + int(ys.max()) + 1)
        parts, count = _split_box_recursive(mask, trimmed, image_width, image_height, min_area, depth + 1)
        pieces.extend(parts)
    if len(pieces) < 2:
        return [box], 0
    return pieces, 1


def _best_gap(local: np.ndarray) -> tuple[str, int, int] | None:
    h, w = local.shape
    candidates: list[tuple[str, int, int, float]] = []
    for axis, counts, cross, length in (("h", local.sum(axis=1), w, h), ("v", local.sum(axis=0), h, w)):
        low = max(1, int(cross * 0.01))
        empty = counts <= low
        start = None
        margin = max(4, int(length * 0.03))
        min_gap = max(4, int(length * 0.01))
        for idx, val in enumerate(empty.tolist() + [False]):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                end = idx
                if start >= margin and end <= length - margin and end - start >= min_gap:
                    candidates.append((axis, start, end, float(end - start) / max(1, length)))
                start = None
    if not candidates:
        return None
    axis, start, end, _score = max(candidates, key=lambda item: item[3])
    return axis, start, end
