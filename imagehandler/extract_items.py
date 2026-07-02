from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

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
SAM_MODEL_REL = Path("models") / "sam_vit_b_01ec64.pth"


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
    image_area = max(1, w * h)

    mask = foreground_mask_from_background(image, threshold=threshold)
    mask = clean_mask(mask, open_size=3, close_size=7, fill_holes=True)
    source_fg = int(mask.sum())

    raw_components = components_bboxes(mask, min_area=min_area, min_size=3)
    initial_boxes = [box for box, _area in raw_components]

    seed_mask = _make_seed_mask(mask, w, h)
    seed_boxes = _seed_boxes(seed_mask, min_area=min_area, image_area=image_area)
    seed_boxes = _expand_seed_boxes_against_original_mask(
        mask=mask,
        seed_boxes=seed_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )

    watershed_boxes = _watershed_boxes(mask, min_area=min_area, image_width=w, image_height=h)
    sam_boxes, sam_info = _sam_proposal_boxes(image, mask, min_area=min_area, image_width=w, image_height=h)

    conservative_merge_distance = max(2, min(merge_distance, 8))
    candidate_boxes = _dedupe_boxes([*seed_boxes, *watershed_boxes, *sam_boxes])
    candidate_boxes = _merge_seed_boxes_safely(
        candidate_boxes,
        image_width=w,
        image_height=h,
        merge_distance=conservative_merge_distance,
    )

    small_independent_boxes = _independent_small_original_boxes(
        raw_boxes=initial_boxes,
        seed_boxes=candidate_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )
    candidate_boxes = _dedupe_boxes([*candidate_boxes, *small_independent_boxes])

    candidate_boxes, split_count = _split_or_reject_broad_boxes(
        mask=mask,
        boxes=candidate_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )

    candidate_boxes, recovery_count = _recover_missing_foreground(
        mask=mask,
        boxes=candidate_boxes,
        image_width=w,
        image_height=h,
        min_area=min_area,
    )
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
    if sam_info.get("status") == "missing_model":
        warnings.append("SAM proposal skipped; model file not found at models/sam_vit_b_01ec64.pth.")
    elif sam_info.get("status") == "missing_dependency":
        warnings.append("SAM proposal skipped; segment-anything is not installed.")

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

    manifest = OperationReport(
        ok=len(warnings) == 0,
        operation="extract-items",
        source=str(input_path),
        mode="sam-watershed-seed-coverage",
        warnings=warnings,
        metrics={
            "items": len(candidate_boxes),
            "initial_components": len(initial_boxes),
            "seed_components": len(seed_boxes),
            "watershed_components": len(watershed_boxes),
            "sam_components": len(sam_boxes),
            "sam_status": sam_info.get("status"),
            "recursive_splits": split_count,
            "coverage_recovered_components": recovery_count,
            "image_width": w,
            "image_height": h,
            "min_area": min_area,
            "merge_distance": merge_distance,
            **coverage,
        },
        boxes=candidate_boxes,
        outputs=outputs,
    )
    manifest.save(out_dir / "manifest.json")

    if debug:
        save_mask(mask, out_dir / "debug_mask.png")
        save_mask(seed_mask, out_dir / "debug_seed_mask.png")
        save_boxes_overlay(image, seed_boxes, out_dir / "debug_seed_boxes.png")
        save_boxes_overlay(image, watershed_boxes, out_dir / "debug_watershed_boxes.png")
        if sam_boxes:
            save_boxes_overlay(image, sam_boxes, out_dir / "debug_sam_boxes.png")
        save_boxes_overlay(image, candidate_boxes, out_dir / "debug_boxes.png")
        _save_coverage_debug(mask, candidate_boxes, out_dir / "debug_coverage_mask.png")

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
    seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return seed.astype(bool)


def _seed_boxes(seed_mask: np.ndarray, min_area: int, image_area: int) -> list[BBox]:
    area_threshold = max(min_area * 2, int(image_area * 0.00045), 64)
    raw = components_bboxes(seed_mask, min_area=area_threshold, min_size=6)
    return sort_boxes_reading_order([box for box, _area in raw])


def _watershed_boxes(mask: np.ndarray, min_area: int, image_width: int, image_height: int) -> list[BBox]:
    binary = mask.astype(bool)
    if not binary.any():
        return []
    dist = ndi.distance_transform_edt(binary)
    if float(dist.max()) <= 0:
        return []

    min_distance = max(6, int(min(image_width, image_height) * 0.018))
    coords = peak_local_max(dist, labels=binary, min_distance=min_distance, exclude_border=False)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx
    if markers.max() == 0:
        return []
    markers = ndi.label(markers > 0)[0]
    labels = watershed(-dist, markers, mask=binary)

    boxes: list[BBox] = []
    min_pixels = max(min_area, int(image_width * image_height * 0.00035))
    for label in range(1, int(labels.max()) + 1):
        component = labels == label
        if int(component.sum()) < min_pixels:
            continue
        ys, xs = np.where(component)
        box = BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        trimmed = _trim_box_to_foreground(mask, box, image_width, image_height)
        if trimmed is not None and trimmed.area >= min_area and not _is_extremely_broad(trimmed, image_width, image_height):
            boxes.append(trimmed)
    return _dedupe_boxes(boxes)


def _sam_proposal_boxes(
    image: Image.Image,
    mask: np.ndarray,
    min_area: int,
    image_width: int,
    image_height: int,
) -> tuple[list[BBox], dict[str, Any]]:
    model_path = Path.cwd() / SAM_MODEL_REL
    if not model_path.is_file():
        return [], {"status": "missing_model"}
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore
    except Exception:
        return [], {"status": "missing_dependency"}
    try:
        import torch  # type: ignore

        sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=24,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.86,
            min_mask_region_area=max(min_area, 64),
        )
        proposals = generator.generate(np.asarray(image.convert("RGB")))
    except Exception as exc:
        return [], {"status": "runtime_error", "error": str(exc)}

    boxes: list[BBox] = []
    source_fg = mask.astype(bool)
    for item in proposals:
        seg = np.asarray(item.get("segmentation"), dtype=bool)
        if seg.shape != source_fg.shape:
            continue
        fg_overlap = seg & source_fg
        area = int(fg_overlap.sum())
        if area < min_area:
            continue
        ys, xs = np.where(fg_overlap)
        box = BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if _is_extremely_broad(box, image_width, image_height):
            continue
        boxes.append(box)
    return _dedupe_boxes(boxes), {"status": "ok", "raw_masks": len(proposals)}


def _expand_seed_boxes_against_original_mask(mask: np.ndarray, seed_boxes: list[BBox], image_width: int, image_height: int, min_area: int) -> list[BBox]:
    out: list[BBox] = []
    for box in seed_boxes:
        expand_x = max(8, int(box.width * 0.14), int(min(image_width, image_height) * 0.006))
        expand_y = max(8, int(box.height * 0.14), int(min(image_width, image_height) * 0.006))
        candidate = BBox(max(0, box.left - expand_x), max(0, box.top - expand_y), min(image_width, box.right + expand_x), min(image_height, box.bottom + expand_y))
        trimmed = _trim_box_to_foreground(mask, candidate, image_width, image_height)
        if trimmed is None or trimmed.area < min_area:
            continue
        if _is_extremely_broad(trimmed, image_width, image_height):
            continue
        out.append(trimmed)
    return _dedupe_boxes(out)


def _merge_seed_boxes_safely(boxes: list[BBox], image_width: int, image_height: int, merge_distance: int) -> list[BBox]:
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


def _independent_small_original_boxes(raw_boxes: list[BBox], seed_boxes: list[BBox], image_width: int, image_height: int, min_area: int) -> list[BBox]:
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


def _recover_missing_foreground(mask: np.ndarray, boxes: list[BBox], image_width: int, image_height: int, min_area: int) -> tuple[list[BBox], int]:
    covered = _coverage_mask(mask, boxes)
    missing = mask.astype(bool) & ~covered
    if int(missing.sum()) < max(min_area, int(mask.sum() * 0.03)):
        return boxes, 0

    missing = clean_mask(missing, open_size=3, close_size=5, fill_holes=False)
    recovered_raw = components_bboxes(missing, min_area=max(min_area, int(mask.sum() * 0.01)), min_size=6)
    recovered: list[BBox] = []
    for box, _area in recovered_raw:
        if _is_extremely_broad(box, image_width, image_height):
            pieces, _splits = _split_box_recursive(mask, box, image_width, image_height, min_area, depth=0)
            recovered.extend(pieces if len(pieces) > 1 else [box])
        else:
            recovered.append(box)
    return _dedupe_boxes([*boxes, *recovered]), len(recovered)


def _coverage_metrics(mask: np.ndarray, boxes: list[BBox]) -> dict[str, float | int]:
    source = int(mask.sum())
    covered = _coverage_mask(mask, boxes)
    covered_count = int((covered & mask.astype(bool)).sum())
    sum_item = 0
    for box in boxes:
        sum_item += int(mask[box.top : box.bottom, box.left : box.right].sum())
    missing = max(0, source - covered_count)
    return {
        "source_foreground_pixels": source,
        "covered_foreground_pixels": covered_count,
        "missing_foreground_pixels": missing,
        "coverage_ratio": float(covered_count / max(1, source)),
        "missing_foreground_ratio": float(missing / max(1, source)),
        "sum_item_foreground_pixels": sum_item,
        "duplication_ratio": float(sum_item / max(1, covered_count)),
    }


def _coverage_mask(mask: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    covered = np.zeros(mask.shape, dtype=bool)
    for box in boxes:
        covered[box.top : box.bottom, box.left : box.right] |= mask[box.top : box.bottom, box.left : box.right].astype(bool)
    return covered


def _save_coverage_debug(mask: np.ndarray, boxes: list[BBox], path: Path) -> None:
    covered = _coverage_mask(mask, boxes)
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask.astype(bool)] = (80, 80, 80)
    img[covered & mask.astype(bool)] = (255, 255, 255)
    Image.fromarray(img, mode="RGB").save(path)


def _split_or_reject_broad_boxes(mask: np.ndarray, boxes: list[BBox], image_width: int, image_height: int, min_area: int) -> tuple[list[BBox], int]:
    out: list[BBox] = []
    split_count = 0
    for box in boxes:
        if _is_broad_candidate(box, image_width, image_height):
            pieces, splits = _split_box_recursive(mask, box, image_width, image_height, min_area, depth=0)
            if splits > 0 and len(pieces) > 1:
                out.extend(pieces)
                split_count += splits
                continue
        out.append(box)
    return out, split_count


def _is_broad_candidate(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return box.area / image_area > 0.18 or box.width / max(1, image_width) > 0.60 or box.height / max(1, image_height) > 0.60


def _is_extremely_broad(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return (
        box.area / image_area > 0.42
        or (box.width / max(1, image_width) > 0.78 and box.height / max(1, image_height) > 0.42)
        or (box.height / max(1, image_height) > 0.78 and box.width / max(1, image_width) > 0.42)
    )


def _is_too_broad_to_merge(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return box.area / image_area > 0.24 or box.width / max(1, image_width) > 0.68 or box.height / max(1, image_height) > 0.72


def _split_box_recursive(mask: np.ndarray, box: BBox, image_width: int, image_height: int, min_area: int, depth: int) -> tuple[list[BBox], int]:
    box = _trim_box_to_foreground(mask, box, image_width, image_height) or box
    if depth >= MAX_RECURSIVE_SPLIT_DEPTH or box.area < max(min_area * 4, 1024):
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
        pieces, splits = _split_box_recursive(mask, child, image_width, image_height, min_area, depth + 1)
        out.extend(pieces)
        split_count += splits
    return out, split_count


def _best_projection_gap(local: np.ndarray, box: BBox) -> tuple[str, int, int] | None:
    h, w = local.shape[:2]
    if h < 24 or w < 24:
        return None
    horizontal = _best_gap_for_projection(local.sum(axis=1), w, h, "h")
    vertical = _best_gap_for_projection(local.sum(axis=0), h, w, "v")
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


def _best_gap_for_projection(counts: np.ndarray, cross_size: int, length: int, axis: str) -> tuple[str, int, int, float] | None:
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
    return int(a.sum()) >= min_pixels and int(b.sum()) >= min_pixels and (box.height >= 40 if axis == "h" else box.width >= 40)


def _boxes_from_split(mask: np.ndarray, box: BBox, axis: str, start: int, end: int, image_width: int, image_height: int, min_area: int) -> list[BBox]:
    if axis == "h":
        slabs = [BBox(box.left, box.top, box.right, box.top + start), BBox(box.left, box.top + end, box.right, box.bottom)]
    else:
        slabs = [BBox(box.left, box.top, box.left + start, box.bottom), BBox(box.left + end, box.top, box.right, box.bottom)]
    out: list[BBox] = []
    for slab in slabs:
        trimmed = _trim_box_to_foreground(mask, slab, image_width, image_height)
        if trimmed is not None and trimmed.area >= min_area and trimmed.width >= 4 and trimmed.height >= 4:
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
    return BBox(left + int(xs.min()), top + int(ys.min()), left + int(xs.max()) + 1, top + int(ys.max()) + 1)


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
