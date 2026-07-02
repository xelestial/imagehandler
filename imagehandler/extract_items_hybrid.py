from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from .debug import save_boxes_overlay, save_mask
from .extract_items_plus import (
    _coverage_metrics,
    _dedupe_boxes,
    _expand_seed_boxes_with_region_growth,
    _independent_small_original_boxes,
    _merge_seed_boxes_safely,
    _recover_missing_foreground,
    _sam_proposal_boxes,
    _save_coverage_debug,
    _seed_boxes,
    _split_or_reject_broad_boxes,
    _watershed_boxes,
)
from .fallback import remove_background_with_fallback
from .io import ensure_output_dir, load_image
from .mask_ops import (
    clean_mask,
    components_bboxes,
    foreground_mask_from_background,
    normalize_size,
    pil_to_rgba_array,
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
    source = Path(input_path)
    out_dir = ensure_output_dir(output_dir)
    image = load_image(source, "RGBA")
    width, height = image.size
    image_area = max(1, width * height)

    support_mask, core_mask, preprocess_info = _build_support_and_core_masks(source, image, out_dir, threshold)
    support_mask = clean_mask(support_mask, open_size=3, close_size=7, fill_holes=True)
    core_mask = clean_mask(core_mask & support_mask, open_size=2, close_size=3, fill_holes=False)

    raw_components = components_bboxes(support_mask, min_area=min_area, min_size=3)
    initial_boxes = [box for box, _area in raw_components]

    seed_mask = _make_hybrid_seed_mask(core_mask, support_mask, width, height)
    seed_core_boxes = _seed_boxes(seed_mask, min_area=min_area, image_area=image_area)
    seed_grown_boxes = _expand_seed_boxes_with_region_growth(
        mask=support_mask,
        seed_mask=seed_mask,
        seed_boxes=seed_core_boxes,
        image_width=width,
        image_height=height,
        min_area=min_area,
    )

    watershed_boxes = _watershed_boxes(core_mask, min_area=min_area, image_width=width, image_height=height)
    sam_boxes, sam_info = _sam_proposal_boxes(image, support_mask, min_area=min_area, image_width=width, image_height=height)

    boxes = _dedupe_boxes([*seed_grown_boxes, *watershed_boxes, *sam_boxes])
    boxes = _merge_seed_boxes_safely(
        boxes,
        image_width=width,
        image_height=height,
        merge_distance=max(2, min(merge_distance, 8)),
    )
    boxes = _dedupe_boxes(
        [
            *boxes,
            *_independent_small_original_boxes(
                raw_boxes=initial_boxes,
                seed_boxes=boxes,
                image_width=width,
                image_height=height,
                min_area=min_area,
            ),
        ]
    )
    boxes, split_count = _split_or_reject_broad_boxes(support_mask, boxes, width, height, min_area)
    boxes, extra_splits = _split_stacked_and_row_boxes(support_mask, boxes, width, height, min_area)
    split_count += extra_splits
    boxes, recovery_count = _recover_missing_foreground(support_mask, boxes, width, height, min_area)
    boxes, final_splits, pruned_parent_boxes = _final_refine_after_recovery(
        support_mask,
        boxes,
        width,
        height,
        min_area,
    )
    split_count += final_splits
    boxes = sort_boxes_reading_order(_dedupe_boxes(boxes))

    item_masks = _assign_instance_masks(support_mask, boxes, width, height)
    mask_stats = _instance_mask_stats(support_mask, boxes, item_masks)

    coverage = _coverage_metrics(support_mask, boxes)
    warnings: list[str] = []
    if not boxes:
        warnings.append("No foreground items were detected.")
    if len(boxes) > 200:
        warnings.append("Very many components were detected; increase min_area or merge_distance.")
    if coverage["coverage_ratio"] < 0.80:
        warnings.append("Low item coverage; many source foreground pixels were not covered by extracted crops.")
    if coverage["duplication_ratio"] > 1.60:
        warnings.append("High item duplication; extracted crops overlap heavily.")

    outputs: list[str] = []
    for idx, (box, item_mask) in enumerate(zip(boxes, item_masks, strict=True), start=1):
        crop = _crop_with_instance_mask(
            image=image,
            box=box,
            item_mask=item_mask,
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

    report = OperationReport(
        ok=len(warnings) == 0,
        operation="extract-items",
        source=str(source),
        mode="hybrid-alpha-support-rgb-core-final-refine-instance-mask-crop",
        warnings=warnings,
        metrics={
            "items": len(boxes),
            "initial_components": len(initial_boxes),
            "seed_core_components": len(seed_core_boxes),
            "seed_grown_components": len(seed_grown_boxes),
            "watershed_components": len(watershed_boxes),
            "sam_components": len(sam_boxes),
            "sam_status": sam_info.get("status"),
            "recursive_splits": split_count,
            "coverage_recovered_components": recovery_count,
            "pruned_parent_boxes": pruned_parent_boxes,
            "instance_mask_assigned_pixels": mask_stats["assigned_pixels"],
            "instance_mask_unassigned_pixels": mask_stats["unassigned_pixels"],
            "instance_mask_empty_items": mask_stats["empty_items"],
            "image_width": width,
            "image_height": height,
            "min_area": min_area,
            "merge_distance": merge_distance,
            **preprocess_info,
            **coverage,
        },
        boxes=boxes,
        outputs=outputs,
    )
    report.save(out_dir / "manifest.json")

    if debug:
        save_mask(support_mask, out_dir / "debug_support_mask.png")
        save_mask(core_mask, out_dir / "debug_core_mask.png")
        save_mask(seed_mask, out_dir / "debug_seed_mask.png")
        save_boxes_overlay(image, seed_core_boxes, out_dir / "debug_seed_core_boxes.png")
        save_boxes_overlay(image, seed_grown_boxes, out_dir / "debug_seed_grown_boxes.png")
        save_boxes_overlay(image, watershed_boxes, out_dir / "debug_watershed_boxes.png")
        if sam_boxes:
            save_boxes_overlay(image, sam_boxes, out_dir / "debug_sam_boxes.png")
        save_boxes_overlay(image, boxes, out_dir / "debug_boxes.png")
        _save_coverage_debug(support_mask, boxes, out_dir / "debug_coverage_mask.png")
        _save_instance_debug(item_masks, out_dir / "debug_instance_assignment.png")

    return report


def _build_support_and_core_masks(source: Path, image: Image.Image, out_dir: Path, threshold: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rgba = pil_to_rgba_array(image)
    alpha = rgba[:, :, 3]
    opaque = Image.fromarray(np.dstack([rgba[:, :, :3], np.full(alpha.shape, 255, dtype=np.uint8)]), mode="RGBA")
    rgb_mask = foreground_mask_from_background(opaque, threshold=threshold)
    info: dict[str, object] = {"rgb_foreground_pixels": int(rgb_mask.sum())}

    if float((alpha < 250).mean()) > 0.005:
        support = alpha > 8
        info.update({
            "items_preprocess": "existing_alpha_support",
            "alpha_foreground_pixels": int(support.sum()),
            "alpha_core_pixels": int((alpha > 128).sum()),
        })
    else:
        support, bg_info = _support_mask_from_bg_pipeline(source, image, out_dir)
        info.update(bg_info)

    edge_mask = _detail_edge_mask(image, support)
    core = (rgb_mask | edge_mask | _alpha_core_from_support(support)) & support
    return support.astype(bool), core.astype(bool), info


def _support_mask_from_bg_pipeline(source: Path, image: Image.Image, out_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    info: dict[str, object] = {"items_preprocess": "rgb_threshold"}
    try:
        with TemporaryDirectory(prefix="imagehandler_items_hybrid_") as tmp:
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
            alpha = np.asarray(removed_img)[:, :, 3]
            support = alpha > 8
            original_arr = np.asarray(image).copy()
            original_arr[:, :, 3] = alpha
            Image.fromarray(alpha, mode="L").save(out_dir / "debug_preprocess_alpha.png")
            Image.fromarray(original_arr, mode="RGBA").save(out_dir / "debug_preprocess_rgba.png")
            info.update({
                "items_preprocess": "alpha_from_bg_fallback_pipeline_support_only",
                "items_preprocess_backend": report.backend or "auto",
                "items_preprocess_selected_attempt": summary.selected_attempt,
                "items_preprocess_selected_verdict": summary.selected_verdict,
                "items_preprocess_selected_score": float(summary.selected_score),
                "alpha_foreground_pixels": int(support.sum()),
                "alpha_core_pixels": int((alpha > 128).sum()),
            })
            return support.astype(bool), info
    except Exception as exc:
        info["items_preprocess_error"] = str(exc)
        fallback = foreground_mask_from_background(image)
        return fallback.astype(bool), info


def _detail_edge_mask(image: Image.Image, support: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0
    return edges & support.astype(bool)


def _alpha_core_from_support(support: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(support.astype(np.uint8), kernel, iterations=1).astype(bool)


def _make_hybrid_seed_mask(core_mask: np.ndarray, support_mask: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    base = core_mask.astype(np.uint8)
    k = max(3, int(min(image_width, image_height) * 0.003))
    if k % 2 == 0:
        k += 1
    k = min(max(k, 3), 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    seed = cv2.morphologyEx(base, cv2.MORPH_OPEN, kernel)
    seed = cv2.erode(seed, kernel, iterations=1)
    seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return (seed.astype(bool) & support_mask.astype(bool))


def _assign_instance_masks(mask: np.ndarray, boxes: list[BBox], image_width: int, image_height: int) -> list[np.ndarray]:
    """Assign foreground pixels to item boxes, favoring small/specific boxes.

    Each foreground pixel is assigned at most once. Smaller boxes claim pixels first;
    large parent boxes then receive only remaining foreground pixels. This preserves
    major parent crops while reducing contamination from nearby child items.
    """
    labels = np.full((image_height, image_width), -1, dtype=np.int32)
    order = sorted(range(len(boxes)), key=lambda idx: (boxes[idx].area, boxes[idx].top, boxes[idx].left))
    for idx in order:
        box = boxes[idx]
        local_fg = mask[box.top : box.bottom, box.left : box.right].astype(bool)
        if not local_fg.any():
            continue
        local_labels = labels[box.top : box.bottom, box.left : box.right]
        claim = local_fg & (local_labels < 0)
        local_labels[claim] = idx

    item_masks: list[np.ndarray] = []
    for idx, box in enumerate(boxes):
        item_mask = labels == idx
        if int(item_mask.sum()) == 0:
            fallback = np.zeros_like(mask, dtype=bool)
            fallback[box.top : box.bottom, box.left : box.right] = mask[box.top : box.bottom, box.left : box.right]
            item_mask = fallback
        item_masks.append(item_mask)
    return item_masks


def _crop_with_instance_mask(
    image: Image.Image,
    box: BBox,
    item_mask: np.ndarray,
    padding: int,
    transparent_bg: bool,
) -> Image.Image:
    width, height = image.size
    padded = box.padded(padding, width, height)
    rgba = np.asarray(image.convert("RGBA")).copy()
    local = rgba[padded.top : padded.bottom, padded.left : padded.right].copy()
    local_mask = item_mask[padded.top : padded.bottom, padded.left : padded.right].astype(bool)
    if local_mask.shape[:2] != local.shape[:2]:
        return image.crop((padded.left, padded.top, padded.right, padded.bottom)).convert("RGBA")

    if transparent_bg:
        local[:, :, 3] = np.where(local_mask, local[:, :, 3], 0).astype(np.uint8)
    else:
        local[~local_mask, 0] = 255
        local[~local_mask, 1] = 255
        local[~local_mask, 2] = 255
        local[~local_mask, 3] = 255
    return Image.fromarray(local, mode="RGBA")


def _instance_mask_stats(mask: np.ndarray, boxes: list[BBox], item_masks: list[np.ndarray]) -> dict[str, int]:
    assigned = np.zeros_like(mask, dtype=bool)
    empty = 0
    for item_mask in item_masks:
        pixels = int(item_mask.sum())
        if pixels == 0:
            empty += 1
        assigned |= item_mask.astype(bool)
    support_pixels = int(mask.sum())
    assigned_pixels = int((assigned & mask).sum())
    return {
        "assigned_pixels": assigned_pixels,
        "unassigned_pixels": max(0, support_pixels - assigned_pixels),
        "empty_items": empty,
    }


def _save_instance_debug(item_masks: list[np.ndarray], path: Path) -> None:
    if not item_masks:
        return
    height, width = item_masks[0].shape
    preview = np.zeros((height, width), dtype=np.uint8)
    for idx, item_mask in enumerate(item_masks, start=1):
        value = 32 + ((idx * 37) % 223)
        preview[item_mask.astype(bool)] = value
    Image.fromarray(preview, mode="L").save(path)


def _final_refine_after_recovery(
    mask: np.ndarray,
    boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> tuple[list[BBox], int, int]:
    boxes, component_splits = _split_broad_boxes_by_components(mask, boxes, image_width, image_height, min_area)
    boxes, stacked_splits = _split_stacked_and_row_boxes(mask, boxes, image_width, image_height, min_area)
    boxes, pruned = _prune_parent_child_boxes(mask, boxes)
    boxes, second_component_splits = _split_broad_boxes_by_components(mask, boxes, image_width, image_height, min_area)
    return _dedupe_boxes(boxes), component_splits + stacked_splits + second_component_splits, pruned


def _split_broad_boxes_by_components(
    mask: np.ndarray,
    boxes: list[BBox],
    image_width: int,
    image_height: int,
    min_area: int,
) -> tuple[list[BBox], int]:
    result: list[BBox] = []
    splits = 0
    for box in boxes:
        if not _should_component_split(box, image_width, image_height):
            result.append(box)
            continue
        local = mask[box.top : box.bottom, box.left : box.right].astype(bool)
        if not local.any():
            continue
        local_min_area = max(min_area, int(local.sum() * 0.035), 32)
        components = components_bboxes(local, min_area=local_min_area, min_size=4)
        child_boxes: list[BBox] = []
        for child, _area in components:
            mapped = BBox(
                box.left + child.left,
                box.top + child.top,
                box.left + child.right,
                box.top + child.bottom,
            )
            if mapped.area >= min_area:
                child_boxes.append(mapped)
        child_boxes = _dedupe_boxes(child_boxes)
        if len(child_boxes) >= 2 and _children_cover_parent(mask, box, child_boxes) >= 0.55:
            result.extend(child_boxes)
            splits += len(child_boxes) - 1
        else:
            result.append(box)
    return _dedupe_boxes(result), splits


def _should_component_split(box: BBox, image_width: int, image_height: int) -> bool:
    image_area = max(1, image_width * image_height)
    return (
        box.area / image_area > 0.055
        or (box.width > image_width * 0.28 and box.height > image_height * 0.05)
        or (box.height > image_height * 0.12 and box.width > image_width * 0.05)
        or box.width > box.height * 2.0
        or box.height > box.width * 1.35
    )


def _children_cover_parent(mask: np.ndarray, parent: BBox, children: list[BBox]) -> float:
    parent_mask = mask[parent.top : parent.bottom, parent.left : parent.right].astype(bool)
    parent_pixels = int(parent_mask.sum())
    if parent_pixels <= 0:
        return 0.0
    covered = np.zeros_like(parent_mask, dtype=bool)
    for child in children:
        left = max(parent.left, child.left) - parent.left
        top = max(parent.top, child.top) - parent.top
        right = min(parent.right, child.right) - parent.left
        bottom = min(parent.bottom, child.bottom) - parent.top
        if right > left and bottom > top:
            covered[top:bottom, left:right] |= parent_mask[top:bottom, left:right]
    return float(int(covered.sum()) / parent_pixels)


def _prune_parent_child_boxes(mask: np.ndarray, boxes: list[BBox]) -> tuple[list[BBox], int]:
    boxes = sort_boxes_reading_order(_dedupe_boxes(boxes))
    remove: set[int] = set()
    for i, parent in enumerate(boxes):
        children: list[BBox] = []
        for j, child in enumerate(boxes):
            if i == j or j in remove:
                continue
            if child.area >= parent.area * 0.78:
                continue
            if _contained_ratio(child, parent) > 0.74 or _iou(child, parent) > 0.45:
                children.append(child)
        if len(children) >= 2 and _children_cover_parent(mask, parent, children) > 0.78:
            remove.add(i)
    return [box for idx, box in enumerate(boxes) if idx not in remove], len(remove)


def _contained_ratio(a: BBox, b: BBox) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    return float(inter / max(1, a.area))


def _iou(a: BBox, b: BBox) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    union = max(1, a.area + b.area - inter)
    return float(inter / union)


def _split_stacked_and_row_boxes(mask: np.ndarray, boxes: list[BBox], image_width: int, image_height: int, min_area: int) -> tuple[list[BBox], int]:
    from .extract_items_plus import _split_box_recursive

    result: list[BBox] = []
    splits = 0
    for box in boxes:
        long_vertical = box.height > box.width * 1.45 and box.height > image_height * 0.12
        long_horizontal = box.width > box.height * 2.2 and box.width > image_width * 0.18
        if long_vertical or long_horizontal:
            pieces, count = _split_box_recursive(mask, box, image_width, image_height, min_area, depth=0)
            if count > 0 and len(pieces) > 1:
                result.extend(pieces)
                splits += count
                continue
        result.append(box)
    return _dedupe_boxes(result), splits
