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
    apply_mask_as_alpha,
    clean_mask,
    components_bboxes,
    crop_mask,
    crop_with_padding,
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
    boxes = sort_boxes_reading_order(_dedupe_boxes(boxes))

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
    for idx, box in enumerate(boxes, start=1):
        crop = crop_with_padding(image, box, padding=padding)
        if transparent_bg:
            local_mask = crop_mask(support_mask, box, padding=padding)
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
        source=str(source),
        mode="hybrid-alpha-support-rgb-core",
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
