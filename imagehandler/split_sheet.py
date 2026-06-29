from __future__ import annotations

from pathlib import Path

from .debug import save_boxes_overlay, save_mask
from .io import ensure_output_dir, load_image
from .mask_ops import (
    clean_mask,
    components_bboxes,
    crop_with_padding,
    foreground_mask_from_background,
    merge_close_boxes,
    normalize_size,
    sort_boxes_x,
    split_equal_boxes,
    vertical_projection_segments,
)
from .reports import BBox, OperationReport


def split_sheet(
    input_path: str | Path,
    output_dir: str | Path,
    views: int = 4,
    padding: int = 24,
    min_area: int = 1000,
    merge_distance: int = 24,
    normalize: int | None = None,
    threshold: float = 28.0,
    debug: bool = False,
) -> OperationReport:
    image = load_image(input_path, "RGBA")
    out_dir = ensure_output_dir(output_dir)
    w, h = image.size

    mask = foreground_mask_from_background(image, threshold=threshold)
    mask = clean_mask(mask, open_size=3, close_size=9, fill_holes=True)

    boxes, mode, warnings = _detect_view_boxes(
        mask=mask,
        image_width=w,
        image_height=h,
        views=views,
        min_area=min_area,
        merge_distance=merge_distance,
    )

    outputs: list[str] = []
    for idx, box in enumerate(boxes, start=1):
        crop = crop_with_padding(image, box, padding=padding)
        if normalize:
            crop = normalize_size(crop, normalize)
        path = out_dir / f"view_{idx:02d}.png"
        crop.save(path)
        outputs.append(str(path))

    report = OperationReport(
        ok=(len(boxes) == views and not warnings),
        operation="split-sheet",
        source=str(input_path),
        mode=mode,
        warnings=warnings,
        metrics={
            "views_requested": views,
            "views_detected": len(boxes),
            "image_width": w,
            "image_height": h,
            "min_area": min_area,
            "merge_distance": merge_distance,
        },
        boxes=boxes,
        outputs=outputs,
    )
    report.save(out_dir / "manifest.json")

    if debug:
        save_mask(mask, out_dir / "debug_mask.png")
        save_boxes_overlay(image, boxes, out_dir / "debug_boxes.png")

    return report


def _detect_view_boxes(
    mask,
    image_width: int,
    image_height: int,
    views: int,
    min_area: int,
    merge_distance: int,
) -> tuple[list[BBox], str, list[str]]:
    warnings: list[str] = []

    components = components_bboxes(mask, min_area=min_area, min_size=12)
    boxes = [box for box, _area in components]
    boxes = merge_close_boxes(boxes, image_width, image_height, distance=merge_distance)

    if len(boxes) >= views:
        # Character sheets can contain extra artifacts; choose the largest view-like boxes.
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)[:views]
        boxes = sort_boxes_x(boxes)
        if len(boxes) > views:
            warnings.append("More components than requested views; selected largest boxes.")
        return boxes, "component", warnings

    projection = vertical_projection_segments(mask, expected=views, min_gap=8, smooth=21)
    if len(projection) == views:
        return sort_boxes_x(projection), "projection", warnings

    warnings.append(
        f"Expected {views} views but component/projection detection found {len(boxes)} usable groups. "
        "Used tightened equal-width fallback."
    )
    return split_equal_boxes(mask, views), "equal_fallback", warnings
