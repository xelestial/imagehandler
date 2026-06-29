from __future__ import annotations

from pathlib import Path

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
    boxes = [box for box, _area in raw_components]
    boxes = merge_close_boxes(boxes, w, h, distance=merge_distance)
    boxes = [b for b in boxes if b.area >= min_area]
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
        mode="component",
        warnings=warnings,
        metrics={
            "items": len(boxes),
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

    return manifest
