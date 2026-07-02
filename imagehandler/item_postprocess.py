from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .reports import BBox, OperationReport


def suppress_duplicate_partial_outputs(report: OperationReport, min_area: int = 120) -> OperationReport:
    """Remove obvious duplicate/partial item crops after extraction.

    This is deliberately conservative. It only removes crops that are mostly
    contained inside a larger crop and look like central residuals, internal
    fragments, or thin/low-fill artifacts. Edge pieces, shoes, belts, and likely
    left/right symmetric parts are protected.
    """

    if not report.outputs or not report.boxes or len(report.outputs) != len(report.boxes):
        return report

    paths = [Path(p) for p in report.outputs]
    if not all(p.exists() for p in paths):
        return report

    stats = [_read_crop_stats(path) for path in paths]
    keep = [True] * len(paths)
    duplicate_dropped = 0
    artifact_dropped = 0

    for idx, (box, stat) in enumerate(zip(report.boxes, stats, strict=True)):
        if _is_post_artifact(box, stat, min_area):
            keep[idx] = False
            artifact_dropped += 1

    for idx, box in enumerate(report.boxes):
        if not keep[idx]:
            continue
        for other_idx, other in enumerate(report.boxes):
            if idx == other_idx or not keep[other_idx]:
                continue
            if _is_duplicate_partial_item(box, other, stats[idx], min_area):
                keep[idx] = False
                duplicate_dropped += 1
                break

    dropped = duplicate_dropped + artifact_dropped
    if dropped <= 0:
        report.metrics["postprocess_duplicate_partial_dropped"] = 0
        report.metrics["postprocess_artifact_dropped"] = 0
        report.metrics["postprocess_mode"] = "duplicate-partial-suppression"
        return report

    kept_records = [
        (box, path.read_bytes())
        for keep_flag, box, path in zip(keep, report.boxes, paths, strict=True)
        if keep_flag
    ]

    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    out_dir = paths[0].parent
    new_boxes: list[BBox] = []
    new_outputs: list[str] = []
    for new_idx, (box, data) in enumerate(kept_records, start=1):
        new_path = out_dir / f"item_{new_idx:03d}.png"
        new_path.write_bytes(data)
        new_boxes.append(box)
        new_outputs.append(str(new_path))

    report.boxes = new_boxes
    report.outputs = new_outputs
    report.metrics["items"] = len(new_boxes)
    report.metrics["postprocess_duplicate_partial_dropped"] = duplicate_dropped
    report.metrics["postprocess_artifact_dropped"] = artifact_dropped
    report.metrics["postprocess_mode"] = "duplicate-partial-suppression"
    report.warnings.append(
        f"Postprocess dropped {duplicate_dropped} duplicate/partial and {artifact_dropped} artifact item crop(s)."
    )
    report.ok = False if report.warnings else report.ok

    manifest_path = out_dir / "manifest.json"
    try:
        report.save(manifest_path)
    except Exception:
        pass
    return report


def _read_crop_stats(path: Path) -> dict[str, float | int | bool]:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3].astype(np.int16)
    nonwhite = (alpha > 8) & np.any(rgb < 246, axis=2)
    strong = (alpha > 8) & np.any(rgb < 232, axis=2)
    pixels = int(nonwhite.sum())
    strong_pixels = int(strong.sum())
    h, w = nonwhite.shape
    if pixels > 0:
        ys, xs = np.where(nonwhite)
        local_w = int(xs.max() - xs.min() + 1)
        local_h = int(ys.max() - ys.min() + 1)
    else:
        local_w = 0
        local_h = 0
    area = max(1, w * h)
    return {
        "width": w,
        "height": h,
        "pixels": pixels,
        "strong_pixels": strong_pixels,
        "fill_ratio": float(pixels / area),
        "strong_fill_ratio": float(strong_pixels / area),
        "local_width": local_w,
        "local_height": local_h,
        "has_visible_nonwhite": bool(pixels > 0),
    }


def _is_post_artifact(box: BBox, stat: dict[str, float | int | bool], min_area: int) -> bool:
    pixels = int(stat["pixels"])
    # Do not judge all-white/pale crops here; they may be legitimate white fabric.
    if pixels <= 0:
        return False
    fill_ratio = float(stat["fill_ratio"])
    short_side = max(1, min(box.width, box.height))
    long_side = max(1, max(box.width, box.height))
    thin_ratio = short_side / long_side
    if short_side <= 5 and long_side >= 36:
        return True
    if thin_ratio < 0.08 and pixels < min_area * 4:
        return True
    if fill_ratio < 0.035 and pixels < min_area * 5:
        return True
    if fill_ratio < 0.06 and long_side / short_side >= 4.8 and pixels < min_area * 6:
        return True
    return False


def _is_duplicate_partial_item(
    box: BBox,
    parent: BBox,
    stat: dict[str, float | int | bool],
    min_area: int,
) -> bool:
    if parent.area <= box.area * 1.7:
        return False
    contained = _contained_ratio(box, parent)
    if contained < 0.72:
        return False

    rel_cx, rel_cy = _relative_center(box, parent)
    if _is_protected_edge_part(rel_cx, rel_cy):
        return False
    if _looks_like_belt_or_horizontal_part(box, parent):
        return False

    pixels = int(stat["pixels"])
    fill_ratio = float(stat["fill_ratio"])
    short_side = max(1, min(box.width, box.height))
    long_side = max(1, max(box.width, box.height))
    aspect = long_side / short_side

    central_vertical_residual = (
        0.34 <= rel_cx <= 0.66
        and 0.16 <= rel_cy <= 0.84
        and box.height >= box.width * 1.12
        and box.area <= parent.area * 0.42
    )
    small_internal_fragment = (
        box.area <= parent.area * 0.22
        and pixels > 0
        and pixels < min_area * 5
        and fill_ratio < 0.34
    )
    long_internal_sliver = aspect >= 3.6 and fill_ratio < 0.24 and box.area <= parent.area * 0.35

    return central_vertical_residual or small_internal_fragment or long_internal_sliver


def _is_protected_edge_part(rel_cx: float, rel_cy: float) -> bool:
    # Protect left/right arms, side panels, shoes, lower leg parts, and top edge bands.
    return rel_cx < 0.25 or rel_cx > 0.75 or rel_cy < 0.14 or rel_cy > 0.82


def _looks_like_belt_or_horizontal_part(box: BBox, parent: BBox) -> bool:
    if box.width <= 0 or box.height <= 0:
        return False
    aspect = box.width / max(1, box.height)
    return aspect >= 1.55 and box.height <= parent.height * 0.34


def _relative_center(box: BBox, parent: BBox) -> tuple[float, float]:
    pcx = (box.left + box.right) / 2.0
    pcy = (box.top + box.bottom) / 2.0
    return (
        (pcx - parent.left) / max(1, parent.width),
        (pcy - parent.top) / max(1, parent.height),
    )


def _contained_ratio(a: BBox, b: BBox) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    return float(inter / max(1, a.area))
