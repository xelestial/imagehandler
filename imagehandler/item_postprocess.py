from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .reports import BBox, OperationReport


def suppress_duplicate_partial_outputs(report: OperationReport, min_area: int = 120) -> OperationReport:
    """Clean extracted item crops after extraction.

    The extractor may produce a useful item mask inside a broad box. This pass
    removes distant disconnected specks/contours inside each crop, trims the crop
    to the remaining visible foreground, then drops obvious duplicate/partial
    item crops conservatively.
    """

    if not report.outputs or not report.boxes or len(report.outputs) != len(report.boxes):
        return report

    paths = [Path(p) for p in report.outputs]
    if not all(p.exists() for p in paths):
        return report

    component_cleanup_count = 0
    trim_count = 0
    for path in paths:
        cleaned, trimmed = _cleanup_and_trim_crop(path, padding=10)
        component_cleanup_count += int(cleaned)
        trim_count += int(trimmed)

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
    if dropped > 0:
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
        report.warnings.append(
            f"Postprocess dropped {duplicate_dropped} duplicate/partial and {artifact_dropped} artifact item crop(s)."
        )
        report.ok = False

    report.metrics["postprocess_duplicate_partial_dropped"] = duplicate_dropped
    report.metrics["postprocess_artifact_dropped"] = artifact_dropped
    report.metrics["postprocess_component_cleanup_count"] = component_cleanup_count
    report.metrics["postprocess_trim_count"] = trim_count
    report.metrics["postprocess_mode"] = "component-cleanup-duplicate-partial-suppression"

    manifest_path = Path(report.outputs[0]).parent / "manifest.json" if report.outputs else None
    if manifest_path is not None:
        try:
            manifest_path.write_text(
                __import__("json").dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    return report


def _cleanup_and_trim_crop(path: Path, padding: int = 10) -> tuple[bool, bool]:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image).copy()
    visible = _visible_mask(arr)
    if int(visible.sum()) < 8:
        return False, False

    cleaned_visible, cleaned = _keep_main_visible_components(visible)
    if int(cleaned_visible.sum()) < 8:
        return False, False

    bg = _background_color(arr)
    transparent_bg = _looks_transparent_background(arr)
    if cleaned:
        if transparent_bg:
            arr[~cleaned_visible, 3] = 0
        else:
            arr[~cleaned_visible, 0] = bg[0]
            arr[~cleaned_visible, 1] = bg[1]
            arr[~cleaned_visible, 2] = bg[2]
            arr[~cleaned_visible, 3] = 255

    ys, xs = np.where(cleaned_visible)
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(arr.shape[1], int(xs.max()) + 1 + padding)
    bottom = min(arr.shape[0], int(ys.max()) + 1 + padding)
    trimmed = left > 0 or top > 0 or right < arr.shape[1] or bottom < arr.shape[0]
    if trimmed:
        arr = arr[top:bottom, left:right]

    if cleaned or trimmed:
        Image.fromarray(arr, mode="RGBA").save(path)
    return cleaned, trimmed


def _visible_mask(arr: np.ndarray) -> np.ndarray:
    alpha = arr[:, :, 3]
    if np.mean(alpha < 250) > 0.05:
        return alpha > 8
    bg = _background_color(arr)
    rgb = arr[:, :, :3].astype(np.int16)
    diff = np.max(np.abs(rgb - np.asarray(bg, dtype=np.int16)), axis=2)
    return (alpha > 8) & (diff > 10)


def _background_color(arr: np.ndarray) -> tuple[int, int, int]:
    h, w = arr.shape[:2]
    patch = max(4, min(16, h // 12, w // 12))
    samples = np.concatenate(
        [
            arr[:patch, :patch, :3].reshape(-1, 3),
            arr[:patch, w - patch :, :3].reshape(-1, 3),
            arr[h - patch :, :patch, :3].reshape(-1, 3),
            arr[h - patch :, w - patch :, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    med = np.median(samples, axis=0).astype(np.uint8)
    return int(med[0]), int(med[1]), int(med[2])


def _looks_transparent_background(arr: np.ndarray) -> bool:
    h, w = arr.shape[:2]
    patch = max(4, min(16, h // 12, w // 12))
    corners = np.concatenate(
        [
            arr[:patch, :patch, 3].reshape(-1),
            arr[:patch, w - patch :, 3].reshape(-1),
            arr[h - patch :, :patch, 3].reshape(-1),
            arr[h - patch :, w - patch :, 3].reshape(-1),
        ]
    )
    return float(np.mean(corners < 8)) > 0.65


def _keep_main_visible_components(visible: np.ndarray) -> tuple[np.ndarray, bool]:
    binary = visible.astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 2:
        return visible.astype(bool), False

    areas = [(label, int(stats[label, cv2.CC_STAT_AREA])) for label in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    main_label, main_area = areas[0]
    if main_area <= 0:
        return visible.astype(bool), False

    main_box = _component_box(stats, main_label)
    keep = labels == main_label
    for label, area in areas[1:]:
        if area < 12:
            continue
        box = _component_box(stats, label)
        close = _box_gap(main_box, box) <= 10
        large = area >= max(60, int(main_area * 0.10))
        medium_attached = close and area >= max(24, int(main_area * 0.035))
        if large or medium_attached:
            keep |= labels == label
    changed = int(keep.sum()) < int(visible.sum())
    return keep.astype(bool), changed


def _component_box(stats: np.ndarray, label: int) -> BBox:
    left = int(stats[label, cv2.CC_STAT_LEFT])
    top = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    return BBox(left, top, left + width, top + height)


def _box_gap(a: BBox, b: BBox) -> int:
    dx = max(0, max(a.left, b.left) - min(a.right, b.right))
    dy = max(0, max(a.top, b.top) - min(a.bottom, b.bottom))
    return max(dx, dy)


def _read_crop_stats(path: Path) -> dict[str, float | int | bool]:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image)
    visible = _visible_mask(arr)
    pixels = int(visible.sum())
    h, w = visible.shape
    if pixels > 0:
        ys, xs = np.where(visible)
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
        "strong_pixels": pixels,
        "fill_ratio": float(pixels / area),
        "strong_fill_ratio": float(pixels / area),
        "local_width": local_w,
        "local_height": local_h,
        "has_visible_nonwhite": bool(pixels > 0),
    }


def _is_post_artifact(box: BBox, stat: dict[str, float | int | bool], min_area: int) -> bool:
    pixels = int(stat["pixels"])
    if pixels <= 0:
        return False
    fill_ratio = float(stat["fill_ratio"])
    short_side = max(1, min(int(stat["width"]), int(stat["height"])))
    long_side = max(1, max(int(stat["width"]), int(stat["height"])))
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
