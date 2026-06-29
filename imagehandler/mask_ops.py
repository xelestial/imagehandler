from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .reports import BBox


def pil_to_rgba_array(image: Image.Image) -> np.ndarray:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return np.asarray(image)


def estimate_background_rgb(
    rgba: np.ndarray,
    border: int = 24,
    sample_alpha_aware: bool = True,
) -> tuple[int, int, int]:
    """Estimate background RGB from border pixels using a median.

    Works better than assuming pure white because generated/JPEG images often
    contain near-white, off-white, or compressed background pixels.
    """
    h, w = rgba.shape[:2]
    b = max(1, min(border, h // 2, w // 2))
    strips = [
        rgba[:b, :, :],
        rgba[h - b :, :, :],
        rgba[:, :b, :],
        rgba[:, w - b :, :],
    ]
    border_pixels = np.concatenate([s.reshape(-1, 4) for s in strips], axis=0)

    if sample_alpha_aware and rgba.shape[2] == 4:
        opaque = border_pixels[:, 3] > 8
        if opaque.any():
            border_pixels = border_pixels[opaque]

    rgb = border_pixels[:, :3].astype(np.float32)
    median = np.median(rgb, axis=0)
    return tuple(int(round(x)) for x in median)


def foreground_mask_from_background(
    image: Image.Image,
    threshold: float = 28.0,
    border: int = 24,
    alpha_threshold: int = 8,
) -> np.ndarray:
    """Create a foreground mask from alpha or estimated background distance."""
    rgba = pil_to_rgba_array(image)
    alpha = rgba[:, :, 3]

    # Existing transparency is authoritative when meaningful.
    transparent_ratio = float((alpha < 250).mean())
    if transparent_ratio > 0.005:
        return alpha > alpha_threshold

    bg = np.array(estimate_background_rgb(rgba, border=border), dtype=np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    distance = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    return distance > threshold


def clean_mask(
    mask: np.ndarray,
    open_size: int = 3,
    close_size: int = 7,
    fill_holes: bool = True,
) -> np.ndarray:
    """Remove specks, join tiny gaps, and optionally fill holes."""
    m = mask.astype(np.uint8) * 255

    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    out = m > 0
    if fill_holes:
        out = ndimage.binary_fill_holes(out)
    return out.astype(bool)


def bbox_from_mask(mask: np.ndarray) -> BBox | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def components_bboxes(
    mask: np.ndarray,
    min_area: int = 80,
    min_size: int = 3,
) -> list[tuple[BBox, int]]:
    """Return connected component boxes and pixel areas."""
    binary = mask.astype(np.uint8)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result: list[tuple[BBox, int]] = []
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if int(area) < min_area or int(w) < min_size or int(h) < min_size:
            continue
        result.append((BBox(int(x), int(y), int(x + w), int(y + h)), int(area)))
    return result


def merge_close_boxes(
    boxes: Iterable[BBox],
    image_width: int,
    image_height: int,
    distance: int = 12,
) -> list[BBox]:
    """Merge boxes whose distance-expanded rectangles overlap."""
    pending = list(boxes)
    changed = True

    while changed:
        changed = False
        merged: list[BBox] = []
        used = [False] * len(pending)

        for i, box in enumerate(pending):
            if used[i]:
                continue

            current = box
            used[i] = True

            for j in range(i + 1, len(pending)):
                if used[j]:
                    continue
                a = current.expand(distance, image_width, image_height)
                b = pending[j].expand(distance, image_width, image_height)
                if a.intersects(b):
                    current = current.union(pending[j])
                    used[j] = True
                    changed = True

            merged.append(current)

        pending = merged

    return pending


def sort_boxes_reading_order(boxes: list[BBox], row_tolerance: int | None = None) -> list[BBox]:
    """Sort boxes top-to-bottom, left-to-right.

    A row-tolerance makes irregular item sheets more stable than raw y/x sorting.
    """
    if not boxes:
        return []

    median_h = np.median([b.height for b in boxes])
    tol = row_tolerance if row_tolerance is not None else max(12, int(median_h * 0.35))

    rows: list[list[BBox]] = []
    for box in sorted(boxes, key=lambda b: (b.top, b.left)):
        placed = False
        cy = box.center[1]
        for row in rows:
            row_cy = float(np.mean([b.center[1] for b in row]))
            if abs(cy - row_cy) <= tol:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    ordered: list[BBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.left))
    return ordered


def sort_boxes_x(boxes: list[BBox]) -> list[BBox]:
    return sorted(boxes, key=lambda b: (b.left, b.top))


def crop_with_padding(
    image: Image.Image,
    box: BBox,
    padding: int = 0,
    fill: tuple[int, int, int, int] = (255, 255, 255, 0),
) -> Image.Image:
    """Crop with optional padding, preserving requested outside area with fill."""
    if padding <= 0:
        return image.crop(tuple(box.to_list()))

    w, h = image.size
    padded = box.padded(padding, w, h)
    cropped = image.crop(tuple(padded.to_list()))

    # If box was clipped at image border, add the missing padding back.
    left_missing = max(0, padding - box.left)
    top_missing = max(0, padding - box.top)
    right_missing = max(0, box.right + padding - w)
    bottom_missing = max(0, box.bottom + padding - h)

    if any((left_missing, top_missing, right_missing, bottom_missing)):
        new_w = cropped.width + left_missing + right_missing
        new_h = cropped.height + top_missing + bottom_missing
        canvas = Image.new("RGBA", (new_w, new_h), fill)
        canvas.paste(cropped, (left_missing, top_missing))
        return canvas

    return cropped


def apply_mask_as_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = pil_to_rgba_array(image).copy()
    rgba[:, :, 3] = (mask.astype(np.uint8) * 255)
    return Image.fromarray(rgba, mode="RGBA")


def crop_mask(mask: np.ndarray, box: BBox, padding: int = 0) -> np.ndarray:
    h, w = mask.shape
    padded = box.padded(padding, w, h)
    return mask[padded.top : padded.bottom, padded.left : padded.right]


def to_square_canvas(
    image: Image.Image,
    fill: tuple[int, int, int, int] = (255, 255, 255, 0),
) -> Image.Image:
    side = max(image.width, image.height)
    canvas = Image.new("RGBA", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def normalize_size(image: Image.Image, size: int, fill: tuple[int, int, int, int] = (255, 255, 255, 0)) -> Image.Image:
    """Fit image into a square canvas of the requested size."""
    img = image.convert("RGBA")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), fill)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
    return canvas


def mask_metrics(mask: np.ndarray) -> dict[str, float | int]:
    h, w = mask.shape
    area = int(mask.sum())
    bbox = bbox_from_mask(mask)
    return {
        "foreground_area": area,
        "foreground_area_ratio": float(area / max(1, h * w)),
        "bbox_area_ratio": float((bbox.area / max(1, h * w)) if bbox else 0.0),
        "touches_border": int(
            bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
        ),
    }


def vertical_projection_segments(
    mask: np.ndarray,
    expected: int,
    min_gap: int = 6,
    smooth: int = 15,
) -> list[BBox]:
    """Find x-ranged segments using vertical foreground projection valleys."""
    h, w = mask.shape
    projection = mask.sum(axis=0).astype(np.float32)

    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        projection = np.convolve(projection, kernel, mode="same")

    threshold = max(1.0, float(projection.max()) * 0.02)
    occupied = projection > threshold

    # Fill tiny internal gaps.
    if min_gap > 1:
        occ = occupied.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_gap, 1))
        occ = cv2.morphologyEx(occ.reshape(1, -1), cv2.MORPH_CLOSE, kernel).reshape(-1)
        occupied = occ > 0

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for x, is_fg in enumerate(occupied):
        if is_fg and start is None:
            start = x
        elif not is_fg and start is not None:
            if x - start >= 2:
                segments.append((start, x))
            start = None
    if start is not None:
        segments.append((start, w))

    # Convert each x segment to the actual y bbox within that range.
    boxes: list[BBox] = []
    for left, right in segments:
        sub = mask[:, left:right]
        ys, xs = np.where(sub)
        if xs.size:
            boxes.append(BBox(left + int(xs.min()), int(ys.min()), left + int(xs.max()) + 1, int(ys.max()) + 1))

    if len(boxes) == expected:
        return boxes

    # If projection did not isolate expected segments, return empty and let caller fallback.
    return []


def split_equal_boxes(mask: np.ndarray, count: int) -> list[BBox]:
    """Last-resort split by equal x ranges, tightened to foreground inside each range."""
    h, w = mask.shape
    boxes: list[BBox] = []
    for i in range(count):
        left = math.floor(i * w / count)
        right = math.floor((i + 1) * w / count)
        sub = mask[:, left:right]
        local = bbox_from_mask(sub)
        if local is None:
            boxes.append(BBox(left, 0, right, h))
        else:
            boxes.append(BBox(left + local.left, local.top, left + local.right, local.bottom))
    return boxes
