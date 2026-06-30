from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def alpha_quality_metrics(image: Image.Image) -> dict[str, Any]:
    """No-reference alpha matte diagnostics for background-removal outputs.

    These metrics do not require a ground-truth matte. They are intended for
    operational QA: detecting binary mattes, suspicious halos, RGB leakage in
    transparent pixels, excessive fragmentation, and jagged silhouettes.
    """
    rgba = np.asarray(image.convert("RGBA"))
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    total = max(1, int(alpha.size))
    transparent = alpha <= 8
    opaque = alpha >= 248
    foreground = alpha > 8
    soft = (alpha > 8) & (alpha < 248)

    fg_count = max(1, int(foreground.sum()))
    soft_count = int(soft.sum())
    transparent_count = max(1, int(transparent.sum()))
    unique_alpha = np.unique(alpha)
    soft_unique_alpha = np.unique(alpha[soft]) if soft_count else np.array([], dtype=np.uint8)

    transparent_rgb_leak = transparent & (np.abs(rgb.astype(np.int16)).sum(axis=2) > 3)
    luminance = (
        0.2126 * rgb[:, :, 0].astype(np.float32)
        + 0.7152 * rgb[:, :, 1].astype(np.float32)
        + 0.0722 * rgb[:, :, 2].astype(np.float32)
    )
    semi_dark = soft & (luminance < 18)
    semi_light = soft & (luminance > 238)

    components = _component_stats(foreground, min_area=max(16, int(total * 0.00005)))
    edge_metrics = _edge_metrics(alpha)

    return {
        "alpha_min": int(alpha.min()) if total else 0,
        "alpha_max": int(alpha.max()) if total else 0,
        "alpha_unique_levels": int(unique_alpha.size),
        "soft_alpha_unique_levels": int(soft_unique_alpha.size),
        "transparent_ratio": float(transparent.mean()),
        "opaque_ratio": float(opaque.mean()),
        "semi_alpha_ratio": float(soft.mean()),
        "soft_alpha_fg_ratio": float(soft_count / fg_count),
        "transparent_rgb_leak_ratio": float(transparent_rgb_leak.sum() / transparent_count),
        "semi_dark_rgb_ratio": float(semi_dark.sum() / max(1, soft_count)),
        "semi_light_rgb_ratio": float(semi_light.sum() / max(1, soft_count)),
        "component_count": components["component_count"],
        "largest_component_ratio": components["largest_component_ratio"],
        "component_area_ratios": components["component_area_ratios"],
        **edge_metrics,
        "width": int(w),
        "height": int(h),
    }


def save_alpha_previews(image: Image.Image, output_path: str | Path) -> list[str]:
    """Save white/black/checkerboard previews next to a transparent output."""
    output = Path(output_path)
    rgba = image.convert("RGBA")
    paths: list[str] = []

    previews = {
        "white": _composite_on_color(rgba, (255, 255, 255, 255)),
        "black": _composite_on_color(rgba, (0, 0, 0, 255)),
        "checker": _composite_on_checker(rgba),
    }
    for name, preview in previews.items():
        path = output.with_name(f"{output.stem}.preview.{name}.png")
        preview.save(path)
        paths.append(str(path))
    return paths


def _composite_on_color(image: Image.Image, color: tuple[int, int, int, int]) -> Image.Image:
    bg = Image.new("RGBA", image.size, color)
    return Image.alpha_composite(bg, image).convert("RGB")


def _composite_on_checker(image: Image.Image, tile: int = 32) -> Image.Image:
    w, h = image.size
    a = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.indices((h, w))
    check = ((xx // tile + yy // tile) % 2) == 0
    a[check] = (225, 225, 225, 255)
    a[~check] = (160, 160, 160, 255)
    bg = Image.fromarray(a, mode="RGBA")
    return Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")


def _component_stats(mask: np.ndarray, min_area: int) -> dict[str, Any]:
    b = mask.astype(np.uint8)
    if b.size == 0 or b.max() == 0:
        return {"component_count": 0, "largest_component_ratio": 0.0, "component_area_ratios": []}

    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(b, connectivity=8)
    areas: list[int] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            areas.append(area)
    if not areas:
        return {"component_count": 0, "largest_component_ratio": 0.0, "component_area_ratios": []}

    total = max(1, sum(areas))
    areas_sorted = sorted(areas, reverse=True)
    return {
        "component_count": len(areas_sorted),
        "largest_component_ratio": float(areas_sorted[0] / total),
        "component_area_ratios": [round(float(a / total), 5) for a in areas_sorted[:12]],
    }


def _edge_metrics(alpha: np.ndarray) -> dict[str, Any]:
    fg = (alpha > 8).astype(np.uint8)
    if fg.size == 0 or fg.max() == 0:
        return {
            "edge_band_ratio": 0.0,
            "edge_jaggedness_ratio": 0.0,
            "edge_transition_mean_alpha": 0.0,
            "edge_transition_std_alpha": 0.0,
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, kernel) > 0
    soft = (alpha > 8) & (alpha < 248)
    transition = edge | soft

    contours, _hierarchy = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw_perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    smoothed = (cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.1) > 0.5).astype(np.uint8)
    contours_s, _hierarchy_s = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smooth_perimeter = float(sum(cv2.arcLength(c, True) for c in contours_s))
    jaggedness = raw_perimeter / max(1.0, smooth_perimeter)

    if transition.any():
        vals = alpha[transition].astype(np.float32)
        mean_alpha = float(vals.mean())
        std_alpha = float(vals.std())
    else:
        mean_alpha = 0.0
        std_alpha = 0.0

    return {
        "edge_band_ratio": float(transition.mean()),
        "edge_jaggedness_ratio": float(jaggedness),
        "edge_transition_mean_alpha": mean_alpha,
        "edge_transition_std_alpha": std_alpha,
    }
