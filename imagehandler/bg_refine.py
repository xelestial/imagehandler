from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .mask_ops import clean_mask, estimate_background_rgb, pil_to_rgba_array


def _background_like_mask(image: Image.Image, threshold: float = 24.0, border: int = 32) -> np.ndarray:
    rgba = pil_to_rgba_array(image)
    bg = np.array(estimate_background_rgb(rgba, border=border), dtype=np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    distance = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    return distance <= float(threshold)


def _low_chroma_light_mask(image: Image.Image, min_value: int = 210, max_chroma: int = 16) -> np.ndarray:
    rgba = pil_to_rgba_array(image)
    rgb = rgba[:, :, :3].astype(np.int16)
    value = rgb.max(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return (value >= int(min_value)) & (chroma <= int(max_chroma))


def _border_connected(binary: np.ndarray) -> np.ndarray:
    b = binary.astype(np.uint8)
    if b.size == 0 or b.max() == 0:
        return np.zeros_like(binary, dtype=bool)

    _n, labels = cv2.connectedComponents(b, connectivity=8)
    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros_like(binary, dtype=bool)
    return np.isin(labels, border_labels)


def _inner_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    if width <= 0:
        return np.zeros_like(mask, dtype=bool)
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return mask.astype(bool) & (dist <= float(width))


def refine_foreground_mask_against_background(
    image: Image.Image,
    mask: np.ndarray,
    mode: str = "safe",
    bg_threshold: float = 24.0,
    fringe_width: int = 1,
) -> np.ndarray:
    """Refine a foreground mask without assuming a white background.

    Modes:
      safe       : morphology only; does not delete pixels by color.
      aggressive : additionally removes only border-connected, low-chroma light
                   pixels. Use only for studio white/off-white backgrounds.
      off        : return mask unchanged.

    The default is intentionally conservative because character skin, blonde
    hair, light clothing, and antialiased edges can be close to a white or gray
    background. Background cleanup by color is only safe when explicitly enabled.
    """
    mode = (mode or "safe").lower().strip()
    m = mask.astype(bool)

    if mode in {"off", "none", "false", "0"}:
        return m

    # Never fill holes for character cutouts. Spaces between arms, legs, hair,
    # and torso are valid background and should remain removable.
    m = clean_mask(m, open_size=1, close_size=2, fill_holes=False)

    if mode not in {"aggressive", "white", "white-bg", "white_bg"}:
        return m.astype(bool)

    # Aggressive mode is restricted to low-chroma light pixels to avoid deleting
    # bright skin or colored foreground edges. It is deliberately not the default.
    bg_like = _background_like_mask(image, threshold=bg_threshold)
    light_neutral = _low_chroma_light_mask(image)
    removable_bg = bg_like & light_neutral

    border_bg = _border_connected(removable_bg)
    m = m & ~border_bg

    if fringe_width > 0:
        boundary = _inner_boundary(m, width=fringe_width)
        m = m & ~(boundary & removable_bg)

    m = clean_mask(m, open_size=1, close_size=1, fill_holes=False)
    return m.astype(bool)


def zero_transparent_rgb(image: Image.Image, alpha_threshold: int = 0) -> Image.Image:
    rgba = np.array(image.convert("RGBA"), copy=True)
    transparent = rgba[:, :, 3] <= int(alpha_threshold)
    rgba[transparent, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")
