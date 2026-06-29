from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .mask_ops import clean_mask, estimate_background_rgb, pil_to_rgba_array


def _background_like_mask(image: Image.Image, threshold: float = 34.0, border: int = 32) -> np.ndarray:
    rgba = pil_to_rgba_array(image)
    bg = np.array(estimate_background_rgb(rgba, border=border), dtype=np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    distance = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    return distance <= float(threshold)


def _border_connected(binary: np.ndarray) -> np.ndarray:
    b = binary.astype(np.uint8)
    if b.size == 0 or b.max() == 0:
        return np.zeros_like(binary, dtype=bool)

    _n, labels = cv2.connectedComponents(b, connectivity=8)
    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros_like(binary, dtype=bool)
    return np.isin(labels, border_labels)


def _inner_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
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
    bg_threshold: float = 34.0,
    fringe_width: int = 2,
) -> np.ndarray:
    """Refine a foreground mask for white/off-white studio backgrounds.

    This fixes two common failure modes:
    - near-white halo pixels left around the outer silhouette
    - background holes between body parts, such as the space between an arm and torso

    The cleanup is intentionally conservative: it mainly removes background-like
    regions that are connected to the image border, so white clothing or white
    foreground details that are not connected to the border are less likely to be
    deleted.
    """
    m = mask.astype(bool)

    # Do not fill holes for human/character cutouts. Arm gaps and negative space
    # are valid background and must remain removable.
    m = clean_mask(m, open_size=2, close_size=3, fill_holes=False)

    bg_like = _background_like_mask(image, threshold=bg_threshold)
    border_bg = _border_connected(bg_like)

    # Clear all background-colored pixels connected to the canvas border. This
    # removes both the outside background and background-colored negative spaces
    # that visually connect to the outside, even if the model marked them opaque.
    m = m & ~border_bg

    # Clear near-white pixels on the inner mask boundary to remove white fringing.
    if fringe_width > 0:
        boundary = _inner_boundary(m, width=fringe_width)
        m = m & ~(boundary & bg_like)

    # Final tiny speck cleanup without hole filling.
    m = clean_mask(m, open_size=1, close_size=2, fill_holes=False)
    return m.astype(bool)


def zero_transparent_rgb(image: Image.Image, alpha_threshold: int = 0) -> Image.Image:
    rgba = np.array(image.convert("RGBA"), copy=True)
    transparent = rgba[:, :, 3] <= int(alpha_threshold)
    rgba[transparent, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")
