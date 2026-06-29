from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

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
    """Refine a support mask without assuming a white background.

    Modes:
      safe/off   : return support mask unchanged. This preserves the model's
                   soft alpha edge in bg_remove.py.
      aggressive : additionally removes only border-connected, low-chroma light
                   pixels. Use only for studio white/off-white backgrounds.
    """
    mode = (mode or "safe").lower().strip()
    m = mask.astype(bool)

    if mode in {"off", "none", "false", "0", "safe", "conservative"}:
        return m

    if mode not in {"aggressive", "white", "white-bg", "white_bg"}:
        return m

    # Aggressive mode is intentionally opt-in. It targets only low-chroma light
    # pixels connected to the canvas border, reducing white halo on white studio
    # backgrounds without treating all background colors as removable.
    m = clean_mask(m, open_size=1, close_size=1, fill_holes=False)
    bg_like = _background_like_mask(image, threshold=bg_threshold)
    light_neutral = _low_chroma_light_mask(image)
    removable_bg = bg_like & light_neutral

    border_bg = _border_connected(removable_bg)
    m = m & ~border_bg

    if fringe_width > 0:
        boundary = _inner_boundary(m, width=fringe_width)
        m = m & ~(boundary & removable_bg)

    return m.astype(bool)


def smooth_alpha_edges(alpha: np.ndarray, radius: float = 0.65) -> np.ndarray:
    """Slightly soften only the alpha transition band.

    This is not the same as replacing the model output with a binary mask. It
    keeps the model's alpha matte, then gently smooths noisy stair-step pixels
    around the silhouette.
    """
    a = np.asarray(alpha).astype(np.uint8)
    if radius <= 0 or a.size == 0 or a.max() == 0:
        return a

    fg = a > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    soft = (a > 0) & (a < 255)
    zone = boundary | soft
    zone = cv2.dilate(zone.astype(np.uint8), kernel, iterations=1) > 0

    blurred = cv2.GaussianBlur(a, (0, 0), sigmaX=float(radius), sigmaY=float(radius))
    out = a.copy()
    out[zone] = blurred[zone]
    out[out < 3] = 0
    out[out > 252] = 255
    return out.astype(np.uint8)


def decontaminate_edge_rgb(
    image: Image.Image,
    solid_alpha: int = 250,
    width: float = 3.0,
) -> Image.Image:
    """Remove black/white matte contamination from semi-transparent edges.

    Many background-removal models output a useful soft alpha but leave edge RGB
    mixed with the old background. When viewed over another background, that
    becomes a dark or light halo. This replaces only near-edge semi-transparent
    RGB values with the nearest fully solid foreground RGB while preserving alpha.
    """
    rgba = np.array(image.convert("RGBA"), copy=True)
    alpha = rgba[:, :, 3]
    fg = alpha > 0
    solid = alpha >= int(solid_alpha)

    if not fg.any() or not solid.any():
        return zero_transparent_rgb(Image.fromarray(rgba, mode="RGBA"))

    dist_inside = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 3)
    edge = (alpha > 0) & (alpha < int(solid_alpha)) & (dist_inside <= float(width))
    if edge.any():
        nearest = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        nearest_rgb = rgba[nearest[0], nearest[1], :3]
        rgba[edge, :3] = nearest_rgb[edge]

    rgba[alpha == 0, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def zero_transparent_rgb(image: Image.Image, alpha_threshold: int = 0) -> Image.Image:
    rgba = np.array(image.convert("RGBA"), copy=True)
    transparent = rgba[:, :, 3] <= int(alpha_threshold)
    rgba[transparent, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")
