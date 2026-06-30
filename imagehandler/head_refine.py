from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .mask_ops import estimate_background_rgb, pil_to_rgba_array


@dataclass(frozen=True)
class HeadRefineResult:
    alpha: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class Roi:
    left: int
    top: int
    right: int
    bottom: int
    method: str

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_list(self) -> list[int | str]:
        return [self.left, self.top, self.right, self.bottom, self.method]


def refine_head_alpha_with_face_priority(
    image: Image.Image,
    alpha: np.ndarray,
    enabled: bool = True,
    bg_threshold: float = 30.0,
    max_removed_roi_ratio: float = 0.08,
) -> HeadRefineResult:
    """Refine only head/face regions, preferring dedicated face libraries.

    Priority:
      1. MediaPipe FaceMesh ROI, if mediapipe is installed and detects faces.
      2. Connected-component top-region heuristic fallback.

    The refinement itself is conservative: inside each head ROI, it removes only
    pixels that look like the image border background and are connected to the
    ROI border. This targets face-side and hair-gap background without running a
    global white cleanup that could damage clothing or white ornaments.
    """
    a = np.asarray(alpha).astype(np.uint8).copy()
    if not enabled or a.size == 0 or a.max() == 0:
        return HeadRefineResult(a, _empty_metrics(enabled=enabled, detector="disabled"))

    rgba = pil_to_rgba_array(image)
    h, w = a.shape[:2]
    rois, detector = _detect_head_rois(image, a)

    total_removed = 0
    skipped = 0
    applied_rois: list[list[int | str]] = []

    for roi in rois:
        if roi.area <= 0:
            continue
        local_alpha = a[roi.top : roi.bottom, roi.left : roi.right]
        local_rgba = rgba[roi.top : roi.bottom, roi.left : roi.right]
        removed = _remove_connected_background_in_roi(
            local_rgba=local_rgba,
            local_alpha=local_alpha,
            bg_threshold=bg_threshold,
            max_removed_roi_ratio=max_removed_roi_ratio,
        )
        if removed is None:
            skipped += 1
            continue
        removed_count = int(removed.sum())
        if removed_count <= 0:
            continue
        local_alpha[removed] = 0
        total_removed += removed_count
        applied_rois.append(roi.to_list())

    return HeadRefineResult(
        a,
        {
            "head_refine_enabled": bool(enabled),
            "head_refine_detector": detector,
            "head_refine_roi_count": len(rois),
            "head_refine_applied_roi_count": len(applied_rois),
            "head_refine_skipped_roi_count": skipped,
            "head_refine_removed_pixels": total_removed,
            "head_refine_removed_ratio": float(total_removed / max(1, h * w)),
            "head_refine_rois": [r.to_list() for r in rois],
            "head_refine_applied_rois": applied_rois,
        },
    )


def _empty_metrics(enabled: bool, detector: str) -> dict[str, Any]:
    return {
        "head_refine_enabled": bool(enabled),
        "head_refine_detector": detector,
        "head_refine_roi_count": 0,
        "head_refine_applied_roi_count": 0,
        "head_refine_skipped_roi_count": 0,
        "head_refine_removed_pixels": 0,
        "head_refine_removed_ratio": 0.0,
        "head_refine_rois": [],
        "head_refine_applied_rois": [],
    }


def _detect_head_rois(image: Image.Image, alpha: np.ndarray) -> tuple[list[Roi], str]:
    rois = _mediapipe_face_rois(image)
    if rois:
        return rois, "mediapipe_facemesh"
    return _heuristic_head_rois(alpha), "heuristic_component_top"


def _mediapipe_face_rois(image: Image.Image) -> list[Roi]:
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return []

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=12,
            refine_landmarks=False,
            min_detection_confidence=0.35,
        )
        result = face_mesh.process(rgb)
        face_mesh.close()
    except Exception:
        return []

    if not result.multi_face_landmarks:
        return []

    rois: list[Roi] = []
    for face in result.multi_face_landmarks:
        xs = np.array([lm.x for lm in face.landmark], dtype=np.float32) * w
        ys = np.array([lm.y for lm in face.landmark], dtype=np.float32) * h
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)

        # Expand face bbox to include hair/ear/neck, but keep it head-local.
        left = int(max(0, x1 - fw * 1.05))
        right = int(min(w, x2 + fw * 1.05))
        top = int(max(0, y1 - fh * 1.45))
        bottom = int(min(h, y2 + fh * 0.65))
        if right - left >= 8 and bottom - top >= 8:
            rois.append(Roi(left, top, right, bottom, "mediapipe_facemesh"))
    return _dedupe_rois(rois)


def _heuristic_head_rois(alpha: np.ndarray) -> list[Roi]:
    fg = alpha > 8
    if not fg.any():
        return []

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    h, w = alpha.shape[:2]
    min_area = max(64, int(h * w * 0.002))
    rois: list[Roi] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if cw <= 0 or ch <= 0:
            continue

        # Top part of each person/component. This supports multi-character sheets
        # and rear/side views where face landmarks are unavailable.
        left = max(0, x - int(cw * 0.16))
        right = min(w, x + cw + int(cw * 0.16))
        top = max(0, y - int(ch * 0.03))
        bottom = min(h, y + int(ch * 0.34))
        if right - left >= 8 and bottom - top >= 8:
            rois.append(Roi(left, top, right, bottom, "heuristic_component_top"))
    return _dedupe_rois(rois)


def _dedupe_rois(rois: list[Roi]) -> list[Roi]:
    if not rois:
        return []
    result: list[Roi] = []
    for roi in sorted(rois, key=lambda r: r.area, reverse=True):
        if any(_roi_iou(roi, existing) > 0.65 for existing in result):
            continue
        result.append(roi)
    return sorted(result, key=lambda r: (r.top, r.left))


def _roi_iou(a: Roi, b: Roi) -> float:
    x1 = max(a.left, b.left)
    y1 = max(a.top, b.top)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, a.area + b.area - inter)
    return float(inter / union)


def _remove_connected_background_in_roi(
    local_rgba: np.ndarray,
    local_alpha: np.ndarray,
    bg_threshold: float,
    max_removed_roi_ratio: float,
) -> np.ndarray | None:
    if local_rgba.size == 0 or local_alpha.size == 0:
        return None

    h, w = local_alpha.shape[:2]
    roi_area = max(1, h * w)
    bg = np.array(estimate_background_rgb(local_rgba, border=max(4, min(24, min(h, w) // 6))), dtype=np.float32)
    rgb = local_rgba[:, :, :3].astype(np.float32)
    distance = np.linalg.norm(rgb - bg[None, None, :], axis=2)

    max_rgb = local_rgba[:, :, :3].max(axis=2).astype(np.int16)
    min_rgb = local_rgba[:, :, :3].min(axis=2).astype(np.int16)
    chroma = max_rgb - min_rgb

    # Prefer background-colored neutral/light pixels. This avoids treating colored
    # hair/skin as removable. It can still remove gray/white studio gaps.
    bg_like = distance <= float(bg_threshold)
    neutral = chroma <= 32
    candidate = bg_like & neutral
    connected = _border_connected(candidate)

    # Only remove pixels that the global model currently keeps. Alpha<=8 is
    # already transparent and does not matter.
    removable = connected & (local_alpha > 8)
    if not removable.any():
        return removable

    # Safety guard: if too much of the head ROI would be removed, skip this ROI.
    # That usually means a white hair ornament, white clothing, or bad ROI.
    removed_ratio = float(removable.sum() / roi_area)
    if removed_ratio > float(max_removed_roi_ratio):
        return None

    # Remove tiny specks, keep connected hair-gap/background paths.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    removable = cv2.morphologyEx(removable.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    return removable


def _border_connected(binary: np.ndarray) -> np.ndarray:
    b = binary.astype(np.uint8)
    if b.size == 0 or b.max() == 0:
        return np.zeros_like(binary, dtype=bool)
    _n, labels = cv2.connectedComponents(b, connectivity=8)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    border = border[border != 0]
    if border.size == 0:
        return np.zeros_like(binary, dtype=bool)
    return np.isin(labels, border)
