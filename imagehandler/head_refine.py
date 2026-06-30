from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .mask_ops import estimate_background_rgb, pil_to_rgba_array


# Common 19-class BiSeNet face-parsing label ids.
# 0 background, 1 skin, 2/3 brows, 4/5 eyes, 6 glasses, 7/8 ears,
# 9 earrings, 10 nose, 11 mouth, 12/13 lips, 14 neck, 15 neck_l,
# 16 cloth, 17 hair, 18 hat.
_BISENET_HEAD_KEEP_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18}
_BISENET_BG_LABEL = 0


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
    max_removed_roi_ratio: float = 0.035,
    bisenet_onnx_path: str | Path | None = None,
) -> HeadRefineResult:
    """Refine only head/face regions, preferring face-specific models.

    Priority:
      1. BiSeNet face parsing ONNX, when a model path is configured and
         MediaPipe FaceMesh provides a face/head ROI.
      2. MediaPipe FaceMesh ROI with conservative connected-background cleanup.
      3. Heuristic component-top ROI is metrics-only and never deletes alpha.

    This deliberately avoids destructive cleanup for heuristic ROIs. Previous
    heuristic fallback could enter shoulders/sleeves on multi-character sheets
    and punch holes in white clothing.
    """
    a = np.asarray(alpha).astype(np.uint8).copy()
    if not enabled or a.size == 0 or a.max() == 0:
        return HeadRefineResult(a, _empty_metrics(enabled=enabled, detector="disabled"))

    rgba = pil_to_rgba_array(image)
    h, w = a.shape[:2]
    rois, detector = _detect_head_rois(image, a)
    parser = _load_bisenet_session(bisenet_onnx_path)
    parser_state = "bisenet_onnx" if parser is not None else "none"

    total_removed = 0
    skipped = 0
    parser_applied = 0
    mediapipe_cleanup_applied = 0
    heuristic_metrics_only = 0
    applied_rois: list[list[int | str]] = []

    for roi in rois:
        if roi.area <= 0:
            continue

        # Critical safety rule: heuristic ROI is not allowed to delete pixels.
        # It is too broad for multi-character costume sheets and can damage
        # sleeves, collars, capes, and white ornaments.
        if roi.method != "mediapipe_facemesh":
            heuristic_metrics_only += 1
            continue

        local_alpha = a[roi.top : roi.bottom, roi.left : roi.right]
        local_rgba = rgba[roi.top : roi.bottom, roi.left : roi.right]

        removed: np.ndarray | None = None
        if parser is not None:
            removed = _remove_bisenet_background_in_roi(
                parser=parser,
                local_rgba=local_rgba,
                local_alpha=local_alpha,
                bg_threshold=bg_threshold,
                max_removed_roi_ratio=max_removed_roi_ratio,
            )
            if removed is not None and removed.any():
                parser_applied += 1

        if removed is None:
            removed = _remove_connected_background_in_roi(
                local_rgba=local_rgba,
                local_alpha=local_alpha,
                bg_threshold=bg_threshold,
                max_removed_roi_ratio=max_removed_roi_ratio,
            )
            if removed is None:
                skipped += 1
                continue
            if removed.any():
                mediapipe_cleanup_applied += 1

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
            "head_refine_parser": parser_state,
            "head_refine_bisenet_model": str(_resolve_bisenet_path(bisenet_onnx_path) or ""),
            "head_refine_roi_count": len(rois),
            "head_refine_applied_roi_count": len(applied_rois),
            "head_refine_skipped_roi_count": skipped,
            "head_refine_parser_applied_roi_count": parser_applied,
            "head_refine_mediapipe_cleanup_applied_roi_count": mediapipe_cleanup_applied,
            "head_refine_heuristic_metrics_only_roi_count": heuristic_metrics_only,
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
        "head_refine_parser": "none",
        "head_refine_bisenet_model": "",
        "head_refine_roi_count": 0,
        "head_refine_applied_roi_count": 0,
        "head_refine_skipped_roi_count": 0,
        "head_refine_parser_applied_roi_count": 0,
        "head_refine_mediapipe_cleanup_applied_roi_count": 0,
        "head_refine_heuristic_metrics_only_roi_count": 0,
        "head_refine_removed_pixels": 0,
        "head_refine_removed_ratio": 0.0,
        "head_refine_rois": [],
        "head_refine_applied_rois": [],
    }


def _detect_head_rois(image: Image.Image, alpha: np.ndarray) -> tuple[list[Roi], str]:
    rois = _mediapipe_face_rois(image)
    if rois:
        return rois, "mediapipe_facemesh"
    # Fallback is kept for diagnostics only. It must not delete alpha.
    return _heuristic_head_rois(alpha), "heuristic_component_top_metrics_only"


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
            max_num_faces=16,
            refine_landmarks=False,
            min_detection_confidence=0.30,
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

        # Narrower than the previous version. The bottom is intentionally only a
        # small extension below the face to avoid shoulders, sleeves, and chest.
        left = int(max(0, x1 - fw * 0.85))
        right = int(min(w, x2 + fw * 0.85))
        top = int(max(0, y1 - fh * 1.30))
        bottom = int(min(h, y2 + fh * 0.22))
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

        left = max(0, x - int(cw * 0.10))
        right = min(w, x + cw + int(cw * 0.10))
        top = max(0, y - int(ch * 0.02))
        bottom = min(h, y + int(ch * 0.24))
        if right - left >= 8 and bottom - top >= 8:
            rois.append(Roi(left, top, right, bottom, "heuristic_component_top_metrics_only"))
    return _dedupe_rois(rois)


def _resolve_bisenet_path(path: str | Path | None) -> Path | None:
    raw = str(path or os.environ.get("IMAGEHANDLER_BISENET_ONNX", "")).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() else None


def _load_bisenet_session(path: str | Path | None):
    model_path = _resolve_bisenet_path(path)
    if model_path is None:
        return None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None
    try:
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception:
        return None


def _remove_bisenet_background_in_roi(
    parser: Any,
    local_rgba: np.ndarray,
    local_alpha: np.ndarray,
    bg_threshold: float,
    max_removed_roi_ratio: float,
) -> np.ndarray | None:
    label_map = _run_bisenet(parser, local_rgba)
    if label_map is None:
        return None

    h, w = local_alpha.shape[:2]
    if label_map.shape[:2] != (h, w):
        label_map = cv2.resize(label_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    keep = np.isin(label_map, list(_BISENET_HEAD_KEEP_LABELS))
    bg = label_map == _BISENET_BG_LABEL
    if keep.sum() < max(16, int(h * w * 0.015)):
        return None

    bg_like = _background_like_neutral(local_rgba, bg_threshold=bg_threshold)
    connected = _border_connected(bg & bg_like)
    removable = connected & (local_alpha > 8)
    if not removable.any():
        return removable

    removed_ratio = float(removable.sum() / max(1, h * w))
    if removed_ratio > float(max_removed_roi_ratio):
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(removable.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0


def _run_bisenet(parser: Any, local_rgba: np.ndarray) -> np.ndarray | None:
    try:
        input_meta = parser.get_inputs()[0]
        input_name = input_meta.name
        shape = list(input_meta.shape)
        # Most exported BiSeNet face-parsing ONNX models use NCHW.
        target_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 512
        target_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 512

        rgb = local_rgba[:, :, :3].astype(np.uint8)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        chw = ((resized - mean) / std).transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        output = parser.run(None, {input_name: chw})[0]
        logits = np.asarray(output)
        if logits.ndim == 4:
            # N,C,H,W or N,H,W,C
            if logits.shape[1] <= 64:
                labels = logits[0].argmax(axis=0)
            else:
                labels = logits[0].argmax(axis=-1)
        elif logits.ndim == 3:
            labels = logits.argmax(axis=0) if logits.shape[0] <= 64 else logits.argmax(axis=-1)
        elif logits.ndim == 2:
            labels = logits
        else:
            return None
        h, w = local_rgba.shape[:2]
        return cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        return None


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
    candidate = _background_like_neutral(local_rgba, bg_threshold=bg_threshold)
    connected = _border_connected(candidate)
    removable = connected & (local_alpha > 8)
    if not removable.any():
        return removable

    removed_ratio = float(removable.sum() / roi_area)
    if removed_ratio > float(max_removed_roi_ratio):
        return None

    # Additional lower-half guard. Hair side gaps are mostly upper/mid ROI;
    # this protects neck ornaments and collars when FaceMesh bbox is imperfect.
    yy = np.arange(h)[:, None]
    removable = removable & (yy <= int(h * 0.82))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    removable = cv2.morphologyEx(removable.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    return removable


def _background_like_neutral(local_rgba: np.ndarray, bg_threshold: float) -> np.ndarray:
    h, w = local_rgba.shape[:2]
    bg = np.array(estimate_background_rgb(local_rgba, border=max(4, min(24, min(h, w) // 6))), dtype=np.float32)
    rgb = local_rgba[:, :, :3].astype(np.float32)
    distance = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    max_rgb = local_rgba[:, :, :3].max(axis=2).astype(np.int16)
    min_rgb = local_rgba[:, :, :3].min(axis=2).astype(np.int16)
    chroma = max_rgb - min_rgb
    return (distance <= float(bg_threshold)) & (chroma <= 30)


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
