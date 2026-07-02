from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .debug import save_boxes_overlay, save_mask
from .io import ensure_output_dir, load_image
from .mask_ops import apply_mask_as_alpha, clean_mask, crop_with_padding, crop_mask, foreground_mask_from_background, sort_boxes_reading_order
from .reports import BBox, OperationReport


HUMAN_PARSING_MODEL_REL = Path("models") / "human_parsing.onnx"

# Common LIP/SCHP-style 20-class human parsing labels.
HUMAN_PARSING_LABELS = {
    1: "hat",
    3: "glove",
    5: "upper_clothes",
    6: "dress",
    7: "coat",
    8: "socks",
    9: "pants",
    10: "jumpsuit",
    11: "scarf",
    12: "skirt",
    18: "left_shoe",
    19: "right_shoe",
}


def separate_person_clothing(
    input_path: str | Path,
    output_dir: str | Path,
    transparent_bg: bool = False,
    debug: bool = False,
    min_area: int = 120,
    threshold: float = 28.0,
    model_path: str | Path | None = None,
) -> OperationReport:
    image = load_image(input_path, "RGBA")
    out_dir = ensure_output_dir(output_dir)
    w, h = image.size

    fg_mask = foreground_mask_from_background(image, threshold=threshold)
    fg_mask = clean_mask(fg_mask, open_size=3, close_size=7, fill_holes=True)

    model = Path(model_path).expanduser() if model_path else Path.cwd() / HUMAN_PARSING_MODEL_REL
    parsed = _run_human_parsing_onnx(image, model) if model.is_file() else None

    warnings: list[str] = []
    records: list[tuple[str, np.ndarray, BBox]] = []
    parser_status = "onnx" if parsed is not None else "fallback_region"

    if parsed is not None:
        records = _records_from_label_map(parsed, fg_mask, min_area=min_area)
        if not records:
            warnings.append("Human parsing model produced no usable clothing labels; fallback regions were used.")
            parser_status = "fallback_region"
            records = _fallback_region_records(fg_mask, min_area=min_area)
    else:
        if not model.is_file():
            warnings.append(f"Human parsing ONNX model not found: {model}")
        records = _fallback_region_records(fg_mask, min_area=min_area)

    boxes = sort_boxes_reading_order([box for _name, _mask, box in records])
    outputs: list[str] = []
    final_boxes: list[BBox] = []
    for idx, (name, local_mask, box) in enumerate(records, start=1):
        crop = crop_with_padding(image, box, padding=16)
        if transparent_bg:
            box_mask = crop_mask(local_mask, box, padding=16)
            if box_mask.shape[1] == crop.width and box_mask.shape[0] == crop.height:
                crop = apply_mask_as_alpha(crop, box_mask)
            else:
                crop = crop.convert("RGBA")
        path = out_dir / f"clothing_{idx:03d}_{name}.png"
        crop.save(path)
        outputs.append(str(path))
        final_boxes.append(box)

    coverage = _coverage_metrics(fg_mask, final_boxes)
    if coverage["coverage_ratio"] < 0.70:
        warnings.append("Low clothing coverage; model/fallback may have missed large garment regions.")

    manifest = OperationReport(
        ok=not any("Low clothing coverage" in w for w in warnings),
        operation="person-clothing",
        source=str(input_path),
        mode=parser_status,
        warnings=warnings,
        metrics={
            "items": len(outputs),
            "image_width": w,
            "image_height": h,
            "human_parsing_model": str(model) if model.is_file() else "",
            **coverage,
        },
        boxes=final_boxes,
        outputs=outputs,
    )
    manifest.save(out_dir / "manifest.json")

    if debug:
        save_mask(fg_mask, out_dir / "debug_foreground_mask.png")
        save_boxes_overlay(image, final_boxes, out_dir / "debug_boxes.png")
        if parsed is not None:
            _label_preview(parsed).save(out_dir / "debug_human_parsing_labels.png")

    return manifest


def _run_human_parsing_onnx(image: Image.Image, model_path: Path) -> np.ndarray | None:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        shape = list(input_meta.shape)
        target_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 512
        target_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 512
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((resized - mean) / std).transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        output = session.run(None, {input_name: tensor})[0]
        labels = _logits_to_labels(output)
        return cv2.resize(labels.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
    except Exception:
        return None


def _logits_to_labels(output: Any) -> np.ndarray:
    logits = np.asarray(output)
    if logits.ndim == 4:
        if logits.shape[1] <= 256:
            return logits[0].argmax(axis=0).astype(np.uint8)
        return logits[0].argmax(axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        if logits.shape[0] <= 256:
            return logits.argmax(axis=0).astype(np.uint8)
        return logits.argmax(axis=-1).astype(np.uint8)
    if logits.ndim == 2:
        return logits.astype(np.uint8)
    raise RuntimeError(f"Unsupported human parsing output shape: {logits.shape}")


def _records_from_label_map(labels: np.ndarray, fg_mask: np.ndarray, min_area: int) -> list[tuple[str, np.ndarray, BBox]]:
    records: list[tuple[str, np.ndarray, BBox]] = []
    for label, name in HUMAN_PARSING_LABELS.items():
        component_mask = (labels == label) & fg_mask.astype(bool)
        if int(component_mask.sum()) < min_area:
            continue
        records.extend(_records_from_mask(name, component_mask, min_area=min_area))
    return records


def _fallback_region_records(fg_mask: np.ndarray, min_area: int) -> list[tuple[str, np.ndarray, BBox]]:
    fg = fg_mask.astype(bool)
    if not fg.any():
        return []
    ys, _xs = np.where(fg)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    height = max(1, bottom - top)
    regions = {
        "upper_region": (top, top + int(height * 0.48)),
        "lower_region": (top + int(height * 0.38), top + int(height * 0.82)),
        "shoe_region": (top + int(height * 0.74), bottom),
    }
    records: list[tuple[str, np.ndarray, BBox]] = []
    for name, (y1, y2) in regions.items():
        region = np.zeros_like(fg, dtype=bool)
        region[max(0, y1) : min(fg.shape[0], y2), :] = fg[max(0, y1) : min(fg.shape[0], y2), :]
        records.extend(_records_from_mask(name, region, min_area=max(min_area, int(fg.sum() * 0.015))))
    return records


def _records_from_mask(name: str, mask: np.ndarray, min_area: int) -> list[tuple[str, np.ndarray, BBox]]:
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out: list[tuple[str, np.ndarray, BBox]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        component = labels == idx
        out.append((name, component, BBox(x, y, x + w, y + h)))
    return out


def _coverage_metrics(mask: np.ndarray, boxes: list[BBox]) -> dict[str, float | int]:
    source = int(mask.sum())
    covered = np.zeros(mask.shape, dtype=bool)
    sum_item = 0
    for box in boxes:
        local = mask[box.top : box.bottom, box.left : box.right].astype(bool)
        covered[box.top : box.bottom, box.left : box.right] |= local
        sum_item += int(local.sum())
    covered_count = int((covered & mask.astype(bool)).sum())
    missing = max(0, source - covered_count)
    return {
        "source_foreground_pixels": source,
        "covered_foreground_pixels": covered_count,
        "missing_foreground_pixels": missing,
        "coverage_ratio": float(covered_count / max(1, source)),
        "missing_foreground_ratio": float(missing / max(1, source)),
        "sum_item_foreground_pixels": sum_item,
        "duplication_ratio": float(sum_item / max(1, covered_count)),
    }


def _label_preview(labels: np.ndarray) -> Image.Image:
    palette = np.array(
        [
            [0, 0, 0], [180, 130, 80], [40, 40, 40], [80, 80, 160], [80, 160, 160],
            [220, 60, 60], [220, 120, 60], [180, 80, 180], [80, 180, 220], [80, 120, 220],
            [120, 80, 220], [220, 180, 80], [220, 80, 160], [240, 190, 150], [190, 130, 100],
            [190, 130, 100], [160, 110, 80], [160, 110, 80], [60, 60, 60], [100, 100, 100],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[labels.astype(np.uint8) % len(palette)], mode="RGB")
