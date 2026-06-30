#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image, height: int, width: int) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized = resized / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((resized - mean) / std).transpose(2, 0, 1)[None, :, :, :]
    return tensor.astype(np.float32)


def label_preview(labels: np.ndarray) -> Image.Image:
    palette = np.array(
        [
            [0, 0, 0], [255, 220, 177], [255, 128, 0], [255, 160, 0], [0, 180, 255],
            [0, 120, 255], [100, 100, 100], [255, 210, 160], [255, 200, 150], [200, 160, 80],
            [255, 80, 100], [200, 60, 90], [255, 100, 140], [220, 80, 120], [180, 140, 100],
            [120, 100, 80], [80, 120, 255], [40, 40, 40], [120, 80, 180],
        ],
        dtype=np.uint8,
    )
    labels = labels.astype(np.uint8)
    return Image.fromarray(palette[labels % len(palette)], mode="RGB")


def logits_to_labels(output: np.ndarray) -> np.ndarray:
    logits = np.asarray(output)
    if logits.ndim == 4:
        if logits.shape[1] <= 64:
            return logits[0].argmax(axis=0).astype(np.uint8)
        return logits[0].argmax(axis=-1).astype(np.uint8)
    if logits.ndim == 3:
        if logits.shape[0] <= 64:
            return logits.argmax(axis=0).astype(np.uint8)
        return logits.argmax(axis=-1).astype(np.uint8)
    if logits.ndim == 2:
        return logits.astype(np.uint8)
    raise RuntimeError(f"Unsupported ONNX output shape: {logits.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify BiSeNet face parsing ONNX model.")
    parser.add_argument("--model", type=Path, default=Path("models/bisenet_face_parsing.onnx"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=Path("models/bisenet_face_parsing.preview.png"))
    parser.add_argument("--json", type=Path, default=Path("models/bisenet_face_parsing.verify.json"))
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = list(input_meta.shape)
    height = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 512
    width = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 512

    if args.image is not None:
        image = Image.open(args.image)
    else:
        image = Image.new("RGB", (width, height), (255, 255, 255))

    tensor = preprocess(image, height, width)
    outputs = session.run(None, {input_name: tensor})
    labels = logits_to_labels(outputs[0])

    preview = label_preview(labels)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.preview)

    report = {
        "model": str(args.model),
        "sha256": sha256_file(args.model),
        "size_bytes": args.model.stat().st_size,
        "input_name": input_name,
        "input_shape": shape,
        "output_shapes": [list(np.asarray(out).shape) for out in outputs],
        "labels_shape": list(labels.shape),
        "unique_labels": sorted(int(x) for x in np.unique(labels).tolist()),
        "preview": str(args.preview),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
