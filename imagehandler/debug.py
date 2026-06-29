from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .reports import BBox


def save_mask(mask, path: str | Path) -> None:
    Image.fromarray((mask.astype("uint8") * 255), mode="L").save(path)


def save_boxes_overlay(
    image: Image.Image,
    boxes: list[BBox],
    path: str | Path,
    labels: bool = True,
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    for i, box in enumerate(boxes, 1):
        draw.rectangle(box.to_list(), outline=(255, 0, 0, 255), width=3)
        if labels:
            draw.text((box.left + 4, box.top + 4), str(i), fill=(255, 0, 0, 255))
    overlay.save(path)
