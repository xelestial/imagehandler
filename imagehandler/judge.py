from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def judge_image(path: str | Path, task: str = "generic", alpha_required: bool | None = None) -> dict:
    p = Path(path)
    image = Image.open(p).convert("RGBA")
    arr = np.asarray(image)
    alpha = arr[:, :, 3]
    w, h = image.size
    has_alpha = bool((alpha < 248).mean() > 0.01)
    if alpha_required is None:
        alpha_required = task == "remove-bg"
    mask = alpha > 8 if has_alpha else _foreground_mask(arr)
    fg_ratio = float(mask.mean())
    ys, xs = np.where(mask)
    failures = []
    warnings = []
    score = 100.0
    if fg_ratio < 0.002:
        failures.append("Foreground is almost empty.")
        score -= 55
    elif fg_ratio < 0.01:
        warnings.append("Foreground is very small.")
        score -= 25
    if fg_ratio > 0.985:
        failures.append("Foreground covers almost the entire image.")
        score -= 55
    elif fg_ratio > 0.95:
        warnings.append("Foreground covers most of the image.")
        score -= 25
    if alpha_required and not has_alpha:
        failures.append("Expected alpha transparency, but no meaningful alpha was found.")
        score -= 45
    touches_border = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
    if touches_border:
        warnings.append("Foreground touches image border.")
        score -= 10
    bbox = None
    if xs.size and ys.size:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    score = max(0.0, min(100.0, round(score, 2)))
    verdict = "FAIL" if failures or score < 60 else "WARN" if score < 85 else "PASS"
    return {
        "ok": verdict == "PASS",
        "verdict": verdict,
        "score": score,
        "task": task,
        "target": str(p),
        "warnings": warnings,
        "failures": failures,
        "metrics": {
            "width": w,
            "height": h,
            "has_meaningful_alpha": has_alpha,
            "foreground_area_ratio": fg_ratio,
            "transparent_ratio": float((alpha < 8).mean()),
            "semi_alpha_ratio": float(((alpha > 8) & (alpha < 248)).mean()),
            "touches_border": touches_border,
            "bbox": bbox,
        },
    }


def judge_directory(path: str | Path, task: str = "extract-items", expected_count: int | None = None, min_count: int = 1) -> dict:
    folder = Path(path)
    images = [p for p in sorted(folder.glob("*.png")) if not p.name.startswith("debug_") and not p.name.endswith(".mask.png")]
    warnings = []
    failures = []
    score = 100.0
    if expected_count is not None and len(images) != expected_count:
        failures.append(f"Expected {expected_count} output images, found {len(images)}.")
        score -= 40
    elif expected_count is None and len(images) < min_count:
        failures.append(f"Expected at least {min_count} output image(s), found {len(images)}.")
        score -= 40
    children = [judge_image(p, "generic") for p in images]
    if children:
        child_avg = sum(c["score"] for c in children) / len(children)
        if child_avg < 85:
            warnings.append("Average child image score is low.")
            score -= min(30, 85 - child_avg)
    else:
        score -= 40
    score = max(0.0, min(100.0, round(score, 2)))
    verdict = "FAIL" if failures or score < 60 else "WARN" if score < 85 else "PASS"
    return {
        "ok": verdict == "PASS",
        "verdict": verdict,
        "score": score,
        "task": task,
        "target": str(folder),
        "warnings": warnings,
        "failures": failures,
        "metrics": {"image_count": len(images), "expected_count": expected_count, "min_count": min_count},
        "children": children,
    }


def judge_path(target: str | Path, task: str = "auto", expected_count: int | None = None, min_count: int = 1, output: str | Path | None = None, debug: bool = False, alpha_required: bool | None = None) -> dict:
    p = Path(target)
    if task == "auto":
        task = "split-sheet" if p.is_dir() and expected_count == 4 else "extract-items" if p.is_dir() else "remove-bg"
    report = judge_directory(p, task, expected_count, min_count) if p.is_dir() else judge_image(p, task, alpha_required)
    out = Path(output) if output else (p / "judge.json" if p.is_dir() else p.with_name(f"{p.stem}.judge.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _foreground_mask(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    b = max(1, min(24, h // 2, w // 2))
    border = np.concatenate([arr[:b, :, :3].reshape(-1, 3), arr[-b:, :, :3].reshape(-1, 3), arr[:, :b, :3].reshape(-1, 3), arr[:, -b:, :3].reshape(-1, 3)], axis=0).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr[:, :, :3].astype(np.float32) - bg[None, None, :], axis=2)
    return dist > 28.0
