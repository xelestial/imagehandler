from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import numpy as np
from PIL import Image

from .alpha_quality import alpha_quality_metrics
from .debug import save_mask
from .io import ensure_output_dir, load_image
from .mask_ops import (
    bbox_from_mask,
    clean_mask,
    components_bboxes,
    foreground_mask_from_background,
    mask_metrics,
)

TaskName = Literal["auto", "generic", "remove-bg", "split-sheet", "extract-items"]
Verdict = Literal["PASS", "WARN", "FAIL"]


@dataclass
class QualityReport:
    ok: bool
    verdict: Verdict
    score: float
    task: str
    target: str
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    children: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def judge_path(
    target: str | Path,
    task: TaskName = "auto",
    expected_count: int | None = None,
    min_count: int = 1,
    output: str | Path | None = None,
    debug: bool = False,
    alpha_required: bool | None = None,
) -> QualityReport:
    target_path = Path(target)
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    if task == "auto":
        task = _infer_task(target_path, expected_count)

    if target_path.is_dir():
        report = judge_directory(
            target_path,
            task=task,
            expected_count=expected_count,
            min_count=min_count,
            alpha_required=alpha_required,
        )
        output_path = Path(output) if output else target_path / "judge.json"
    else:
        report = judge_image(
            target_path,
            task=task,
            alpha_required=alpha_required,
        )
        output_path = Path(output) if output else target_path.with_name(f"{target_path.stem}.judge.json")
        if debug:
            mask = _mask_for_image(load_image(target_path, "RGBA"), task=task)
            save_mask(mask, target_path.with_name(f"{target_path.stem}.judge_mask.png"))

    report.save(output_path)
    return report


def judge_image(
    image_path: str | Path,
    task: str = "generic",
    alpha_required: bool | None = None,
) -> QualityReport:
    path = Path(image_path)
    image = load_image(path, "RGBA")
    w, h = image.size
    rgba = np.asarray(image)
    alpha = rgba[:, :, 3]

    alpha_min = int(alpha.min())
    alpha_max = int(alpha.max())
    transparent_ratio = float((alpha < 8).mean())
    semi_alpha_ratio = float(((alpha > 8) & (alpha < 248)).mean())
    has_meaningful_alpha = bool((alpha < 248).mean() > 0.01)

    if alpha_required is None:
        alpha_required = task == "remove-bg"

    mask = _mask_for_image(image, task=task)
    # Keep judge morphology small. Heavy cleanup hides real edge/component issues.
    mask = clean_mask(mask, open_size=1, close_size=2, fill_holes=False)
    base_metrics = mask_metrics(mask)
    alpha_metrics = alpha_quality_metrics(image)
    bbox = bbox_from_mask(mask)
    components = components_bboxes(mask, min_area=max(16, int(w * h * 0.00005)), min_size=2)
    component_areas = [area for _box, area in components]
    largest_component_ratio = float(max(component_areas) / max(1, sum(component_areas))) if component_areas else 0.0
    component_count = int(alpha_metrics.get("component_count", len(components)))
    layout_hint = _layout_hint_for_components(component_count, task)

    warnings: list[str] = []
    failures: list[str] = []
    score = 100.0

    fg_ratio = float(base_metrics["foreground_area_ratio"])
    bbox_ratio = float(base_metrics["bbox_area_ratio"])
    touches_border = bool(base_metrics["touches_border"])

    if fg_ratio < 0.002:
        failures.append("Foreground is almost empty.")
        score -= 55
    elif fg_ratio < 0.01:
        warnings.append("Foreground is very small; object may have been erased.")
        score -= 25

    if fg_ratio > 0.985:
        failures.append("Foreground covers almost the entire image.")
        score -= 55
    elif fg_ratio > 0.95:
        warnings.append("Foreground covers most of the image; background removal may have failed.")
        score -= 25

    if bbox_ratio > 0.98 and task == "remove-bg":
        warnings.append("Foreground bounding box is nearly full-canvas.")
        score -= 15

    if touches_border:
        warnings.append("Foreground touches image border; crop may be clipped or padding may be insufficient.")
        score -= 10

    if len(components) == 0:
        failures.append("No connected foreground component was detected.")
        score -= 50
    elif len(components) > 80:
        warnings.append("Too many foreground components; mask may contain noise or sheet extraction may be over-segmented.")
        score -= 15

    if component_areas and largest_component_ratio < 0.45 and task in {"generic", "remove-bg"}:
        if layout_hint == "single_subject_or_unknown":
            warnings.append("Foreground is fragmented into several similarly sized components.")
            score -= 12
        else:
            warnings.append(f"Foreground has {component_count} main components; treated as {layout_hint}.")
            score -= 2

    if alpha_required and not has_meaningful_alpha:
        failures.append("Expected a transparent/alpha result, but the image has no meaningful transparency.")
        score -= 45

    if task == "remove-bg":
        score = _score_alpha_quality(alpha_metrics, warnings, failures, score)
        if transparent_ratio < 0.02:
            warnings.append("Very little transparent background was produced.")
            score -= 20

    margins = _bbox_margins(bbox, w, h) if bbox else None
    if margins:
        min_margin_ratio = min(margins.values())
        if min_margin_ratio < 0.005:
            warnings.append("Detected object is extremely close to at least one canvas edge.")
            score -= 8

    score = max(0.0, min(100.0, round(score, 2)))
    verdict = _verdict(score, failures)

    metrics: dict[str, Any] = {
        "width": w,
        "height": h,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "has_meaningful_alpha": has_meaningful_alpha,
        "transparent_ratio": transparent_ratio,
        "semi_alpha_ratio": semi_alpha_ratio,
        "component_count": len(components),
        "largest_component_ratio": largest_component_ratio,
        "layout_hint": layout_hint,
        **base_metrics,
        "alpha_quality": alpha_metrics,
    }
    if bbox:
        metrics["bbox"] = bbox.to_list()
        metrics["bbox_margin_ratio"] = margins

    return QualityReport(
        ok=verdict == "PASS",
        verdict=verdict,
        score=score,
        task=task,
        target=str(path),
        warnings=warnings,
        failures=failures,
        metrics=metrics,
    )


def _score_alpha_quality(
    metrics: dict[str, Any],
    warnings: list[str],
    failures: list[str],
    score: float,
) -> float:
    unique_levels = int(metrics.get("alpha_unique_levels", 0))
    soft_fg_ratio = float(metrics.get("soft_alpha_fg_ratio", 0.0))
    transparent_leak = float(metrics.get("transparent_rgb_leak_ratio", 0.0))
    jaggedness = float(metrics.get("edge_jaggedness_ratio", 0.0))
    semi_dark = float(metrics.get("semi_dark_rgb_ratio", 0.0))
    semi_light = float(metrics.get("semi_light_rgb_ratio", 0.0))
    edge_band = float(metrics.get("edge_band_ratio", 0.0))

    if transparent_leak > 0.01:
        warnings.append("Transparent pixels contain RGB data; this often creates halos in game engines.")
        score -= 18
    elif transparent_leak > 0.001:
        warnings.append("Small transparent RGB leakage detected.")
        score -= 8

    if unique_levels <= 2:
        warnings.append("Alpha matte is binary; edge quality is likely jagged.")
        score -= 18
    elif unique_levels < 16:
        warnings.append("Alpha matte has very few levels; edge may be too hard.")
        score -= 10

    if soft_fg_ratio < 0.015:
        warnings.append("Very little soft alpha transition; silhouette may look cut out.")
        score -= 8
    elif soft_fg_ratio > 0.45:
        warnings.append("Too much soft alpha transition; result may look blurry or transparent around edges.")
        score -= 8

    if jaggedness > 1.28:
        warnings.append("Severe jagged alpha boundary detected.")
        score -= 14
    elif jaggedness > 1.16:
        warnings.append("Jagged alpha boundary detected.")
        score -= 8

    if semi_dark > 0.35:
        warnings.append("Many semi-transparent edge pixels are very dark; black halo may appear.")
        score -= 8
    if semi_light > 0.55:
        warnings.append("Many semi-transparent edge pixels are very light; white halo may appear.")
        score -= 6

    if edge_band > 0.18:
        warnings.append("Very wide alpha edge band; matte may be over-smoothed.")
        score -= 6

    return score


def judge_directory(
    directory: str | Path,
    task: str = "extract-items",
    expected_count: int | None = None,
    min_count: int = 1,
    alpha_required: bool | None = None,
) -> QualityReport:
    folder = Path(directory)
    images = _result_images(folder)
    warnings: list[str] = []
    failures: list[str] = []
    score = 100.0

    if not images:
        failures.append("No result PNG images were found in the directory.")
        score -= 80

    if expected_count is not None and len(images) != expected_count:
        failures.append(f"Expected {expected_count} output images, found {len(images)}.")
        score -= min(50, abs(len(images) - expected_count) * 15 + 15)
    elif expected_count is None and len(images) < min_count:
        failures.append(f"Expected at least {min_count} output image(s), found {len(images)}.")
        score -= 45

    children: list[dict[str, Any]] = []
    child_scores: list[float] = []
    sizes: list[tuple[int, int]] = []
    fg_ratios: list[float] = []

    for image_path in images:
        child = judge_image(image_path, task="generic", alpha_required=alpha_required)
        children.append(child.to_dict())
        child_scores.append(child.score)
        sizes.append((int(child.metrics["width"]), int(child.metrics["height"])))
        fg_ratios.append(float(child.metrics["foreground_area_ratio"]))

    if child_scores:
        average_child = mean(child_scores)
        if average_child < 85:
            score -= min(35, (85 - average_child) * 0.8)

    if task == "split-sheet" and sizes:
        width_cv = _coefficient_of_variation([s[0] for s in sizes])
        height_cv = _coefficient_of_variation([s[1] for s in sizes])
        if width_cv > 0.35:
            warnings.append("Split view output widths vary heavily; one view may be incorrectly cropped.")
            score -= 15
        if height_cv > 0.20:
            warnings.append("Split view output heights vary heavily; normalization or crop boxes may be inconsistent.")
            score -= 15
    else:
        width_cv = _coefficient_of_variation([s[0] for s in sizes]) if sizes else 0.0
        height_cv = _coefficient_of_variation([s[1] for s in sizes]) if sizes else 0.0

    if task == "extract-items" and fg_ratios:
        tiny_items = sum(1 for r in fg_ratios if r < 0.01)
        full_items = sum(1 for r in fg_ratios if r > 0.95)
        if tiny_items:
            warnings.append(f"{tiny_items} extracted item(s) look nearly empty.")
            score -= min(20, tiny_items * 5)
        if full_items:
            warnings.append(f"{full_items} extracted item(s) have almost no surrounding background/alpha.")
            score -= min(20, full_items * 5)

    manifest_warnings = _read_existing_manifest_warnings(folder)
    if manifest_warnings:
        warnings.extend(f"upstream: {w}" for w in manifest_warnings)
        score -= min(20, len(manifest_warnings) * 5)

    score = max(0.0, min(100.0, round(score, 2)))
    verdict = _verdict(score, failures)

    return QualityReport(
        ok=verdict == "PASS",
        verdict=verdict,
        score=score,
        task=task,
        target=str(folder),
        warnings=warnings,
        failures=failures,
        metrics={
            "image_count": len(images),
            "expected_count": expected_count,
            "min_count": min_count,
            "average_child_score": round(mean(child_scores), 2) if child_scores else 0.0,
            "min_child_score": min(child_scores) if child_scores else 0.0,
            "width_cv": width_cv,
            "height_cv": height_cv,
        },
        children=children,
    )


def _mask_for_image(image: Image.Image, task: str) -> np.ndarray:
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    if bool((alpha < 248).mean() > 0.01):
        return alpha > 8
    return foreground_mask_from_background(image)


def _infer_task(target: Path, expected_count: int | None) -> str:
    if target.is_dir():
        if expected_count == 4:
            return "split-sheet"
        return "extract-items"
    rgba = np.asarray(load_image(target, "RGBA"))
    if bool((rgba[:, :, 3] < 248).mean() > 0.01):
        return "remove-bg"
    return "generic"


def _layout_hint_for_components(component_count: int, task: str) -> str:
    if task == "remove-bg":
        if component_count == 1:
            return "single_subject"
        if 2 <= component_count <= 6:
            return "multi_view_or_multi_subject_sheet"
        if 7 <= component_count <= 12:
            return "large_multi_subject_sheet"
    return "single_subject_or_unknown"


def _result_images(folder: Path) -> list[Path]:
    ignored_prefixes = ("debug_",)
    ignored_suffixes = (
        ".mask.png",
        ".judge_mask.png",
        ".preview.white.png",
        ".preview.black.png",
        ".preview.checker.png",
    )
    images: list[Path] = []
    for p in sorted(folder.glob("*.png")):
        name = p.name
        if name.startswith(ignored_prefixes):
            continue
        if any(name.endswith(s) for s in ignored_suffixes):
            continue
        if name in {"atlas.png"}:
            continue
        images.append(p)
    return images


def _bbox_margins(bbox, width: int, height: int) -> dict[str, float]:
    return {
        "left": float(bbox.left / max(1, width)),
        "top": float(bbox.top / max(1, height)),
        "right": float((width - bbox.right) / max(1, width)),
        "bottom": float((height - bbox.bottom) / max(1, height)),
    }


def _coefficient_of_variation(values: list[int]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    if avg == 0:
        return 0.0
    return float(pstdev(values) / avg)


def _read_existing_manifest_warnings(folder: Path) -> list[str]:
    warnings: list[str] = []
    for name in ("manifest.json", "report.json"):
        path = folder / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw = data.get("warnings")
        if isinstance(raw, list):
            warnings.extend(str(x) for x in raw)
    return warnings


def _verdict(score: float, failures: list[str]) -> Verdict:
    if failures or score < 60:
        return "FAIL"
    if score < 90:
        return "WARN"
    return "PASS"
