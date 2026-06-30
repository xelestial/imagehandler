from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .bg_remove import DEFAULT_REMBG_MODEL, normalize_rembg_model, remove_background
from .extract_items import extract_items
from .judge import QualityReport, judge_directory, judge_image
from .reports import OperationReport
from .split_sheet import split_sheet


VERDICT_RANK = {"FAIL": 0, "WARN": 1, "PASS": 2}
REMBG_MODEL_FALLBACKS = [
    "birefnet-general",
    "birefnet-portrait",
    "isnet-anime",
    "isnet-general-use",
]
PREVIEW_SUFFIXES = (
    ".preview.white.png",
    ".preview.black.png",
    ".preview.checker.png",
)


@dataclass
class AttemptRecord:
    name: str
    verdict: str
    score: float
    accepted: bool
    target: str
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "score": self.score,
            "accepted": self.accepted,
            "target": self.target,
            "warnings": self.warnings,
            "failures": self.failures,
            "details": self.details,
        }


@dataclass
class FallbackSummary:
    operation: str
    source: str
    target: str
    selected_attempt: str
    selected_verdict: str
    selected_score: float
    accepted: bool
    attempts: list[AttemptRecord]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "operation": self.operation,
                    "source": self.source,
                    "target": self.target,
                    "selected_attempt": self.selected_attempt,
                    "selected_verdict": self.selected_verdict,
                    "selected_score": self.selected_score,
                    "accepted": self.accepted,
                    "attempts": [a.to_dict() for a in self.attempts],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _is_acceptable(report: QualityReport, accept_verdict: str, min_score: float) -> bool:
    return (
        VERDICT_RANK.get(report.verdict, 0) >= VERDICT_RANK.get(accept_verdict, 2)
        and report.score >= min_score
    )


def _best_attempt(attempts: list[tuple[AttemptRecord, QualityReport, Any, Path]]):
    return max(attempts, key=lambda item: (VERDICT_RANK.get(item[1].verdict, 0), item[1].score))


def _attempt_name(backend: str, cfg: dict[str, Any]) -> str:
    if backend == "rembg":
        return f"rembg:{cfg.get('model') or DEFAULT_REMBG_MODEL}"
    return backend


def _append_unique(candidates: list[tuple[str, dict[str, Any]]], backend: str, cfg: dict[str, Any]) -> None:
    if not any(
        existing_backend == backend and existing_cfg == cfg
        for existing_backend, existing_cfg in candidates
    ):
        candidates.append((backend, cfg))


def _background_candidates(
    backend: str, model: str | None, alpha_matting: bool
) -> list[tuple[str, dict[str, Any]]]:
    backend = backend.lower()
    candidates: list[tuple[str, dict[str, Any]]] = []

    if backend == "auto":
        first_model = normalize_rembg_model(model)
        _append_unique(candidates, "rembg", {"model": first_model, "alpha_matting": bool(alpha_matting)})
        for candidate_model in REMBG_MODEL_FALLBACKS:
            _append_unique(candidates, "rembg", {"model": candidate_model, "alpha_matting": False})
        _append_unique(candidates, "transparent", {})
        _append_unique(candidates, "classical", {})
        return candidates

    if backend == "rembg":
        first_model = normalize_rembg_model(model)
        _append_unique(candidates, "rembg", {"model": first_model, "alpha_matting": bool(alpha_matting)})
        for candidate_model in REMBG_MODEL_FALLBACKS:
            _append_unique(candidates, "rembg", {"model": candidate_model, "alpha_matting": False})
        return candidates

    _append_unique(candidates, backend, {})
    if backend != "transparent":
        _append_unique(candidates, "transparent", {})
    if backend != "classical":
        _append_unique(candidates, "classical", {})
    return candidates


def _copy_bg_sidecars(selected_output: Path, output_path: Path) -> list[str]:
    outputs = [str(output_path)]

    selected_mask = selected_output.with_name(f"{selected_output.stem}.mask.png")
    final_mask = output_path.with_name(f"{output_path.stem}.mask.png")
    if selected_mask.exists():
        shutil.copy2(selected_mask, final_mask)
        outputs.append(str(final_mask))

    for suffix in PREVIEW_SUFFIXES:
        selected_preview = selected_output.with_name(f"{selected_output.stem}{suffix}")
        final_preview = output_path.with_name(f"{output_path.stem}{suffix}")
        if selected_preview.exists():
            shutil.copy2(selected_preview, final_preview)
            outputs.append(str(final_preview))

    return outputs


def remove_background_with_fallback(
    input_path: str | Path,
    output_path: str | Path,
    backend: str = "auto",
    model: str | None = None,
    alpha_matting: bool = False,
    mask_only: bool = False,
    postprocess: bool = True,
    feather: float = 0.0,
    accept_verdict: str = "PASS",
    min_score: float = 85.0,
) -> tuple[OperationReport, FallbackSummary]:
    output_path = Path(output_path)
    candidates = _background_candidates(backend, model, alpha_matting)

    attempts: list[tuple[AttemptRecord, QualityReport, OperationReport, Path]] = []
    selected_idx = 0

    with TemporaryDirectory(prefix="imagehandler_bg_retry_") as tmpdir:
        tmp_root = Path(tmpdir)
        for idx, (candidate_backend, cfg) in enumerate(candidates, start=1):
            name = _attempt_name(candidate_backend, cfg)
            safe_name = name.replace(":", "_").replace("/", "_")
            attempt_output = tmp_root / f"attempt_{idx:02d}_{safe_name}.png"
            try:
                report = remove_background(
                    input_path=input_path,
                    output_path=attempt_output,
                    backend=candidate_backend,
                    model=cfg.get("model"),
                    alpha_matting=cfg.get("alpha_matting", False),
                    mask_only=mask_only,
                    postprocess=postprocess,
                    feather=feather,
                    cleanup_mode="safe",
                )
                judge = judge_image(attempt_output, task="remove-bg", alpha_required=not mask_only)
            except Exception as exc:
                judge = QualityReport(
                    ok=False,
                    verdict="FAIL",
                    score=0.0,
                    task="remove-bg",
                    target=str(attempt_output),
                    failures=[f"attempt runtime error: {exc}"],
                )
                report = OperationReport(ok=False, operation="remove-bg", source=str(input_path), backend=name)

            accepted = _is_acceptable(judge, accept_verdict, min_score)
            record = AttemptRecord(
                name=name,
                verdict=judge.verdict,
                score=judge.score,
                accepted=accepted,
                target=str(attempt_output),
                warnings=list(judge.warnings),
                failures=list(judge.failures),
                details={"backend": candidate_backend, **cfg},
            )
            attempts.append((record, judge, report, attempt_output))
            if accepted:
                selected_idx = len(attempts) - 1
                break

        if not attempts:
            raise RuntimeError("No fallback attempt was executed.")

        if not attempts[selected_idx][0].accepted:
            best = _best_attempt(attempts)
            selected_idx = attempts.index(best)

        selected_record, selected_judge, selected_report, selected_output = attempts[selected_idx]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_output, output_path)
        copied_outputs = _copy_bg_sidecars(selected_output, output_path)

        selected_report_path = selected_output.with_name(f"{selected_output.stem}.report.json")
        final_report_path = output_path.with_name(f"{output_path.stem}.report.json")
        if selected_report_path.exists():
            shutil.copy2(selected_report_path, final_report_path)

        selected_report.outputs = copied_outputs
        selected_report.source = str(input_path)
        selected_report.backend = selected_record.name
        selected_report.ok = selected_judge.verdict in {"PASS", "WARN"}
        selected_report.save(output_path.with_name(f"{output_path.stem}.report.json"))
        summary = FallbackSummary(
            operation="remove-bg",
            source=str(input_path),
            target=str(output_path),
            selected_attempt=selected_record.name,
            selected_verdict=selected_judge.verdict,
            selected_score=selected_judge.score,
            accepted=selected_record.accepted,
            attempts=[a[0] for a in attempts],
        )
        summary.save(output_path.with_name(f"{output_path.stem}.fallback.json"))
        return selected_report, summary


def split_sheet_with_retry(
    input_path: str | Path,
    output_dir: str | Path,
    views: int = 4,
    padding: int = 24,
    min_area: int = 1000,
    merge_distance: int = 24,
    normalize: int | None = None,
    threshold: float = 28.0,
    debug: bool = False,
    accept_verdict: str = "PASS",
    min_score: float = 85.0,
) -> tuple[OperationReport, FallbackSummary]:
    output_dir = Path(output_dir)
    candidates = [
        {"threshold": threshold, "min_area": min_area, "merge_distance": merge_distance},
        {"threshold": max(8.0, threshold - 8.0), "min_area": max(64, min_area // 2), "merge_distance": merge_distance},
        {"threshold": threshold + 8.0, "min_area": min_area, "merge_distance": merge_distance + 12},
        {"threshold": threshold, "min_area": max(64, min_area // 3), "merge_distance": merge_distance + 24},
    ]

    attempts: list[tuple[AttemptRecord, QualityReport, OperationReport, Path]] = []
    selected_idx = 0
    with TemporaryDirectory(prefix="imagehandler_split_retry_") as tmpdir:
        tmp_root = Path(tmpdir)
        for idx, cfg in enumerate(candidates, start=1):
            attempt_dir = tmp_root / f"attempt_{idx:02d}"
            report = split_sheet(
                input_path=input_path,
                output_dir=attempt_dir,
                views=views,
                padding=padding,
                min_area=cfg["min_area"],
                merge_distance=cfg["merge_distance"],
                normalize=normalize,
                threshold=cfg["threshold"],
                debug=debug,
            )
            judge = judge_directory(attempt_dir, task="split-sheet", expected_count=views)
            accepted = _is_acceptable(judge, accept_verdict, min_score)
            record = AttemptRecord(
                name=f"strategy:{idx}",
                verdict=judge.verdict,
                score=judge.score,
                accepted=accepted,
                target=str(attempt_dir),
                warnings=list(judge.warnings),
                failures=list(judge.failures),
                details=cfg,
            )
            attempts.append((record, judge, report, attempt_dir))
            if accepted:
                selected_idx = len(attempts) - 1
                break

        best = _best_attempt(attempts)
        if not attempts[selected_idx][0].accepted:
            selected_idx = attempts.index(best)

        selected_record, selected_judge, selected_report, selected_dir = attempts[selected_idx]
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in output_dir.glob("*"):
            if item.is_file():
                item.unlink()
        for src in selected_dir.glob("*"):
            if src.is_file():
                shutil.copy2(src, output_dir / src.name)

        selected_report.outputs = [str(output_dir / src.name) for src in sorted(selected_dir.glob("view_*.png"))]
        selected_report.source = str(input_path)
        selected_report.save(output_dir / "manifest.json")
        summary = FallbackSummary(
            operation="split-sheet",
            source=str(input_path),
            target=str(output_dir),
            selected_attempt=selected_record.name,
            selected_verdict=selected_judge.verdict,
            selected_score=selected_judge.score,
            accepted=selected_record.accepted,
            attempts=[a[0] for a in attempts],
        )
        summary.save(output_dir / "fallback.json")
        return selected_report, summary


def extract_items_with_retry(
    input_path: str | Path,
    output_dir: str | Path,
    padding: int = 16,
    min_area: int = 120,
    merge_distance: int = 12,
    square_canvas: bool = False,
    normalize: int | None = None,
    transparent_bg: bool = False,
    threshold: float = 28.0,
    debug: bool = False,
    accept_verdict: str = "PASS",
    min_score: float = 85.0,
    min_count: int = 1,
) -> tuple[OperationReport, FallbackSummary]:
    output_dir = Path(output_dir)
    candidates = [
        {"threshold": threshold, "min_area": min_area, "merge_distance": merge_distance},
        {"threshold": max(8.0, threshold - 8.0), "min_area": max(32, min_area // 2), "merge_distance": merge_distance},
        {"threshold": threshold + 8.0, "min_area": min_area, "merge_distance": merge_distance * 2},
        {"threshold": threshold, "min_area": max(16, min_area // 4), "merge_distance": merge_distance + 16},
    ]

    attempts: list[tuple[AttemptRecord, QualityReport, OperationReport, Path]] = []
    selected_idx = 0
    with TemporaryDirectory(prefix="imagehandler_extract_retry_") as tmpdir:
        tmp_root = Path(tmpdir)
        for idx, cfg in enumerate(candidates, start=1):
            attempt_dir = tmp_root / f"attempt_{idx:02d}"
            report = extract_items(
                input_path=input_path,
                output_dir=attempt_dir,
                padding=padding,
                min_area=cfg["min_area"],
                merge_distance=cfg["merge_distance"],
                square_canvas=square_canvas,
                normalize=normalize,
                transparent_bg=transparent_bg,
                threshold=cfg["threshold"],
                debug=debug,
            )
            judge = judge_directory(attempt_dir, task="extract-items", min_count=min_count, alpha_required=transparent_bg)
            accepted = _is_acceptable(judge, accept_verdict, min_score)
            record = AttemptRecord(
                name=f"strategy:{idx}",
                verdict=judge.verdict,
                score=judge.score,
                accepted=accepted,
                target=str(attempt_dir),
                warnings=list(judge.warnings),
                failures=list(judge.failures),
                details=cfg,
            )
            attempts.append((record, judge, report, attempt_dir))
            if accepted:
                selected_idx = len(attempts) - 1
                break

        best = _best_attempt(attempts)
        if not attempts[selected_idx][0].accepted:
            selected_idx = attempts.index(best)

        selected_record, selected_judge, selected_report, selected_dir = attempts[selected_idx]
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in output_dir.glob("*"):
            if item.is_file():
                item.unlink()
        for src in selected_dir.glob("*"):
            if src.is_file():
                shutil.copy2(src, output_dir / src.name)

        selected_report.outputs = [str(output_dir / src.name) for src in sorted(selected_dir.glob("item_*.png"))]
        selected_report.source = str(input_path)
        selected_report.save(output_dir / "manifest.json")
        summary = FallbackSummary(
            operation="extract-items",
            source=str(input_path),
            target=str(output_dir),
            selected_attempt=selected_record.name,
            selected_verdict=selected_judge.verdict,
            selected_score=selected_judge.score,
            accepted=selected_record.accepted,
            attempts=[a[0] for a in attempts],
        )
        summary.save(output_dir / "fallback.json")
        return selected_report, summary
