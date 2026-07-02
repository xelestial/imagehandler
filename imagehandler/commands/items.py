from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

import imagehandler.extract_items as base_items_module
from imagehandler.batch import BatchResult, iter_image_files, move_input_to_failed, move_input_to_job_input, relative_output_dir
from imagehandler.extract_items_hybrid import extract_items
from imagehandler.workspace import resolve_output_for_task

from .common import print_batch_result, print_job_paths, print_operation_report

app = typer.Typer(no_args_is_help=True, help="Item extraction menu.")


def _configure_sam(enabled: bool, checkpoint: Optional[Path]) -> None:
    if enabled:
        base_items_module.SAM_MODEL_REL = checkpoint or (Path("models") / "sam_vit_b_01ec64.pth")
    else:
        base_items_module.SAM_MODEL_REL = Path("__sam_disabled__") / "missing.pth"


@app.command("extract")
def extract_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    workspace: Optional[Path] = typer.Option(None),
    job: Optional[str] = typer.Option(None),
    padding: int = typer.Option(16),
    min_area: int = typer.Option(120),
    merge_distance: int = typer.Option(12),
    square_canvas: bool = typer.Option(False),
    normalize_size: Optional[int] = typer.Option(None),
    transparent_bg: bool = typer.Option(False),
    threshold: float = typer.Option(28.0),
    debug: bool = typer.Option(False),
    retry_on_fail: bool = typer.Option(False),
    accept_verdict: str = typer.Option("PASS"),
    min_score: float = typer.Option(85.0),
    min_count: int = typer.Option(1),
    sam_enabled: bool = typer.Option(False),
    sam_checkpoint: Optional[Path] = typer.Option(None),
):
    _configure_sam(sam_enabled, sam_checkpoint)
    out_dir, job_paths = resolve_output_for_task("items", input_path, output_dir, workspace, job)
    report = extract_items(
        input_path=input_path,
        output_dir=out_dir,
        padding=padding,
        min_area=min_area,
        merge_distance=merge_distance,
        square_canvas=square_canvas,
        normalize=normalize_size,
        transparent_bg=transparent_bg,
        threshold=threshold,
        debug=debug,
    )
    print_operation_report(report)
    print_job_paths(job_paths)


@app.command("batch-extract")
def batch_extract_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    workspace: Optional[Path] = typer.Option(None),
    recursive: bool = typer.Option(False),
    pattern: Optional[str] = typer.Option(None),
    padding: int = typer.Option(16),
    min_area: int = typer.Option(120),
    merge_distance: int = typer.Option(12),
    square_canvas: bool = typer.Option(False),
    normalize_size: Optional[int] = typer.Option(None),
    transparent_bg: bool = typer.Option(False),
    threshold: float = typer.Option(28.0),
    debug: bool = typer.Option(False),
    retry_on_fail: bool = typer.Option(False),
    min_count: int = typer.Option(1),
    continue_on_error: bool = typer.Option(True),
    sam_enabled: bool = typer.Option(False),
    sam_checkpoint: Optional[Path] = typer.Option(None),
):
    _configure_sam(sam_enabled, sam_checkpoint)
    files = iter_image_files(input_path, recursive=recursive, pattern=pattern)
    result = BatchResult(operation="items batch-extract", total=len(files))
    root = input_path if input_path.is_dir() else input_path.parent
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        job_paths = None
        if output_dir is None:
            dst_dir, job_paths = resolve_output_for_task("items", src, None, workspace, None)
        else:
            dst_dir = relative_output_dir(src, root, output_dir)
        try:
            extract_items(src, dst_dir, padding=padding, min_area=min_area, merge_distance=merge_distance, square_canvas=square_canvas, normalize=normalize_size, transparent_bg=transparent_bg, threshold=threshold, debug=debug)
            result.succeeded += 1
            result.outputs.append(str(dst_dir))
            if job_paths is not None:
                moved = move_input_to_job_input(src, root, job_paths.input_root)
                if moved is not None:
                    result.moved_to_job_input.append(str(moved))
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{src}: {exc}")
            moved = move_input_to_failed(src, root)
            if moved is not None:
                result.moved_to_failed.append(str(moved))
            if not continue_on_error:
                raise
    print_batch_result(result)
