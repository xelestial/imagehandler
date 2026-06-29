from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from imagehandler.batch import BatchResult, iter_image_files, move_input_to_complete, relative_output_dir
from imagehandler.extract_items import extract_items
from imagehandler.fallback import extract_items_with_retry
from imagehandler.workspace import resolve_output_for_task

from .common import print_batch_result, print_fallback_summary, print_job_paths, print_operation_report

app = typer.Typer(no_args_is_help=True, help="Item and equipment sheet extraction menu.")


@app.command("extract")
def extract_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory. If omitted, auto-create a job folder under workspace/jobs/."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    job: Optional[str] = typer.Option(None, help="Optional job folder name. If omitted, use the input filename. If the name already exists, a date suffix is appended."),
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
):
    out_dir, job_paths = resolve_output_for_task("items", input_path, output_dir, workspace, job)
    if retry_on_fail:
        report, summary = extract_items_with_retry(
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
            accept_verdict=accept_verdict,
            min_score=min_score,
            min_count=min_count,
        )
        print_operation_report(report)
        print_fallback_summary(summary)
        print_job_paths(job_paths)
        return
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
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory. If omitted, auto-create one job folder per file under workspace/jobs/."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
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
):
    files = iter_image_files(input_path, recursive=recursive, pattern=pattern)
    result = BatchResult(operation="items batch-extract", total=len(files))
    root = input_path if input_path.is_dir() else input_path.parent
    auto_job_outputs = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        if output_dir is None:
            dst_dir, job_paths = resolve_output_for_task("items", src, None, workspace, None)
            auto_job_outputs.append(str(job_paths.job_root) if job_paths else str(dst_dir.parent))
        else:
            dst_dir = relative_output_dir(src, root, output_dir)
        try:
            if retry_on_fail:
                extract_items_with_retry(src, dst_dir, padding=padding, min_area=min_area, merge_distance=merge_distance, square_canvas=square_canvas, normalize=normalize_size, transparent_bg=transparent_bg, threshold=threshold, debug=debug, min_count=min_count)
            else:
                extract_items(src, dst_dir, padding=padding, min_area=min_area, merge_distance=merge_distance, square_canvas=square_canvas, normalize=normalize_size, transparent_bg=transparent_bg, threshold=threshold, debug=debug)
            result.succeeded += 1
            result.outputs.append(str(dst_dir))
            moved = move_input_to_complete(src, root)
            if moved is not None:
                result.moved_to_complete.append(str(moved))
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{src}: {exc}")
            if not continue_on_error:
                raise
    print_batch_result(result)
    if auto_job_outputs:
        for item in auto_job_outputs:
            typer.echo(f"job folder: {item}")
