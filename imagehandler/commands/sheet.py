from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from imagehandler.batch import (
    BatchResult,
    iter_image_files,
    move_input_to_failed,
    move_input_to_job_input,
    relative_output_dir,
)
from imagehandler.fallback import split_sheet_with_fallback
from imagehandler.split_sheet import split_sheet
from imagehandler.workspace import resolve_output_for_task

from .common import print_batch_result, print_fallback_summary, print_job_paths, print_operation_report

app = typer.Typer(no_args_is_help=True, help="Character sheet splitting menu.")


@app.command("split")
def split_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory. If omitted, use workspace/sheets/jobs/<job>/output/."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    job: Optional[str] = typer.Option(None, help="Optional job folder name. If omitted, use the input filename."),
    views: int = typer.Option(4, help="Number of expected views."),
    padding: int = typer.Option(24),
    min_area: int = typer.Option(1000),
    merge_distance: int = typer.Option(24),
    normalize_size: Optional[int] = typer.Option(None),
    threshold: float = typer.Option(28.0),
    debug: bool = typer.Option(False),
    retry_on_fail: bool = typer.Option(False),
    accept_verdict: str = typer.Option("PASS"),
    min_score: float = typer.Option(85.0),
):
    out_dir, job_paths = resolve_output_for_task("sheets", input_path, output_dir, workspace, job)
    if retry_on_fail:
        report, summary = split_sheet_with_fallback(
            input_path=input_path,
            output_dir=out_dir,
            views=views,
            min_score=min_score,
        )
        print_operation_report(report)
        print_fallback_summary(summary)
        print_job_paths(job_paths)
        return
    report = split_sheet(
        input_path=input_path,
        output_dir=out_dir,
        views=views,
        padding=padding,
        min_area=min_area,
        merge_distance=merge_distance,
        normalize=normalize_size,
        threshold=threshold,
        debug=debug,
    )
    print_operation_report(report)
    print_job_paths(job_paths)


@app.command("batch-split")
def batch_split_cmd(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory. If omitted, use workspace/sheets/jobs/<job>/output/ per file."),
    workspace: Optional[Path] = typer.Option(None, help="Workspace root used when --output is omitted. Default: ./workspace"),
    recursive: bool = typer.Option(False),
    pattern: Optional[str] = typer.Option(None),
    views: int = typer.Option(4),
    padding: int = typer.Option(24),
    min_area: int = typer.Option(1000),
    merge_distance: int = typer.Option(24),
    normalize_size: Optional[int] = typer.Option(None),
    threshold: float = typer.Option(28.0),
    debug: bool = typer.Option(False),
    retry_on_fail: bool = typer.Option(False),
    continue_on_error: bool = typer.Option(True),
):
    files = iter_image_files(input_path, recursive=recursive, pattern=pattern)
    result = BatchResult(operation="sheet batch-split", total=len(files))
    root = input_path if input_path.is_dir() else input_path.parent
    output_jobs = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        job_paths = None
        if output_dir is None:
            dst_dir, job_paths = resolve_output_for_task("sheets", src, None, workspace, None)
            output_jobs.append(str(job_paths.output_root) if job_paths else str(dst_dir))
        else:
            dst_dir = relative_output_dir(src, root, output_dir)
        try:
            if retry_on_fail:
                split_sheet_with_fallback(src, dst_dir, views=views, min_score=85.0)
            else:
                split_sheet(src, dst_dir, views=views, padding=padding, min_area=min_area, merge_distance=merge_distance, normalize=normalize_size, threshold=threshold, debug=debug)
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
    for item in output_jobs:
        typer.echo(f"output job folder: {item}")
